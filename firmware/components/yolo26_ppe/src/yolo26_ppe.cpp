#include "yolo26_ppe.hpp"

#include "dl_math.hpp"
#include "dl_tool.hpp"
#include "esp_log.h"
#include "esp_timer.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ppe {

static const char *TAG = "yolo26_ppe";
constexpr int YOLO26PPE::kStrides[3];

const char *YOLO26PPE::class_name(int class_id)
{
    switch (class_id) {
    case CLASS_HELMET:
        return "helmet";
    case CLASS_NO_HELMET:
        return "no_helmet";
    case CLASS_PERSON:
        return "person";
    default:
        return "unknown";
    }
}

YOLO26PPE::YOLO26PPE(dl::Model *model, int top_k, float conf_thresh)
    : m_model(model),
      m_preprocessor(nullptr),
      m_input_w(512),
      m_input_h(512),
      m_top_k(top_k),
      m_conf_thresh(conf_thresh),
      m_grid_h{0, 0, 0},
      m_grid_w{0, 0, 0}
{
    m_preprocessor = new dl::image::ImagePreprocessor(model, {0, 0, 0}, {255, 255, 255});
    m_preprocessor->enable_letterbox({114, 114, 114});

    auto inputs = model->get_inputs();
    if (!inputs.empty()) {
        dl::TensorBase *in_tensor = inputs.begin()->second;
        // ESP-DL NHWC shape: [1, H, W, C]
        m_input_h = in_tensor->shape[1];
        m_input_w = in_tensor->shape[2];
        for (int i = 0; i < 3; i++) {
            m_grid_h[i] = m_input_h / kStrides[i];
            m_grid_w[i] = m_input_w / kStrides[i];
        }
        ESP_LOGI(TAG,
                 "Initialized YOLO26PPE input=%dx%d grids=[%dx%d, %dx%d, %dx%d]",
                 m_input_w,
                 m_input_h,
                 m_grid_w[0],
                 m_grid_h[0],
                 m_grid_w[1],
                 m_grid_h[1],
                 m_grid_w[2],
                 m_grid_h[2]);
    }
}

YOLO26PPE::~YOLO26PPE()
{
    delete m_preprocessor;
}

template <typename BoxT, typename ClsT>
void YOLO26PPE::decode_grid(dl::TensorBase *p_box,
                            dl::TensorBase *p_cls,
                            int stride,
                            int grid_h,
                            int grid_w,
                            int num_classes,
                            std::vector<dl::detect::result_t> &candidates)
{
    const float box_scale = DL_SCALE(p_box->exponent);
    const float cls_scale = DL_SCALE(p_cls->exponent);
    const BoxT *raw_box = static_cast<const BoxT *>(p_box->data);
    const ClsT *raw_cls = static_cast<const ClsT *>(p_cls->data);

    const float raw_thresh_f = dl::math::inverse_sigmoid(m_conf_thresh);
    const float scaled_thresh = std::floor(raw_thresh_f / cls_scale);
    const ClsT cls_thresh = static_cast<ClsT>(
        std::clamp(scaled_thresh,
                   static_cast<float>(std::numeric_limits<ClsT>::min()),
                   static_cast<float>(std::numeric_limits<ClsT>::max())));

    for (int h = 0; h < grid_h; h++) {
        for (int w = 0; w < grid_w; w++) {
            const int pixel_idx = h * grid_w + w;
            const int cls_offset = pixel_idx * num_classes;

            float max_score = -1.0f;
            int best_cls_id = -1;

            for (int c = 0; c < num_classes; c++) {
                const ClsT raw_val_t = raw_cls[cls_offset + c];
                if (raw_val_t <= cls_thresh) {
                    continue;
                }
                const float raw_val = dl::dequantize(raw_val_t, cls_scale);
                const float score = dl::math::sigmoid(raw_val);
                if (score > max_score) {
                    max_score = score;
                    best_cls_id = c;
                }
            }

            if (best_cls_id < 0 || max_score < m_conf_thresh) {
                continue;
            }

            const int box_offset = pixel_idx * 4;
            const float d_l = dl::dequantize(raw_box[box_offset + 0], box_scale);
            const float d_t = dl::dequantize(raw_box[box_offset + 1], box_scale);
            const float d_r = dl::dequantize(raw_box[box_offset + 2], box_scale);
            const float d_b = dl::dequantize(raw_box[box_offset + 3], box_scale);

            const float cx = static_cast<float>(w) + 0.5f;
            const float cy = static_cast<float>(h) + 0.5f;
            const int x1 = static_cast<int>(std::lround((cx - d_l) * stride));
            const int y1 = static_cast<int>(std::lround((cy - d_t) * stride));
            const int x2 = static_cast<int>(std::lround((cx + d_r) * stride));
            const int y2 = static_cast<int>(std::lround((cy + d_b) * stride));

            if (x2 <= x1 || y2 <= y1) {
                continue;
            }
            candidates.push_back({best_cls_id, max_score, {x1, y1, x2, y2}, {}});
        }
    }
}

static float box_iou(const Detection &a, const Detection &b)
{
    const int ix1 = std::max(a.x1, b.x1);
    const int iy1 = std::max(a.y1, b.y1);
    const int ix2 = std::min(a.x2, b.x2);
    const int iy2 = std::min(a.y2, b.y2);
    const int iw = std::max(0, ix2 - ix1);
    const int ih = std::max(0, iy2 - iy1);
    const float inter = static_cast<float>(iw * ih);
    if (inter <= 0.0f) {
        return 0.0f;
    }
    const float area_a = static_cast<float>((a.x2 - a.x1) * (a.y2 - a.y1));
    const float area_b = static_cast<float>((b.x2 - b.x1) * (b.y2 - b.y1));
    const float denom = area_a + area_b - inter;
    return denom > 0.0f ? (inter / denom) : 0.0f;
}

void YOLO26PPE::map_to_source(const std::vector<dl::detect::result_t> &raw,
                              int src_w,
                              int src_h,
                              std::vector<Detection> &out) const
{
    // Exact inverse of ESP-DL ImagePreprocessor letterbox geometry
    const float scale_x = static_cast<float>(m_input_w) / static_cast<float>(src_w);
    const float scale_y = static_cast<float>(m_input_h) / static_cast<float>(src_h);
    const float min_scale = std::min(scale_x, scale_y);

    int pad_top = 0;
    int pad_left = 0;
    int act_w = m_input_w;
    int act_h = m_input_h;

    if (scale_x < scale_y) {
        const int pad_h = m_input_h - static_cast<int>(min_scale * src_h);
        pad_top = pad_h / 2;
        act_h = m_input_h - pad_h;
    } else {
        const int pad_w = m_input_w - static_cast<int>(min_scale * src_w);
        pad_left = pad_w / 2;
        act_w = m_input_w - pad_w;
    }

    const float inv_x = static_cast<float>(src_w) / static_cast<float>(std::max(1, act_w));
    const float inv_y = static_cast<float>(src_h) / static_cast<float>(std::max(1, act_h));

    std::vector<Detection> mapped;
    mapped.reserve(raw.size());

    for (const auto &r : raw) {
        int x1 = static_cast<int>(std::lround((r.box[0] - pad_left) * inv_x));
        int y1 = static_cast<int>(std::lround((r.box[1] - pad_top) * inv_y));
        int x2 = static_cast<int>(std::lround((r.box[2] - pad_left) * inv_x));
        int y2 = static_cast<int>(std::lround((r.box[3] - pad_top) * inv_y));

        x1 = std::clamp(x1, 0, src_w - 1);
        y1 = std::clamp(y1, 0, src_h - 1);
        x2 = std::clamp(x2, 0, src_w - 1);
        y2 = std::clamp(y2, 0, src_h - 1);
        if (x2 - x1 < 4 || y2 - y1 < 4) {
            continue;
        }
        mapped.push_back({r.category, r.score, x1, y1, x2, y2});
    }

    std::sort(mapped.begin(), mapped.end(), [](const Detection &a, const Detection &b) {
        return a.score > b.score;
    });

    // Lightweight same-class dedup guard (keeps NMS-free one2one clean under quantization)
    out.clear();
    out.reserve(mapped.size());
    for (const auto &cand : mapped) {
        bool duplicate = false;
        for (const auto &kept : out) {
            if (kept.class_id == cand.class_id && box_iou(kept, cand) > 0.65f) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            out.push_back(cand);
        }
    }
}

std::vector<Detection> YOLO26PPE::run(const dl::image::img_t &img, InferenceTiming *timing)
{
    const int64_t t0 = esp_timer_get_time();
    m_preprocessor->preprocess(img);
    const int64_t t1 = esp_timer_get_time();

    m_model->run(dl::RUNTIME_MODE_MULTI_CORE);
    const int64_t t2 = esp_timer_get_time();

    const auto &outputs = m_model->get_outputs();
    static const char *kBoxNames[3] = {"one2one_p3_box", "one2one_p4_box", "one2one_p5_box"};
    static const char *kClsNames[3] = {"one2one_p3_cls", "one2one_p4_cls", "one2one_p5_cls"};

    std::vector<dl::detect::result_t> candidates;
    candidates.reserve(static_cast<size_t>(m_top_k) * 2);

    for (int i = 0; i < 3; i++) {
        auto it_box = outputs.find(kBoxNames[i]);
        auto it_cls = outputs.find(kClsNames[i]);
        if (it_box == outputs.end() || it_cls == outputs.end()) {
            ESP_LOGE(TAG, "Missing expected output tensor %s or %s", kBoxNames[i], kClsNames[i]);
            return {};
        }

        dl::TensorBase *p_box = it_box->second;
        dl::TensorBase *p_cls = it_cls->second;
        const int grid_h = p_cls->shape[1];
        const int grid_w = p_cls->shape[2];
        const int num_classes = p_cls->shape[3];

        if (p_box->dtype == dl::DATA_TYPE_INT8 && p_cls->dtype == dl::DATA_TYPE_INT8) {
            decode_grid<int8_t, int8_t>(p_box, p_cls, kStrides[i], grid_h, grid_w, num_classes, candidates);
        } else if (p_box->dtype == dl::DATA_TYPE_INT16 && p_cls->dtype == dl::DATA_TYPE_INT16) {
            decode_grid<int16_t, int16_t>(p_box, p_cls, kStrides[i], grid_h, grid_w, num_classes, candidates);
        } else if (p_box->dtype == dl::DATA_TYPE_INT16 && p_cls->dtype == dl::DATA_TYPE_INT8) {
            decode_grid<int16_t, int8_t>(p_box, p_cls, kStrides[i], grid_h, grid_w, num_classes, candidates);
        } else if (p_box->dtype == dl::DATA_TYPE_INT8 && p_cls->dtype == dl::DATA_TYPE_INT16) {
            decode_grid<int8_t, int16_t>(p_box, p_cls, kStrides[i], grid_h, grid_w, num_classes, candidates);
        } else {
            ESP_LOGE(TAG,
                     "Unsupported output tensor dtypes at scale %d: box=%d cls=%d",
                     i,
                     static_cast<int>(p_box->dtype),
                     static_cast<int>(p_cls->dtype));
            return {};
        }
    }

    if (static_cast<int>(candidates.size()) > m_top_k) {
        std::nth_element(candidates.begin(),
                         candidates.begin() + m_top_k,
                         candidates.end(),
                         dl::detect::greater_box);
        candidates.resize(static_cast<size_t>(m_top_k));
    }

    std::vector<Detection> results;
    map_to_source(candidates, img.width, img.height, results);

    const int64_t t3 = esp_timer_get_time();
    if (timing) {
        timing->pre_ms = static_cast<uint32_t>((t1 - t0) / 1000);
        timing->inf_ms = static_cast<uint32_t>((t2 - t1) / 1000);
        timing->post_ms = static_cast<uint32_t>((t3 - t2) / 1000);
    }

    return results;
}

} // namespace ppe
