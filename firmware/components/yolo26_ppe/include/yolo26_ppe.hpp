#pragma once

#include "ppe_types.hpp"

#include "dl_detect_define.hpp"
#include "dl_image_preprocessor.hpp"
#include "dl_image_process.hpp"
#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ppe {

class YOLO26PPE {
public:
    static constexpr int kDefaultTopK = 32;
    static constexpr float kDefaultConfThresh = 0.30f;

    static const char *class_name(int class_id);

    YOLO26PPE(dl::Model *model, int top_k = kDefaultTopK, float conf_thresh = kDefaultConfThresh);
    ~YOLO26PPE();

    YOLO26PPE(const YOLO26PPE &) = delete;
    YOLO26PPE &operator=(const YOLO26PPE &) = delete;

    void set_conf_threshold(float thresh) { m_conf_thresh = thresh; }
    float conf_threshold() const { return m_conf_thresh; }
    int input_width() const { return m_input_w; }
    int input_height() const { return m_input_h; }

    /**
     * @brief Run full preprocess -> model->run() -> postprocess pipeline on an input image.
     * Coordinates returned in `Detection` are un-letterboxed back to (img.width x img.height).
     */
    std::vector<Detection> run(const dl::image::img_t &img, InferenceTiming *timing = nullptr);

private:
    template <typename BoxT, typename ClsT>
    void decode_grid(dl::TensorBase *p_box,
                     dl::TensorBase *p_cls,
                     int stride,
                     int grid_h,
                     int grid_w,
                     int num_classes,
                     std::vector<dl::detect::result_t> &candidates);

    void map_to_source(const std::vector<dl::detect::result_t> &raw,
                       int src_w,
                       int src_h,
                       std::vector<Detection> &out) const;

    dl::Model *m_model;
    dl::image::ImagePreprocessor *m_preprocessor;
    int m_input_w;
    int m_input_h;
    int m_top_k;
    float m_conf_thresh;
    int m_grid_h[3];
    int m_grid_w[3];
    static constexpr int kStrides[3] = {8, 16, 32};
};

} // namespace ppe
