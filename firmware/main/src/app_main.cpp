#include "app_config.hpp"
#include "audio_alert.hpp"
#include "bsp/m5stack_core_s3.h"
#include "camera_capture.hpp"
#include "display_ui.hpp"
#include "dl_model_base.hpp"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include "sdkconfig.h"
#include "violation_engine.hpp"
#include "wecom_notifier.hpp"
#include "yolo26_ppe.hpp"

#include <atomic>
#include <cstdio>

extern const uint8_t ppe_model_espdl_start[] asm("_binary_yolo26n_ppe_320_s8_s3_espdl_start");

namespace {

static const char *TAG = "app_main";

struct AppContext {
    ppe::CameraCapture camera;
    ppe::DisplayUI ui;
    ppe::AudioAlert audio;
    ppe::WeComNotifier wecom;
    dl::Model *model = nullptr;
    ppe::YOLO26PPE *detector = nullptr;
    ppe::ViolationEngine *engine = nullptr;

    SemaphoreHandle_t snap_mutex = nullptr;
    ppe::InferenceSnapshot latest_snap;
    std::atomic<bool> paused{false};
};

static AppContext *g_ctx = nullptr;

static void inference_task(void *arg)
{
    auto *ctx = static_cast<AppContext *>(arg);
    const auto &cam_info = ctx->camera.info();
    const size_t rgb888_bytes = static_cast<size_t>(cam_info.width) * cam_info.height * 3;
    auto *rgb888_buf = static_cast<uint8_t *>(heap_caps_malloc(rgb888_bytes, MALLOC_CAP_SPIRAM));
    if (!rgb888_buf) {
        ESP_LOGE(TAG, "Failed to allocate %zu B RGB888 buffer for inference", rgb888_bytes);
        vTaskDelete(nullptr);
        return;
    }

    ESP_LOGI(TAG,
             "Inference task started on Core %d (camera=%lux%lu -> YOLO26n %dx%d)",
             xPortGetCoreID(),
             static_cast<unsigned long>(cam_info.width),
             static_cast<unsigned long>(cam_info.height),
             ctx->detector->input_width(),
             ctx->detector->input_height());

    int64_t last_mem_log_ms = 0;

    while (true) {
        if (ctx->paused.load(std::memory_order_relaxed)) {
            ctx->engine->reset();
            vTaskDelay(pdMS_TO_TICKS(150));
            continue;
        }

        if (!ctx->camera.grab_rgb888_snapshot(rgb888_buf, 2000)) {
            vTaskDelay(pdMS_TO_TICKS(50));
            continue;
        }

        dl::image::img_t img = {};
        img.data = rgb888_buf;
        img.width = static_cast<uint16_t>(cam_info.width);
        img.height = static_cast<uint16_t>(cam_info.height);
        img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888;

        ppe::InferenceTiming timing{};
        auto detections = ctx->detector->run(img, &timing);
        const int64_t now_ms = esp_timer_get_time() / 1000;

        auto decision = ctx->engine->evaluate(
            detections,
            static_cast<int>(cam_info.width),
            static_cast<int>(cam_info.height),
            now_ms,
            timing);

        if (xSemaphoreTake(ctx->snap_mutex, pdMS_TO_TICKS(50)) == pdTRUE) {
            ctx->latest_snap = decision.snapshot;
            xSemaphoreGive(ctx->snap_mutex);
        }

        ESP_LOGI(TAG,
                 "Pre=%lums Inf=%lums Post=%lums | helmet=%d no_helmet=%d fallback=%d person=%d | stage=%d (%d/%d)",
                 static_cast<unsigned long>(timing.pre_ms),
                 static_cast<unsigned long>(timing.inf_ms),
                 static_cast<unsigned long>(timing.post_ms),
                 decision.snapshot.helmet_count,
                 decision.snapshot.no_helmet_count,
                 decision.snapshot.fallback_violation_count,
                 decision.snapshot.person_count,
                 static_cast<int>(decision.snapshot.stage),
                 decision.snapshot.consecutive_violations,
                 CONFIG_PPE_DEBOUNCE_FRAMES);

        if (decision.trigger_short_beep) {
            ctx->audio.trigger_short_beep();
        }
        if (decision.trigger_voice_alert) {
            ctx->audio.trigger_voice_alert();
        }
        if (decision.trigger_wecom_alert) {
            ctx->wecom.enqueue_alert(
                rgb888_buf,
                static_cast<int>(cam_info.width),
                static_cast<int>(cam_info.height),
                decision.snapshot);
        }

        if (now_ms - last_mem_log_ms > 30000) {
            last_mem_log_ms = now_ms;
            ESP_LOGI(TAG,
                     "Free DRAM=%zu KB | Free PSRAM=%zu KB",
                     heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024,
                     heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024);
        }

        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

} // namespace

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "=== M5Stack CoreS3 PPE Helmet Monitor (YOLO26n 320x320) ===");

    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_ERROR_CHECK(bsp_i2c_init());
    bsp_io_expander_init();

    g_ctx = new AppContext();
    g_ctx->snap_mutex = xSemaphoreCreateMutex();

    // 1. Display + Touch splash
    ESP_ERROR_CHECK(g_ctx->ui.init());
    g_ctx->ui.show_splash("INIT AUDIO & WI-FI...", "CORES3 + ESP-DL YOLO26N");

    // 2. Speaker (AW88298) & Wi-Fi + WeCom Notifier
    if (g_ctx->audio.init(CONFIG_PPE_SPEAKER_VOLUME) != ESP_OK) {
        ESP_LOGW(TAG, "Audio unavailable; continuing with screen + WeCom alerts");
    }
    if (g_ctx->wecom.init() != ESP_OK) {
        ESP_LOGW(TAG, "WeCom/Wi-Fi init failed; continuing offline");
    }

    // 3. Load YOLO26n 320x320 .espdl model (param_copy=true loads weights into PSRAM for faster inference;
    //    automatically falls back to zero-copy Flash if PSRAM is tight)
    g_ctx->ui.show_splash("LOADING YOLO26N 320X320...", "PSRAM WEIGHTS + DUAL-CORE");
    g_ctx->model = new dl::Model(
        reinterpret_cast<const char *>(ppe_model_espdl_start),
        fbs::MODEL_LOCATION_IN_FLASH_RODATA,
        0,
        dl::MEMORY_MANAGER_GREEDY,
        nullptr,
        true);
    if (!g_ctx->model->is_built()) {
        ESP_LOGW(TAG, "PSRAM weight copy build failed; retrying with zero-copy Flash (param_copy=false)");
        delete g_ctx->model;
        g_ctx->model = new dl::Model(
            reinterpret_cast<const char *>(ppe_model_espdl_start),
            fbs::MODEL_LOCATION_IN_FLASH_RODATA,
            0,
            dl::MEMORY_MANAGER_GREEDY,
            nullptr,
            false);
    }
    if (!g_ctx->model->is_built()) {
        ESP_LOGE(TAG, "Failed to build YOLO26n ESP-DL model execution plan!");
        g_ctx->ui.show_splash("MODEL INIT FAILED!", "CHECK PSRAM / .ESPDL");
        return;
    }

    const float conf_thresh = static_cast<float>(CONFIG_PPE_CONF_THRESH_PCT) / 100.0f;
    g_ctx->detector = new ppe::YOLO26PPE(g_ctx->model, ppe::YOLO26PPE::kDefaultTopK, conf_thresh);

    ppe::ViolationConfig vcfg{};
    vcfg.conf_thresh = conf_thresh;
    vcfg.enable_person_fallback =
#ifdef CONFIG_PPE_PERSON_FALLBACK_ENABLE
        true;
#else
        false;
#endif
    vcfg.debounce_frames = CONFIG_PPE_DEBOUNCE_FRAMES;
    vcfg.wecom_cooldown_ms = static_cast<int64_t>(CONFIG_PPE_WECOM_COOLDOWN_SEC) * 1000;
    vcfg.voice_cooldown_ms = static_cast<int64_t>(CONFIG_PPE_VOICE_COOLDOWN_SEC) * 1000;
    g_ctx->engine = new ppe::ViolationEngine(vcfg);

    // 4. Start GC0308 Camera (official CoreS3 BSP 320x240 RGB565_BE @ 20fps — clean DVP DMA without Quad PSRAM overflow)
    g_ctx->ui.show_splash("STARTING GC0308 CAMERA...", "WAITING FOR VIDEO STREAM");
    ESP_ERROR_CHECK(g_ctx->camera.init(320, 240));

    ppe::TouchCallbacks touch_cbs{};
    touch_cbs.on_toggle_mute = []() {
        const bool next = !g_ctx->audio.is_muted();
        g_ctx->audio.set_muted(next);
    };
    touch_cbs.on_toggle_pause = []() {
        const bool next = !g_ctx->paused.load(std::memory_order_relaxed);
        g_ctx->paused.store(next, std::memory_order_relaxed);
        ESP_LOGI(TAG, "Detection %s by touch", next ? "PAUSED" : "RESUMED");
    };

    ESP_ERROR_CHECK(g_ctx->camera.start(
        [touch_cbs](const uint8_t *rgb565, uint32_t w, uint32_t h, bool is_be) {
            g_ctx->ui.poll_touch(touch_cbs);

            ppe::InferenceSnapshot snap_copy;
            if (xSemaphoreTake(g_ctx->snap_mutex, 0) == pdTRUE) {
                snap_copy = g_ctx->latest_snap;
                xSemaphoreGive(g_ctx->snap_mutex);
            }

            ppe::UiRuntimeStatus st{};
            st.wifi_connected = g_ctx->wecom.is_wifi_connected();
            st.time_synced = g_ctx->wecom.is_time_synced();
            st.muted = g_ctx->audio.is_muted();
            st.paused = g_ctx->paused.load(std::memory_order_relaxed);
            st.wecom_sending = g_ctx->wecom.is_sending();
            st.wecom_sent_count = g_ctx->wecom.sent_count();
            st.last_wecom_ok = g_ctx->wecom.last_send_ok();
            st.last_wecom_ms = g_ctx->wecom.last_send_ms();

            g_ctx->ui.render_frame(rgb565, w, h, is_be, snap_copy, st);
        },
        /*core_id=*/0));

    // 5. Launch dedicated YOLO26n inference loop on Core 1 (stack in PSRAM to preserve internal DRAM for Wi-Fi/DMA)
    BaseType_t ok = xTaskCreatePinnedToCoreWithCaps(
        inference_task, "yolo26_inf", 16384, g_ctx, 5, nullptr, 1, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (ok != pdPASS) {
        ESP_LOGE(TAG, "Failed to create YOLO26 inference task!");
    }

    ESP_LOGI(TAG,
             "System running | Free DRAM=%zu KB | Free PSRAM=%zu KB",
             heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024,
             heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024);
}
