#pragma once

#include "app_config.hpp"
#include "esp_err.h"
#include "esp_event_base.h"

#include <atomic>
#include <cstdint>

namespace ppe {

class WeComNotifier {
public:
    static constexpr int kSnapWidth = 320;
    static constexpr int kSnapHeight = 240;

    WeComNotifier() = default;
    ~WeComNotifier() = default;

    esp_err_t init();

    /**
     * @brief Enqueue an asynchronous WeCom alert (Markdown text + annotated 320x240 JPEG image).
     *        Takes ownership of no external buffers; copies & downscales source RGB888 internally.
     */
    bool enqueue_alert(const uint8_t *src_rgb888,
                       int src_w,
                       int src_h,
                       const InferenceSnapshot &snap);

    bool is_wifi_connected() const { return m_wifi_connected.load(std::memory_order_relaxed); }
    bool is_time_synced() const { return m_time_synced.load(std::memory_order_relaxed); }
    bool is_sending() const { return m_sending.load(std::memory_order_relaxed); }
    uint32_t sent_count() const { return m_sent_count.load(std::memory_order_relaxed); }
    bool last_send_ok() const { return m_last_ok.load(std::memory_order_relaxed); }
    int64_t last_send_ms() const { return m_last_send_ms.load(std::memory_order_relaxed); }

private:
    struct AlertJob {
        uint8_t *rgb888_320x240 = nullptr;
        InferenceSnapshot snap;
    };

    static void wifi_event_handler(void *arg, esp_event_base_t base, int32_t id, void *data);
    static void wecom_task_trampoline(void *arg);
    void wecom_task_loop();

    esp_err_t post_json(const char *json_body, size_t len);
    bool send_markdown_alert(const InferenceSnapshot &snap);
    bool send_image_alert(const uint8_t *rgb888_320x240);
    void start_sntp_once();

    void *m_queue = nullptr; // QueueHandle_t of AlertJob*
    std::atomic<bool> m_wifi_connected{false};
    std::atomic<bool> m_time_synced{false};
    std::atomic<bool> m_sending{false};
    std::atomic<uint32_t> m_sent_count{0};
    std::atomic<bool> m_last_ok{false};
    std::atomic<int64_t> m_last_send_ms{0};
    bool m_sntp_started = false;
};

} // namespace ppe
