#pragma once

#include "esp_err.h"

#include <cstddef>
#include <cstdint>
#include <functional>

namespace ppe {

struct CameraFrameInfo {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t pixelformat = 0;
    bool is_rgb565_be = true;
};

using FramePreviewCallback = std::function<void(const uint8_t *rgb565_buf,
                                                uint32_t width,
                                                uint32_t height,
                                                bool is_big_endian)>;

class CameraCapture {
public:
    CameraCapture() = default;
    ~CameraCapture() = default;

    esp_err_t init(uint32_t prefer_width = 640, uint32_t prefer_height = 480);
    esp_err_t start(FramePreviewCallback preview_cb, int core_id = 0);

    const CameraFrameInfo &info() const { return m_info; }

    /**
     * @brief Wait for the next fresh camera frame and convert it into caller's RGB888 buffer.
     *        Caller must allocate `dst_rgb888` with at least width * height * 3 bytes.
     */
    bool grab_rgb888_snapshot(uint8_t *dst_rgb888, uint32_t timeout_ms = 2000);

    /**
     * @brief Utility to convert RGB565 (BE or LE) buffer to packed RGB888.
     */
    static void rgb565_to_rgb888(const uint8_t *src,
                                 uint8_t *dst,
                                 uint32_t width,
                                 uint32_t height,
                                 bool is_big_endian);

private:
    static void capture_task_trampoline(void *arg);
    void capture_task_loop();

    static constexpr int kNumBufs = 2;
    int m_fd = -1;
    CameraFrameInfo m_info{};
    uint8_t *m_buffers[kNumBufs] = {nullptr, nullptr};
    size_t m_buf_len = 0;

    FramePreviewCallback m_preview_cb;
    uint8_t *m_snapshot_rgb565 = nullptr;
    void *m_snap_mutex = nullptr;
    void *m_snap_ready_sem = nullptr;
    volatile bool m_want_snapshot = false;
};

} // namespace ppe
