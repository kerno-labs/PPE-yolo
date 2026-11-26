#include "camera_capture.hpp"

#include "bsp/m5stack_core_s3.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_video_device.h"
#include "esp_video_ioctl.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "linux/videodev2.h"

#include <cstring>
#include <fcntl.h>
#include <inttypes.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace ppe {

static const char *TAG = "camera_capture";

void CameraCapture::rgb565_to_rgb888(const uint8_t *src,
                                     uint8_t *dst,
                                     uint32_t width,
                                     uint32_t height,
                                     bool is_big_endian)
{
    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (is_big_endian) {
        for (size_t i = 0; i < n; i++) {
            const uint16_t p = (static_cast<uint16_t>(src[2 * i]) << 8) | src[2 * i + 1];
            const uint8_t r = static_cast<uint8_t>((p >> 11) & 0x1F);
            const uint8_t g = static_cast<uint8_t>((p >> 5) & 0x3F);
            const uint8_t b = static_cast<uint8_t>(p & 0x1F);
            dst[3 * i + 0] = static_cast<uint8_t>((r << 3) | (r >> 2));
            dst[3 * i + 1] = static_cast<uint8_t>((g << 2) | (g >> 4));
            dst[3 * i + 2] = static_cast<uint8_t>((b << 3) | (b >> 2));
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            const uint16_t p = static_cast<uint16_t>(src[2 * i]) | (static_cast<uint16_t>(src[2 * i + 1]) << 8);
            const uint8_t r = static_cast<uint8_t>((p >> 11) & 0x1F);
            const uint8_t g = static_cast<uint8_t>((p >> 5) & 0x3F);
            const uint8_t b = static_cast<uint8_t>(p & 0x1F);
            dst[3 * i + 0] = static_cast<uint8_t>((r << 3) | (r >> 2));
            dst[3 * i + 1] = static_cast<uint8_t>((g << 2) | (g >> 4));
            dst[3 * i + 2] = static_cast<uint8_t>((b << 3) | (b >> 2));
        }
    }
}

esp_err_t CameraCapture::init(uint32_t prefer_width, uint32_t prefer_height)
{
    esp_err_t err = bsp_camera_start(nullptr);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "bsp_camera_start failed: %s", esp_err_to_name(err));
        return err;
    }

    m_fd = open(BSP_CAMERA_DEVICE, O_RDONLY);
    if (m_fd < 0) {
        ESP_LOGE(TAG, "Failed to open camera device %s", BSP_CAMERA_DEVICE);
        return ESP_FAIL;
    }

    struct v4l2_capability cap = {};
    if (ioctl(m_fd, VIDIOC_QUERYCAP, &cap) == 0) {
        ESP_LOGI(TAG, "Camera driver=%s card=%s", cap.driver, cap.card);
    }

    struct v4l2_format fmt = {};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(m_fd, VIDIOC_G_FMT, &fmt) != 0) {
        ESP_LOGE(TAG, "VIDIOC_G_FMT failed");
        close(m_fd);
        m_fd = -1;
        return ESP_FAIL;
    }

    // Attempt to negotiate preferred resolution (640x480 VGA for 512x512 YOLO26n)
    if (prefer_width > 0 && prefer_height > 0 &&
        (fmt.fmt.pix.width != prefer_width || fmt.fmt.pix.height != prefer_height)) {
        struct v4l2_format req_fmt = fmt;
        req_fmt.fmt.pix.width = prefer_width;
        req_fmt.fmt.pix.height = prefer_height;
        if (ioctl(m_fd, VIDIOC_S_FMT, &req_fmt) == 0) {
            ESP_LOGI(TAG, "Negotiated preferred camera resolution %" PRIu32 "x%" PRIu32,
                     prefer_width, prefer_height);
        } else {
            ESP_LOGW(TAG, "Preferred %" PRIu32 "x%" PRIu32 " not supported by current sensor mode; "
                          "keeping driver default %" PRIu32 "x%" PRIu32,
                     prefer_width, prefer_height, fmt.fmt.pix.width, fmt.fmt.pix.height);
        }
    }

    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(m_fd, VIDIOC_G_FMT, &fmt) != 0) {
        ESP_LOGE(TAG, "VIDIOC_G_FMT (final) failed");
        close(m_fd);
        m_fd = -1;
        return ESP_FAIL;
    }

    m_info.width = fmt.fmt.pix.width;
    m_info.height = fmt.fmt.pix.height;
    m_info.pixelformat = fmt.fmt.pix.pixelformat;
    m_info.is_rgb565_be = (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_RGB565);

    ESP_LOGI(TAG, "Active capture: %" PRIu32 "x%" PRIu32 ", fourcc=0x%08" PRIx32 " (rgb565_be=%d)",
             m_info.width, m_info.height, m_info.pixelformat, m_info.is_rgb565_be ? 1 : 0);

    // Request MMAP capture buffers
    struct v4l2_requestbuffers req = {};
    req.count = kNumBufs;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(m_fd, VIDIOC_REQBUFS, &req) != 0) {
        ESP_LOGE(TAG, "VIDIOC_REQBUFS failed");
        close(m_fd);
        m_fd = -1;
        return ESP_FAIL;
    }

    for (int i = 0; i < kNumBufs; i++) {
        struct v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (ioctl(m_fd, VIDIOC_QUERYBUF, &buf) != 0) {
            ESP_LOGE(TAG, "VIDIOC_QUERYBUF[%d] failed", i);
            return ESP_FAIL;
        }
        m_buffers[i] = static_cast<uint8_t *>(
            mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, m_fd, buf.m.offset));
        if (!m_buffers[i] || m_buffers[i] == MAP_FAILED) {
            ESP_LOGE(TAG, "mmap[%d] failed", i);
            return ESP_ERR_NO_MEM;
        }
        m_buf_len = buf.length;
        if (ioctl(m_fd, VIDIOC_QBUF, &buf) != 0) {
            ESP_LOGE(TAG, "VIDIOC_QBUF[%d] failed", i);
            return ESP_FAIL;
        }
    }

    m_snapshot_rgb565 = static_cast<uint8_t *>(
        heap_caps_malloc(static_cast<size_t>(m_info.width) * m_info.height * 2, MALLOC_CAP_SPIRAM));
    if (!m_snapshot_rgb565) {
        ESP_LOGE(TAG, "Failed to allocate PSRAM snapshot buffer");
        return ESP_ERR_NO_MEM;
    }

    m_snap_mutex = xSemaphoreCreateMutex();
    m_snap_ready_sem = xSemaphoreCreateBinary();
    if (!m_snap_mutex || !m_snap_ready_sem) {
        return ESP_ERR_NO_MEM;
    }

    return ESP_OK;
}

esp_err_t CameraCapture::start(FramePreviewCallback preview_cb, int core_id)
{
    m_preview_cb = std::move(preview_cb);
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(m_fd, VIDIOC_STREAMON, &type) != 0) {
        ESP_LOGE(TAG, "VIDIOC_STREAMON failed");
        return ESP_FAIL;
    }

    BaseType_t ok = xTaskCreatePinnedToCoreWithCaps(
        capture_task_trampoline, "cam_capture", 4096, this, 6, nullptr, core_id, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (ok != pdPASS) {
        ESP_LOGE(TAG, "Failed to create camera capture task");
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}

bool CameraCapture::grab_rgb888_snapshot(uint8_t *dst_rgb888, uint32_t timeout_ms)
{
    if (!dst_rgb888 || !m_snapshot_rgb565) {
        return false;
    }

    // Clear stale semaphore token and request next frame
    xSemaphoreTake(static_cast<SemaphoreHandle_t>(m_snap_ready_sem), 0);
    m_want_snapshot = true;

    if (xSemaphoreTake(static_cast<SemaphoreHandle_t>(m_snap_ready_sem), pdMS_TO_TICKS(timeout_ms)) != pdTRUE) {
        m_want_snapshot = false;
        return false;
    }

    xSemaphoreTake(static_cast<SemaphoreHandle_t>(m_snap_mutex), portMAX_DELAY);
    rgb565_to_rgb888(m_snapshot_rgb565, dst_rgb888, m_info.width, m_info.height, m_info.is_rgb565_be);
    xSemaphoreGive(static_cast<SemaphoreHandle_t>(m_snap_mutex));
    return true;
}

void CameraCapture::capture_task_trampoline(void *arg)
{
    static_cast<CameraCapture *>(arg)->capture_task_loop();
}

void CameraCapture::capture_task_loop()
{
    const size_t frame_bytes = static_cast<size_t>(m_info.width) * m_info.height * 2;
    while (true) {
        struct v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if (ioctl(m_fd, VIDIOC_DQBUF, &buf) != 0) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        uint8_t *frame_ptr = m_buffers[buf.index];
        if (m_want_snapshot) {
            if (xSemaphoreTake(static_cast<SemaphoreHandle_t>(m_snap_mutex), pdMS_TO_TICKS(20)) == pdTRUE) {
                memcpy(m_snapshot_rgb565, frame_ptr, frame_bytes);
                m_want_snapshot = false;
                xSemaphoreGive(static_cast<SemaphoreHandle_t>(m_snap_mutex));
                xSemaphoreGive(static_cast<SemaphoreHandle_t>(m_snap_ready_sem));
            }
        }

        if (m_preview_cb) {
            m_preview_cb(frame_ptr, m_info.width, m_info.height, m_info.is_rgb565_be);
        }

        if (ioctl(m_fd, VIDIOC_QBUF, &buf) != 0) {
            ESP_LOGW(TAG, "VIDIOC_QBUF failed");
        }
        vTaskDelay(pdMS_TO_TICKS(5));
    }
}

} // namespace ppe
