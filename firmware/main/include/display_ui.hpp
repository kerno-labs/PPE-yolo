#pragma once

#include "app_config.hpp"
#include "esp_err.h"

#include <cstdint>
#include <functional>

namespace ppe {

struct TouchCallbacks {
    std::function<void()> on_toggle_mute;
    std::function<void()> on_toggle_pause;
};

class DisplayUI {
public:
    static constexpr int kWidth = 320;
    static constexpr int kHeight = 240;

    DisplayUI() = default;
    ~DisplayUI() = default;

    esp_err_t init();
    void show_splash(const char *line1, const char *line2);

    /**
     * @brief Render live camera frame + detection overlays + HUD + bottom touch buttons to CoreS3 LCD.
     */
    void render_frame(const uint8_t *cam_rgb565,
                      uint32_t cam_w,
                      uint32_t cam_h,
                      bool cam_big_endian,
                      const InferenceSnapshot &snap,
                      const UiRuntimeStatus &status);

    /**
     * @brief Poll capacitive touch panel (non-blocking) and invoke mute/pause callbacks with debounce.
     */
    void poll_touch(const TouchCallbacks &cbs);

    /**
     * @brief Draw scaled detection bounding boxes & compact labels directly onto an RGB888 image
     *        (used to annotate the 320x240 WeCom JPEG snapshot).
     */
    static void draw_boxes_rgb888(uint8_t *rgb888,
                                  int img_w,
                                  int img_h,
                                  const InferenceSnapshot &snap);

private:
    void fill_rect(int x, int y, int w, int h, uint16_t color_be);
    void draw_rect(int x, int y, int w, int h, int thickness, uint16_t color_be);
    void draw_char(int x, int y, char c, uint16_t fg_be, uint16_t bg_be, bool opaque_bg, int scale);
    void draw_text(int x, int y, const char *str, uint16_t fg_be, uint16_t bg_be, bool opaque_bg, int scale = 1);
    void flush();

    void *m_panel = nullptr; // esp_lcd_panel_handle_t
    void *m_io = nullptr;    // esp_lcd_panel_io_handle_t
    void *m_touch = nullptr; // esp_lcd_touch_handle_t
    uint16_t *m_fb_bufs[2] = {nullptr, nullptr}; // Ping-pong 320x240 RGB565_BE framebuffers in PSRAM
    uint16_t *m_fb = nullptr;                    // Active draw framebuffer pointer
    int m_fb_idx = 0;
    int64_t m_last_touch_ms = 0;
};

} // namespace ppe
