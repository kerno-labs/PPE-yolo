#include "display_ui.hpp"

#include "bsp/display.h"
#include "bsp/m5stack_core_s3.h"
#include "bsp/touch.h"
#include "esp_heap_caps.h"
#include "esp_lcd_panel_ops.h"
#include "esp_lcd_touch.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "sdkconfig.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>

namespace ppe {

static const char *TAG = "display_ui";

// Pack 8-bit RGB into big-endian RGB565 matching CoreS3 ILI9342C panel (BSP_LCD_BIGENDIAN=1)
static inline uint16_t rgb565_be(uint8_t r, uint8_t g, uint8_t b)
{
    const uint16_t v = static_cast<uint16_t>(((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3));
    return static_cast<uint16_t>((v >> 8) | (v << 8));
}

static inline uint16_t bswap16(uint16_t v)
{
    return static_cast<uint16_t>((v >> 8) | (v << 8));
}

// Compact 5x7 uppercase/digit/symbol bitmap font (columns LSB=top, 5 bytes per glyph)
static const uint8_t *glyph_5x7(char c)
{
    static const uint8_t kBlank[5]  = {0x00, 0x00, 0x00, 0x00, 0x00};
    static const uint8_t kExcl[5]   = {0x00, 0x00, 0x5F, 0x00, 0x00};
    static const uint8_t kPct[5]    = {0x23, 0x13, 0x08, 0x64, 0x62};
    static const uint8_t kLParen[5] = {0x00, 0x1C, 0x22, 0x41, 0x00};
    static const uint8_t kRParen[5] = {0x00, 0x41, 0x22, 0x1C, 0x00};
    static const uint8_t kDash[5]   = {0x08, 0x08, 0x08, 0x08, 0x08};
    static const uint8_t kDot[5]    = {0x00, 0x60, 0x60, 0x00, 0x00};
    static const uint8_t kSlash[5]  = {0x20, 0x10, 0x08, 0x04, 0x02};
    static const uint8_t kColon[5]  = {0x00, 0x36, 0x36, 0x00, 0x00};
    static const uint8_t kQuest[5]  = {0x02, 0x01, 0x51, 0x09, 0x06};
    static const uint8_t kLBrk[5]   = {0x00, 0x7F, 0x41, 0x41, 0x00};
    static const uint8_t kRBrk[5]   = {0x00, 0x41, 0x41, 0x7F, 0x00};
    static const uint8_t kDigits[10][5] = {
        {0x3E, 0x51, 0x49, 0x45, 0x3E}, // 0
        {0x00, 0x42, 0x7F, 0x40, 0x00}, // 1
        {0x42, 0x61, 0x51, 0x49, 0x46}, // 2
        {0x21, 0x41, 0x45, 0x4B, 0x31}, // 3
        {0x18, 0x14, 0x12, 0x7F, 0x10}, // 4
        {0x27, 0x45, 0x45, 0x45, 0x39}, // 5
        {0x3C, 0x4A, 0x49, 0x49, 0x30}, // 6
        {0x01, 0x71, 0x09, 0x05, 0x03}, // 7
        {0x36, 0x49, 0x49, 0x49, 0x36}, // 8
        {0x06, 0x49, 0x49, 0x29, 0x1E}, // 9
    };
    static const uint8_t kLetters[26][5] = {
        {0x7E, 0x11, 0x11, 0x11, 0x7E}, // A
        {0x7F, 0x49, 0x49, 0x49, 0x36}, // B
        {0x3E, 0x41, 0x41, 0x41, 0x22}, // C
        {0x7F, 0x41, 0x41, 0x22, 0x1C}, // D
        {0x7F, 0x49, 0x49, 0x49, 0x41}, // E
        {0x7F, 0x09, 0x09, 0x09, 0x01}, // F
        {0x3E, 0x41, 0x49, 0x49, 0x7A}, // G
        {0x7F, 0x08, 0x08, 0x08, 0x7F}, // H
        {0x00, 0x41, 0x7F, 0x41, 0x00}, // I
        {0x20, 0x40, 0x41, 0x3F, 0x01}, // J
        {0x7F, 0x08, 0x14, 0x22, 0x41}, // K
        {0x7F, 0x40, 0x40, 0x40, 0x40}, // L
        {0x7F, 0x02, 0x0C, 0x02, 0x7F}, // M
        {0x7F, 0x04, 0x08, 0x10, 0x7F}, // N
        {0x3E, 0x41, 0x41, 0x41, 0x3E}, // O
        {0x7F, 0x09, 0x09, 0x09, 0x06}, // P
        {0x3E, 0x41, 0x51, 0x21, 0x5E}, // Q
        {0x7F, 0x09, 0x19, 0x29, 0x46}, // R
        {0x46, 0x49, 0x49, 0x49, 0x31}, // S
        {0x01, 0x01, 0x7F, 0x01, 0x01}, // T
        {0x3F, 0x40, 0x40, 0x40, 0x3F}, // U
        {0x1F, 0x20, 0x40, 0x20, 0x1F}, // V
        {0x3F, 0x40, 0x38, 0x40, 0x3F}, // W
        {0x63, 0x14, 0x08, 0x14, 0x63}, // X
        {0x07, 0x08, 0x70, 0x08, 0x07}, // Y
        {0x61, 0x51, 0x49, 0x45, 0x43}, // Z
    };

    const unsigned char uc = static_cast<unsigned char>(std::toupper(static_cast<unsigned char>(c)));
    if (uc >= 'A' && uc <= 'Z') return kLetters[uc - 'A'];
    if (uc >= '0' && uc <= '9') return kDigits[uc - '0'];
    switch (uc) {
    case '!': return kExcl;
    case '%': return kPct;
    case '(': return kLParen;
    case ')': return kRParen;
    case '-': return kDash;
    case '.': return kDot;
    case '/': return kSlash;
    case ':': return kColon;
    case '?': return kQuest;
    case '[': return kLBrk;
    case ']': return kRBrk;
    default:  return kBlank;
    }
}

esp_err_t DisplayUI::init()
{
    for (int i = 0; i < 2; i++) {
        m_fb_bufs[i] = static_cast<uint16_t *>(
            heap_caps_calloc(kWidth * kHeight, sizeof(uint16_t), MALLOC_CAP_SPIRAM));
        if (!m_fb_bufs[i]) {
            ESP_LOGE(TAG, "Failed to allocate 320x240 LCD framebuffer[%d] in PSRAM", i);
            return ESP_ERR_NO_MEM;
        }
    }
    m_fb_idx = 0;
    m_fb = m_fb_bufs[0];

    bsp_display_config_t disp_cfg = {
        .max_transfer_sz = kWidth * 20 * sizeof(uint16_t),
    };
    esp_lcd_panel_handle_t panel = nullptr;
    esp_lcd_panel_io_handle_t io = nullptr;
    esp_err_t err = bsp_display_new(&disp_cfg, &panel, &io);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "bsp_display_new failed: %s", esp_err_to_name(err));
        return err;
    }
    m_panel = panel;
    m_io = io;

    esp_lcd_panel_disp_on_off(panel, true);
    bsp_display_backlight_on();

    esp_lcd_touch_handle_t tp = nullptr;
    bsp_touch_config_t tp_cfg = {};
    if (bsp_touch_new(&tp_cfg, &tp) == ESP_OK) {
        m_touch = tp;
        ESP_LOGI(TAG, "CoreS3 touch panel initialized");
    } else {
        ESP_LOGW(TAG, "Touch panel init failed; continuing without touch buttons");
    }

    ESP_LOGI(TAG, "DisplayUI initialized (%dx%d RGB565_BE)", kWidth, kHeight);
    return ESP_OK;
}

void DisplayUI::fill_rect(int x, int y, int w, int h, uint16_t color_be)
{
    const int x0 = std::clamp(x, 0, kWidth);
    const int y0 = std::clamp(y, 0, kHeight);
    const int x1 = std::clamp(x + w, 0, kWidth);
    const int y1 = std::clamp(y + h, 0, kHeight);
    for (int py = y0; py < y1; py++) {
        uint16_t *row = m_fb + py * kWidth;
        for (int px = x0; px < x1; px++) {
            row[px] = color_be;
        }
    }
}

void DisplayUI::draw_rect(int x, int y, int w, int h, int thickness, uint16_t color_be)
{
    if (w <= 0 || h <= 0) return;
    fill_rect(x, y, w, thickness, color_be);
    fill_rect(x, y + h - thickness, w, thickness, color_be);
    fill_rect(x, y, thickness, h, color_be);
    fill_rect(x + w - thickness, y, thickness, h, color_be);
}

void DisplayUI::draw_char(int x, int y, char c, uint16_t fg_be, uint16_t bg_be, bool opaque_bg, int scale)
{
    const uint8_t *cols = glyph_5x7(c);
    for (int col = 0; col < 5; col++) {
        const uint8_t bits = cols[col];
        for (int row = 0; row < 7; row++) {
            const bool on = (bits >> row) & 1;
            if (on) {
                fill_rect(x + col * scale, y + row * scale, scale, scale, fg_be);
            } else if (opaque_bg) {
                fill_rect(x + col * scale, y + row * scale, scale, scale, bg_be);
            }
        }
    }
}

void DisplayUI::draw_text(int x, int y, const char *str, uint16_t fg_be, uint16_t bg_be, bool opaque_bg, int scale)
{
    int cx = x;
    while (*str) {
        draw_char(cx, y, *str, fg_be, bg_be, opaque_bg, scale);
        cx += 6 * scale;
        str++;
    }
}

void DisplayUI::flush()
{
    if (!m_panel || !m_fb) return;
    esp_lcd_panel_draw_bitmap(static_cast<esp_lcd_panel_handle_t>(m_panel), 0, 0, kWidth, kHeight, m_fb);
    // Flip to the other PSRAM framebuffer so the next frame never overwrites the buffer being read by SPI DMA
    m_fb_idx ^= 1;
    m_fb = m_fb_bufs[m_fb_idx];
}

void DisplayUI::show_splash(const char *line1, const char *line2)
{
    const uint16_t bg = rgb565_be(18, 22, 30);
    const uint16_t accent = rgb565_be(0, 200, 140);
    const uint16_t white = rgb565_be(245, 248, 255);
    const uint16_t gray = rgb565_be(160, 175, 195);

    fill_rect(0, 0, kWidth, kHeight, bg);
    draw_rect(8, 8, kWidth - 16, kHeight - 16, 2, accent);
    draw_text(28, 72, "PPE HELMET MONITOR", accent, bg, false, 2);
    draw_text(28, 118, line1 ? line1 : "", white, bg, false, 1);
    draw_text(28, 140, line2 ? line2 : "", gray, bg, false, 1);
    flush();
}

void DisplayUI::render_frame(const uint8_t *cam_rgb565,
                             uint32_t cam_w,
                             uint32_t cam_h,
                             bool cam_big_endian,
                             const InferenceSnapshot &snap,
                             const UiRuntimeStatus &status)
{
    if (!m_fb || !cam_rgb565 || cam_w == 0 || cam_h == 0) {
        return;
    }

    // 1. Copy / fast downsample camera RGB565 to 320x240 RGB565_BE framebuffer
    const uint16_t *src16 = reinterpret_cast<const uint16_t *>(cam_rgb565);
    if (cam_w == kWidth && cam_h == kHeight) {
        if (cam_big_endian) {
            memcpy(m_fb, src16, kWidth * kHeight * sizeof(uint16_t));
        } else {
            for (int i = 0; i < kWidth * kHeight; i++) {
                m_fb[i] = bswap16(src16[i]);
            }
        }
    } else if (cam_w == kWidth * 2 && cam_h == kHeight * 2) {
        // Fast 2x2 stride downsample for VGA 640x480 -> 320x240
        for (int y = 0; y < kHeight; y++) {
            const uint16_t *srow = src16 + (static_cast<size_t>(y) * 2) * cam_w;
            uint16_t *drow = m_fb + y * kWidth;
            if (cam_big_endian) {
                for (int x = 0; x < kWidth; x++) {
                    drow[x] = srow[x * 2];
                }
            } else {
                for (int x = 0; x < kWidth; x++) {
                    drow[x] = bswap16(srow[x * 2]);
                }
            }
        }
    } else {
        for (int y = 0; y < kHeight; y++) {
            const uint32_t sy = (static_cast<uint32_t>(y) * cam_h) / kHeight;
            const uint16_t *srow = src16 + static_cast<size_t>(sy) * cam_w;
            uint16_t *drow = m_fb + y * kWidth;
            for (int x = 0; x < kWidth; x++) {
                const uint32_t sx = (static_cast<uint32_t>(x) * cam_w) / kWidth;
                drow[x] = cam_big_endian ? srow[sx] : bswap16(srow[sx]);
            }
        }
    }

    const uint16_t col_green  = rgb565_be(0, 230, 118);
    const uint16_t col_red    = rgb565_be(255, 23, 68);
    const uint16_t col_orange = rgb565_be(255, 160, 0);
    const uint16_t col_cyan   = rgb565_be(41, 182, 246);
    const uint16_t col_white  = rgb565_be(255, 255, 255);
    const uint16_t col_dark   = rgb565_be(18, 22, 30);
    const uint16_t col_btn    = rgb565_be(36, 44, 58);

    // 2. Draw detection bounding boxes scaled from inference source coordinates to 320x240
    if (snap.src_width > 0 && snap.src_height > 0 && !status.paused) {
        const float sx = static_cast<float>(kWidth) / static_cast<float>(snap.src_width);
        const float sy = static_cast<float>(kHeight) / static_cast<float>(snap.src_height);

        for (const auto &b : snap.boxes) {
            const int x1 = std::clamp(static_cast<int>(b.x1 * sx), 0, kWidth - 1);
            const int y1 = std::clamp(static_cast<int>(b.y1 * sy), 22, kHeight - 36);
            const int x2 = std::clamp(static_cast<int>(b.x2 * sx), x1 + 2, kWidth - 1);
            const int y2 = std::clamp(static_cast<int>(b.y2 * sy), y1 + 2, kHeight - 34);

            uint16_t box_col = col_cyan;
            int thick = 2;
            char label[32];
            const int pct = static_cast<int>(b.score * 100.0f + 0.5f);

            if (b.is_violation) {
                box_col = b.is_fallback ? col_orange : col_red;
                thick = 3;
                snprintf(label, sizeof(label), "%s %d%%", b.is_fallback ? "NO HELM?" : "NO HELM", pct);
            } else if (b.class_id == CLASS_HELMET) {
                box_col = col_green;
                snprintf(label, sizeof(label), "HELM %d%%", pct);
            } else {
                box_col = col_cyan;
                snprintf(label, sizeof(label), "PERSON %d%%", pct);
            }

            draw_rect(x1, y1, x2 - x1, y2 - y1, thick, box_col);
            const int lw = static_cast<int>(strlen(label)) * 6 + 4;
            const int ly = (y1 >= 34) ? (y1 - 11) : (y1 + 3);
            fill_rect(x1, ly, lw, 11, box_col);
            draw_text(x1 + 2, ly + 2, label, col_dark, box_col, false, 1);
        }
    }

    // 3. Top Status Bar (22px)
    fill_rect(0, 0, kWidth, 22, col_dark);
    char top_left[40];
    if (status.paused) {
        snprintf(top_left, sizeof(top_left), "PPE PAUSED");
    } else {
        const uint32_t total_ms = snap.timing.pre_ms + snap.timing.inf_ms + snap.timing.post_ms;
        snprintf(top_left, sizeof(top_left), "H:%d !:%d %luMS",
                 snap.helmet_count, snap.total_violations, static_cast<unsigned long>(total_ms));
    }
    draw_text(6, 7, top_left, col_white, col_dark, false, 1);

    // Wi-Fi & WeCom indicators on top right
    const uint16_t wifi_col = status.wifi_connected ? col_green : col_orange;
    fill_rect(222, 7, 7, 7, wifi_col);
    draw_text(233, 7, status.wifi_connected ? "WIFI" : "OFF ", wifi_col, col_dark, false, 1);

    char wc_str[16];
    snprintf(wc_str, sizeof(wc_str), "WC:%lu", static_cast<unsigned long>(status.wecom_sent_count));
    draw_text(266, 7, wc_str, status.wecom_sending ? col_orange : col_cyan, col_dark, false, 1);

    // 4. Alert Banner / Border overlay when violating
    if (!status.paused && snap.stage == ALERT_WARNING_STAGE1) {
        char warn_buf[40];
        snprintf(warn_buf, sizeof(warn_buf), "WARNING: NO HELMET (%d/%d)",
                 snap.consecutive_violations, CONFIG_PPE_DEBOUNCE_FRAMES);
        draw_rect(0, 22, kWidth, kHeight - 56, 3, col_orange);
        fill_rect(24, 26, kWidth - 48, 18, col_orange);
        draw_text(44, 31, warn_buf, col_dark, col_orange, false, 1);
    } else if (!status.paused && snap.stage == ALERT_CONFIRMED_STAGE2) {
        const bool flash = ((esp_timer_get_time() / 300000) & 1) == 0;
        const uint16_t border_col = flash ? col_red : col_orange;
        draw_rect(0, 22, kWidth, kHeight - 56, 4, border_col);
        fill_rect(18, 26, kWidth - 36, 22, col_red);
        draw_text(44, 30, "ALARM: NO HELMET!", col_white, col_red, false, 2);
    }

    // 5. Bottom Touch Control Bar (34px: y = 206..239)
    fill_rect(0, 206, kWidth, 34, col_dark);

    // Left button: MUTE / MUTED
    const uint16_t mute_bg = status.muted ? col_orange : col_btn;
    const uint16_t mute_fg = status.muted ? col_dark : col_white;
    fill_rect(4, 209, 152, 28, mute_bg);
    draw_rect(4, 209, 152, 28, 1, col_white);
    draw_text(status.muted ? 42 : 48, 219, status.muted ? "[MUTED]" : "[MUTE]", mute_fg, mute_bg, false, 1);

    // Right button: PAUSE / RESUME
    const uint16_t pause_bg = status.paused ? col_cyan : col_btn;
    const uint16_t pause_fg = status.paused ? col_dark : col_white;
    fill_rect(164, 209, 152, 28, pause_bg);
    draw_rect(164, 209, 152, 28, 1, col_white);
    draw_text(status.paused ? 196 : 202, 219, status.paused ? "[RESUME]" : "[PAUSE]", pause_fg, pause_bg, false, 1);

    flush();
}

void DisplayUI::poll_touch(const TouchCallbacks &cbs)
{
    if (!m_touch) return;

    auto *tp = static_cast<esp_lcd_touch_handle_t>(m_touch);
    if (esp_lcd_touch_read_data(tp) != ESP_OK) {
        return;
    }

    esp_lcd_touch_point_data_t pt[1] = {};
    uint8_t count = 0;
    if (esp_lcd_touch_get_data(tp, pt, &count, 1) != ESP_OK || count == 0) {
        return;
    }

    const int64_t now_ms = esp_timer_get_time() / 1000;
    if (now_ms - m_last_touch_ms < 350) {
        return;
    }

    // Touch hit zones match the bottom bar (generous y >= 195 hit area)
    if (pt[0].y >= 195) {
        m_last_touch_ms = now_ms;
        if (pt[0].x < kWidth / 2) {
            if (cbs.on_toggle_mute) cbs.on_toggle_mute();
        } else {
            if (cbs.on_toggle_pause) cbs.on_toggle_pause();
        }
    }
}

static void rgb888_fill_rect(uint8_t *img, int w, int h, int rx, int ry, int rw, int rh,
                             uint8_t r, uint8_t g, uint8_t b)
{
    const int x0 = std::clamp(rx, 0, w);
    const int y0 = std::clamp(ry, 0, h);
    const int x1 = std::clamp(rx + rw, 0, w);
    const int y1 = std::clamp(ry + rh, 0, h);
    for (int py = y0; py < y1; py++) {
        uint8_t *row = img + static_cast<size_t>(py) * w * 3;
        for (int px = x0; px < x1; px++) {
            row[px * 3 + 0] = r;
            row[px * 3 + 1] = g;
            row[px * 3 + 2] = b;
        }
    }
}

static void rgb888_draw_text(uint8_t *img, int w, int h, int x, int y, const char *str,
                             uint8_t r, uint8_t g, uint8_t b)
{
    int cx = x;
    while (*str) {
        const uint8_t *cols = glyph_5x7(*str);
        for (int col = 0; col < 5; col++) {
            const uint8_t bits = cols[col];
            for (int row = 0; row < 7; row++) {
                if ((bits >> row) & 1) {
                    const int px = cx + col;
                    const int py = y + row;
                    if (px >= 0 && px < w && py >= 0 && py < h) {
                        uint8_t *p = img + (static_cast<size_t>(py) * w + px) * 3;
                        p[0] = r;
                        p[1] = g;
                        p[2] = b;
                    }
                }
            }
        }
        cx += 6;
        str++;
    }
}

void DisplayUI::draw_boxes_rgb888(uint8_t *rgb888, int img_w, int img_h, const InferenceSnapshot &snap)
{
    if (!rgb888 || snap.src_width <= 0 || snap.src_height <= 0) return;

    const float sx = static_cast<float>(img_w) / static_cast<float>(snap.src_width);
    const float sy = static_cast<float>(img_h) / static_cast<float>(snap.src_height);

    for (const auto &b : snap.boxes) {
        const int x1 = std::clamp(static_cast<int>(b.x1 * sx), 0, img_w - 1);
        const int y1 = std::clamp(static_cast<int>(b.y1 * sy), 0, img_h - 1);
        const int x2 = std::clamp(static_cast<int>(b.x2 * sx), x1 + 2, img_w - 1);
        const int y2 = std::clamp(static_cast<int>(b.y2 * sy), y1 + 2, img_h - 1);

        uint8_t r = 41, g = 182, bl = 246;
        int thick = 2;
        char label[32];
        const int pct = static_cast<int>(b.score * 100.0f + 0.5f);

        if (b.is_violation) {
            r = 255;
            g = b.is_fallback ? 160 : 23;
            bl = b.is_fallback ? 0 : 68;
            thick = 3;
            snprintf(label, sizeof(label), "%s %d%%", b.is_fallback ? "NO HELM?" : "NO HELM", pct);
        } else if (b.class_id == CLASS_HELMET) {
            r = 0; g = 230; bl = 118;
            snprintf(label, sizeof(label), "HELM %d%%", pct);
        } else {
            snprintf(label, sizeof(label), "PERSON %d%%", pct);
        }

        const int bw = x2 - x1;
        const int bh = y2 - y1;
        rgb888_fill_rect(rgb888, img_w, img_h, x1, y1, bw, thick, r, g, bl);
        rgb888_fill_rect(rgb888, img_w, img_h, x1, y2 - thick, bw, thick, r, g, bl);
        rgb888_fill_rect(rgb888, img_w, img_h, x1, y1, thick, bh, r, g, bl);
        rgb888_fill_rect(rgb888, img_w, img_h, x2 - thick, y1, thick, bh, r, g, bl);

        const int lw = static_cast<int>(strlen(label)) * 6 + 4;
        const int ly = (y1 >= 12) ? (y1 - 11) : (y1 + 3);
        rgb888_fill_rect(rgb888, img_w, img_h, x1, ly, lw, 11, r, g, bl);
        rgb888_draw_text(rgb888, img_w, img_h, x1 + 2, ly + 2, label, 16, 16, 16);
    }
}

} // namespace ppe
