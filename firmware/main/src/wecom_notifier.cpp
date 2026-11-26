#include "wecom_notifier.hpp"

#include "display_ui.hpp"
#include "esp_crt_bundle.h"
#include "esp_event.h"
#include "esp_heap_caps.h"
#include "esp_http_client.h"
#include "esp_jpeg_enc.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_netif_sntp.h"
#include "esp_rom_md5.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "mbedtls/base64.h"
#include "sdkconfig.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

namespace ppe {

static const char *TAG = "wecom_notifier";

static void downscale_rgb888(const uint8_t *src, int sw, int sh, uint8_t *dst, int dw, int dh)
{
    if (sw == dw && sh == dh) {
        memcpy(dst, src, static_cast<size_t>(dw) * dh * 3);
        return;
    }
    for (int y = 0; y < dh; y++) {
        const int sy = (y * sh) / dh;
        const uint8_t *srow = src + static_cast<size_t>(sy) * sw * 3;
        uint8_t *drow = dst + static_cast<size_t>(y) * dw * 3;
        for (int x = 0; x < dw; x++) {
            const int sx = (x * sw) / dw;
            drow[x * 3 + 0] = srow[sx * 3 + 0];
            drow[x * 3 + 1] = srow[sx * 3 + 1];
            drow[x * 3 + 2] = srow[sx * 3 + 2];
        }
    }
}

void WeComNotifier::wifi_event_handler(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    auto *self = static_cast<WeComNotifier *>(arg);
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        self->m_wifi_connected.store(false, std::memory_order_relaxed);
        ESP_LOGW(TAG, "Wi-Fi disconnected, retrying...");
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        auto *event = static_cast<ip_event_got_ip_t *>(data);
        ESP_LOGI(TAG, "Wi-Fi connected! IP=" IPSTR, IP2STR(&event->ip_info.ip));
        self->m_wifi_connected.store(true, std::memory_order_relaxed);
        self->start_sntp_once();
    }
}

void WeComNotifier::start_sntp_once()
{
    if (m_sntp_started) return;
    m_sntp_started = true;

    setenv("TZ", "CST-8", 1);
    tzset();

    esp_sntp_config_t cfg = ESP_NETIF_SNTP_DEFAULT_CONFIG("ntp.aliyun.com");
    esp_netif_sntp_init(&cfg);
    ESP_LOGI(TAG, "SNTP time sync started (UTC+8)");
}

esp_err_t WeComNotifier::init()
{
    m_queue = xQueueCreate(2, sizeof(AlertJob *));
    if (!m_queue) {
        return ESP_ERR_NO_MEM;
    }

    const char *ssid = CONFIG_PPE_WIFI_SSID;
    if (!ssid || ssid[0] == '\0') {
        ESP_LOGW(TAG, "CONFIG_PPE_WIFI_SSID is empty — running in offline-only mode (local alerts active)");
        return ESP_OK;
    }

    esp_err_t err = esp_netif_init();
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "esp_netif_init failed: %s", esp_err_to_name(err));
        return err;
    }
    err = esp_event_loop_create_default();
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "esp_event_loop_create_default failed: %s", esp_err_to_name(err));
        return err;
    }
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    err = esp_wifi_init(&cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_wifi_init failed: %s (free DRAM=%zu KB)",
                 esp_err_to_name(err),
                 heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024);
        return err;
    }

    err = esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &WeComNotifier::wifi_event_handler, this, nullptr);
    if (err != ESP_OK) return err;
    err = esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &WeComNotifier::wifi_event_handler, this, nullptr);
    if (err != ESP_OK) return err;

    wifi_config_t wifi_cfg = {};
    strlcpy(reinterpret_cast<char *>(wifi_cfg.sta.ssid), CONFIG_PPE_WIFI_SSID, sizeof(wifi_cfg.sta.ssid));
    strlcpy(reinterpret_cast<char *>(wifi_cfg.sta.password), CONFIG_PPE_WIFI_PASSWORD, sizeof(wifi_cfg.sta.password));
    wifi_cfg.sta.threshold.authmode = (strlen(CONFIG_PPE_WIFI_PASSWORD) > 0) ? WIFI_AUTH_WPA2_PSK : WIFI_AUTH_OPEN;

    err = esp_wifi_set_mode(WIFI_MODE_STA);
    if (err != ESP_OK) return err;
    err = esp_wifi_set_config(WIFI_IF_STA, &wifi_cfg);
    if (err != ESP_OK) return err;
    err = esp_wifi_start();
    if (err != ESP_OK) return err;

    BaseType_t ok = xTaskCreatePinnedToCoreWithCaps(
        wecom_task_trampoline, "wecom_alert", 8192, this, 3, nullptr, 0, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (ok != pdPASS) {
        ESP_LOGE(TAG, "Failed to create WeCom worker task");
        return ESP_ERR_NO_MEM;
    }

    ESP_LOGI(TAG, "WeCom notifier initialized (SSID=%s)", CONFIG_PPE_WIFI_SSID);
    return ESP_OK;
}

bool WeComNotifier::enqueue_alert(const uint8_t *src_rgb888,
                                  int src_w,
                                  int src_h,
                                  const InferenceSnapshot &snap)
{
    const char *url = CONFIG_PPE_WECOM_WEBHOOK_URL;
    if (!url || url[0] == '\0') {
        ESP_LOGW(TAG, "WeCom webhook URL empty; skipping remote push");
        return false;
    }
    if (!m_wifi_connected.load(std::memory_order_relaxed)) {
        ESP_LOGW(TAG, "Wi-Fi offline; skipping WeCom alert push");
        return false;
    }
    if (m_sending.load(std::memory_order_relaxed)) {
        ESP_LOGW(TAG, "Previous WeCom push still in progress; skipping");
        return false;
    }

    auto *job = new AlertJob();
    job->snap = snap;
    // ESP32-S3 esp_new_jpeg SIMD encoder requires 16-byte aligned input buffer
    job->rgb888_320x240 = static_cast<uint8_t *>(
        heap_caps_aligned_alloc(16, kSnapWidth * kSnapHeight * 3, MALLOC_CAP_SPIRAM));
    if (!job->rgb888_320x240) {
        delete job;
        ESP_LOGE(TAG, "Out of PSRAM for WeCom snapshot");
        return false;
    }

    downscale_rgb888(src_rgb888, src_w, src_h, job->rgb888_320x240, kSnapWidth, kSnapHeight);
    DisplayUI::draw_boxes_rgb888(job->rgb888_320x240, kSnapWidth, kSnapHeight, snap);

    if (xQueueSend(static_cast<QueueHandle_t>(m_queue), &job, 0) != pdTRUE) {
        heap_caps_free(job->rgb888_320x240);
        delete job;
        return false;
    }
    return true;
}

struct HttpRespBuf {
    char data[256];
    size_t len;
};

static esp_err_t wecom_http_event_handler(esp_http_client_event_t *evt)
{
    if (evt->event_id == HTTP_EVENT_ON_DATA && evt->user_data && evt->data && evt->data_len > 0) {
        auto *buf = static_cast<HttpRespBuf *>(evt->user_data);
        const size_t avail = (sizeof(buf->data) - 1) - buf->len;
        const size_t copy_n = std::min(avail, static_cast<size_t>(evt->data_len));
        if (copy_n > 0) {
            memcpy(buf->data + buf->len, evt->data, copy_n);
            buf->len += copy_n;
            buf->data[buf->len] = '\0';
        }
    }
    return ESP_OK;
}

esp_err_t WeComNotifier::post_json(const char *json_body, size_t len)
{
    HttpRespBuf resp = {};

    esp_http_client_config_t http_cfg = {};
    http_cfg.url = CONFIG_PPE_WECOM_WEBHOOK_URL;
    http_cfg.method = HTTP_METHOD_POST;
    http_cfg.timeout_ms = 12000;
    http_cfg.crt_bundle_attach = esp_crt_bundle_attach;
    http_cfg.buffer_size = 2048;
    http_cfg.buffer_size_tx = 2048;
    http_cfg.event_handler = wecom_http_event_handler;
    http_cfg.user_data = &resp;

    esp_http_client_handle_t client = esp_http_client_init(&http_cfg);
    if (!client) {
        return ESP_FAIL;
    }

    esp_http_client_set_header(client, "Content-Type", "application/json; charset=utf-8");
    esp_http_client_set_post_field(client, json_body, static_cast<int>(len));

    esp_err_t err = esp_http_client_perform(client);
    if (err == ESP_OK) {
        const int status = esp_http_client_get_status_code(client);
        ESP_LOGI(TAG, "WeCom HTTP %d | resp=%s", status, resp.data);
        const bool api_ok = (resp.len == 0) || (strstr(resp.data, "\"errcode\":0") != nullptr);
        if (status < 200 || status >= 300 || !api_ok) {
            ESP_LOGW(TAG, "WeCom API rejected request (status=%d, body=%s)", status, resp.data);
            err = ESP_FAIL;
        }
    } else {
        ESP_LOGE(TAG, "WeCom HTTPS POST failed: %s", esp_err_to_name(err));
    }

    esp_http_client_cleanup(client);
    return err;
}

bool WeComNotifier::send_markdown_alert(const InferenceSnapshot &snap)
{
    char time_str[40] = {0};
    time_t now = 0;
    time(&now);
    struct tm tm_info = {};
    localtime_r(&now, &tm_info);
    if (tm_info.tm_year >= (2024 - 1900)) {
        m_time_synced.store(true, std::memory_order_relaxed);
        strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", &tm_info);
    } else {
        const int64_t up_s = esp_timer_get_time() / 1000000;
        snprintf(time_str, sizeof(time_str), "开机运行 %lld 秒", static_cast<long long>(up_s));
    }

    const uint32_t inf_ms = snap.timing.pre_ms + snap.timing.inf_ms + snap.timing.post_ms;
    char body[768];
    const int n = snprintf(
        body,
        sizeof(body),
        "{\"msgtype\":\"markdown\",\"markdown\":{\"content\":"
        "\"### <font color=\\\"warning\\\">安全帽未佩戴告警</font>\\n"
        "> **设备编号**: %s\\n"
        "> **告警时间**: %s\\n"
        "> **未戴头盔人数**: <font color=\\\"warning\\\">%d</font>（直接检出 %d / 兜底 %d）\\n"
        "> **已戴头盔人数**: %d | **最高置信度**: %.0f%%\\n"
        "> **端侧推理耗时**: %lu ms（现场抓拍图随后发送）\"}}",
        CONFIG_PPE_DEVICE_ID,
        time_str,
        snap.total_violations,
        snap.no_helmet_count,
        snap.fallback_violation_count,
        snap.helmet_count,
        snap.max_violation_score * 100.0f,
        static_cast<unsigned long>(inf_ms));

    if (n <= 0 || static_cast<size_t>(n) >= sizeof(body)) {
        return false;
    }
    return post_json(body, static_cast<size_t>(n)) == ESP_OK;
}

bool WeComNotifier::send_image_alert(const uint8_t *rgb888_320x240)
{
    // 1. Encode 320x240 RGB888 to JPEG via esp_new_jpeg
    jpeg_enc_config_t jcfg = DEFAULT_JPEG_ENC_CONFIG();
    jcfg.width = kSnapWidth;
    jcfg.height = kSnapHeight;
    jcfg.src_type = JPEG_PIXEL_FORMAT_RGB888;
    jcfg.subsampling = JPEG_SUBSAMPLE_420;
    jcfg.quality = CONFIG_PPE_JPEG_QUALITY;
    jcfg.rotate = JPEG_ROTATE_0D;
    jcfg.task_enable = false;

    jpeg_enc_handle_t enc = nullptr;
    if (jpeg_enc_open(&jcfg, &enc) != JPEG_ERR_OK || !enc) {
        ESP_LOGE(TAG, "jpeg_enc_open failed");
        return false;
    }

    constexpr size_t kMaxJpegSize = 64 * 1024;
    auto *jpg_buf = static_cast<uint8_t *>(
        heap_caps_aligned_alloc(16, kMaxJpegSize, MALLOC_CAP_SPIRAM));
    if (!jpg_buf) {
        jpeg_enc_close(enc);
        return false;
    }

    int jpg_len = 0;
    const jpeg_error_t jerr = jpeg_enc_process(
        enc,
        rgb888_320x240,
        kSnapWidth * kSnapHeight * 3,
        jpg_buf,
        static_cast<int>(kMaxJpegSize),
        &jpg_len);
    jpeg_enc_close(enc);

    if (jerr != JPEG_ERR_OK || jpg_len <= 0) {
        ESP_LOGE(TAG, "jpeg_enc_process failed (err=%d, len=%d)", static_cast<int>(jerr), jpg_len);
        heap_caps_free(jpg_buf);
        return false;
    }

    // 2. Compute lowercase hex MD5 of raw JPEG bytes (using ESP32-S3 ROM MD5)
    md5_context_t md5_ctx;
    uint8_t digest[16];
    esp_rom_md5_init(&md5_ctx);
    esp_rom_md5_update(&md5_ctx, jpg_buf, static_cast<uint32_t>(jpg_len));
    esp_rom_md5_final(digest, &md5_ctx);

    char md5_hex[33];
    for (int i = 0; i < 16; i++) {
        snprintf(md5_hex + i * 2, 3, "%02x", digest[i]);
    }
    md5_hex[32] = '\0';

    // 3. Base64-encode JPEG directly into JSON body buffer in PSRAM
    const size_t b64_cap = ((static_cast<size_t>(jpg_len) + 2) / 3) * 4 + 8;
    const size_t json_cap = b64_cap + 128;
    auto *json_buf = static_cast<char *>(heap_caps_malloc(json_cap, MALLOC_CAP_SPIRAM));
    if (!json_buf) {
        heap_caps_free(jpg_buf);
        return false;
    }

    const char *prefix = "{\"msgtype\":\"image\",\"image\":{\"base64\":\"";
    const size_t prefix_len = strlen(prefix);
    memcpy(json_buf, prefix, prefix_len);

    size_t b64_len = 0;
    const int b64_rc = mbedtls_base64_encode(
        reinterpret_cast<unsigned char *>(json_buf + prefix_len),
        b64_cap,
        &b64_len,
        jpg_buf,
        static_cast<size_t>(jpg_len));
    heap_caps_free(jpg_buf);

    if (b64_rc != 0) {
        ESP_LOGE(TAG, "mbedtls_base64_encode failed: %d", b64_rc);
        heap_caps_free(json_buf);
        return false;
    }

    const int tail_len = snprintf(
        json_buf + prefix_len + b64_len,
        json_cap - prefix_len - b64_len,
        "\",\"md5\":\"%s\"}}",
        md5_hex);

    const size_t total_len = prefix_len + b64_len + static_cast<size_t>(tail_len);
    ESP_LOGI(TAG, "Uploading WeCom snapshot JPEG=%d B (payload=%zu B, md5=%s)", jpg_len, total_len, md5_hex);

    const bool ok = (post_json(json_buf, total_len) == ESP_OK);
    heap_caps_free(json_buf);
    return ok;
}

void WeComNotifier::wecom_task_trampoline(void *arg)
{
    static_cast<WeComNotifier *>(arg)->wecom_task_loop();
}

void WeComNotifier::wecom_task_loop()
{
    auto *queue = static_cast<QueueHandle_t>(m_queue);
    AlertJob *job = nullptr;

    while (true) {
        if (xQueueReceive(queue, &job, portMAX_DELAY) == pdTRUE && job) {
            m_sending.store(true, std::memory_order_relaxed);
            const bool ok_text = send_markdown_alert(job->snap);
            const bool ok_img = send_image_alert(job->rgb888_320x240);
            const bool ok = ok_text && ok_img;

            m_last_ok.store(ok, std::memory_order_relaxed);
            m_last_send_ms.store(esp_timer_get_time() / 1000, std::memory_order_relaxed);
            if (ok) {
                m_sent_count.fetch_add(1, std::memory_order_relaxed);
            }
            m_sending.store(false, std::memory_order_relaxed);

            heap_caps_free(job->rgb888_320x240);
            delete job;
            job = nullptr;
        }
    }
}

} // namespace ppe
