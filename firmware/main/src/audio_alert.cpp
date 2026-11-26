#include "audio_alert.hpp"

#include "bsp/m5stack_core_s3.h"
#include "esp_codec_dev.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"

#include <algorithm>
#include <cmath>
#include <cstring>

extern const uint8_t alert_zh_pcm_start[] asm("_binary_alert_zh_pcm_start");
extern const uint8_t alert_zh_pcm_end[] asm("_binary_alert_zh_pcm_end");

namespace ppe {

static const char *TAG = "audio_alert";
static constexpr int kSampleRate = 16000;
static constexpr float kTwoPi = 6.28318530718f;

esp_err_t AudioAlert::init(int volume_pct)
{
    esp_codec_dev_handle_t spk = bsp_audio_codec_speaker_init();
    if (!spk) {
        ESP_LOGE(TAG, "Failed to initialize CoreS3 speaker codec (AW88298)");
        return ESP_FAIL;
    }

    int vol = std::clamp(volume_pct, 0, 100);
    esp_codec_dev_set_out_vol(spk, vol);

    esp_codec_dev_sample_info_t fs = {};
    fs.sample_rate = kSampleRate;
    fs.channel = 1;
    fs.bits_per_sample = 16;
    if (esp_codec_dev_open(spk, &fs) != ESP_CODEC_DEV_OK) {
        ESP_LOGE(TAG, "Failed to open speaker codec at 16kHz 16-bit mono");
        return ESP_FAIL;
    }

    m_codec_dev = spk;
    m_queue = xQueueCreate(4, sizeof(AudioAlertType));
    if (!m_queue) {
        ESP_LOGE(TAG, "Failed to create audio command queue");
        return ESP_ERR_NO_MEM;
    }

    m_initialized = true;
    BaseType_t ok = xTaskCreatePinnedToCoreWithCaps(
        audio_task_trampoline, "audio_alert", 4096, this, 4, nullptr, 0, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (ok != pdPASS) {
        ESP_LOGE(TAG, "Failed to create audio task");
        return ESP_ERR_NO_MEM;
    }

    const size_t voice_bytes = static_cast<size_t>(alert_zh_pcm_end - alert_zh_pcm_start);
    ESP_LOGI(TAG,
             "Audio alert ready (vol=%d%%, embedded voice=%.1f KB / %.2fs)",
             vol,
             static_cast<float>(voice_bytes) / 1024.0f,
             static_cast<float>(voice_bytes) / static_cast<float>(kSampleRate * 2));
    return ESP_OK;
}

void AudioAlert::set_muted(bool muted)
{
    m_muted.store(muted, std::memory_order_relaxed);
    if (m_codec_dev) {
        esp_codec_dev_set_out_mute(static_cast<esp_codec_dev_handle_t>(m_codec_dev), muted);
    }
    ESP_LOGI(TAG, "Speaker %s", muted ? "MUTED" : "UNMUTED");
}

void AudioAlert::trigger_short_beep()
{
    if (!m_initialized || is_muted()) {
        return;
    }
    AudioAlertType cmd = AUDIO_CMD_BEEP_SHORT;
    xQueueSend(static_cast<QueueHandle_t>(m_queue), &cmd, 0);
}

void AudioAlert::trigger_voice_alert()
{
    if (!m_initialized || is_muted()) {
        return;
    }
    AudioAlertType cmd = AUDIO_CMD_VOICE_ALERT;
    xQueueSend(static_cast<QueueHandle_t>(m_queue), &cmd, 0);
}

void AudioAlert::audio_task_trampoline(void *arg)
{
    static_cast<AudioAlert *>(arg)->audio_task_loop();
}

void AudioAlert::play_tone(float freq_hz, int duration_ms, float amplitude)
{
    if (!m_codec_dev || is_muted()) {
        return;
    }
    auto *spk = static_cast<esp_codec_dev_handle_t>(m_codec_dev);
    const int total_samples = (kSampleRate * duration_ms) / 1000;
    const int fade_samples = std::min(total_samples / 6, 160);
    int16_t buf[256];

    int produced = 0;
    while (produced < total_samples && !is_muted()) {
        const int chunk = std::min(256, total_samples - produced);
        for (int i = 0; i < chunk; i++) {
            const int idx = produced + i;
            float env = 1.0f;
            if (idx < fade_samples) {
                env = static_cast<float>(idx) / static_cast<float>(std::max(1, fade_samples));
            } else if (idx > total_samples - fade_samples) {
                env = static_cast<float>(total_samples - idx) / static_cast<float>(std::max(1, fade_samples));
            }
            const float sample = std::sinf(kTwoPi * freq_hz * static_cast<float>(idx) / kSampleRate);
            buf[i] = static_cast<int16_t>(sample * amplitude * env * 32000.0f);
        }
        esp_codec_dev_write(spk, buf, chunk * static_cast<int>(sizeof(int16_t)));
        produced += chunk;
    }
}

void AudioAlert::play_embedded_voice()
{
    if (!m_codec_dev || is_muted()) {
        return;
    }
    auto *spk = static_cast<esp_codec_dev_handle_t>(m_codec_dev);
    const uint8_t *ptr = alert_zh_pcm_start;
    const size_t total_len = static_cast<size_t>(alert_zh_pcm_end - alert_zh_pcm_start);
    constexpr size_t kChunkBytes = 1024;

    // Brief double attention chime before spoken warning
    play_tone(1600.0f, 80, 0.45f);
    vTaskDelay(pdMS_TO_TICKS(50));
    play_tone(2100.0f, 90, 0.50f);
    vTaskDelay(pdMS_TO_TICKS(70));

    // Copy each chunk from Flash .rodata into writable RAM because esp_codec_dev_write
    // may scale samples in-place when software volume is active.
    uint8_t chunk_buf[kChunkBytes];
    size_t offset = 0;
    while (offset < total_len && !is_muted()) {
        const size_t n = std::min(kChunkBytes, total_len - offset);
        std::memcpy(chunk_buf, ptr + offset, n);
        esp_codec_dev_write(spk, chunk_buf, static_cast<int>(n));
        offset += n;
    }
}

void AudioAlert::audio_task_loop()
{
    auto *queue = static_cast<QueueHandle_t>(m_queue);
    AudioAlertType cmd;
    while (true) {
        if (xQueueReceive(queue, &cmd, portMAX_DELAY) == pdTRUE) {
            if (is_muted()) {
                continue;
            }
            if (cmd == AUDIO_CMD_BEEP_SHORT) {
                play_tone(1800.0f, 160, 0.45f);
            } else if (cmd == AUDIO_CMD_VOICE_ALERT) {
                play_embedded_voice();
            }
            // Drain stale duplicate requests queued during playback
            xQueueReset(queue);
        }
    }
}

} // namespace ppe
