#pragma once

#include "esp_err.h"

#include <atomic>
#include <cstdint>

namespace ppe {

enum AudioAlertType : uint8_t {
    AUDIO_CMD_BEEP_SHORT = 1,
    AUDIO_CMD_VOICE_ALERT = 2,
};

class AudioAlert {
public:
    AudioAlert() = default;
    ~AudioAlert() = default;

    esp_err_t init(int volume_pct = 85);

    void trigger_short_beep();
    void trigger_voice_alert();

    void set_muted(bool muted);
    bool is_muted() const { return m_muted.load(std::memory_order_relaxed); }

private:
    static void audio_task_trampoline(void *arg);
    void audio_task_loop();
    void play_tone(float freq_hz, int duration_ms, float amplitude = 0.55f);
    void play_embedded_voice();

    void *m_codec_dev = nullptr; // esp_codec_dev_handle_t
    void *m_queue = nullptr;     // QueueHandle_t
    std::atomic<bool> m_muted{false};
    bool m_initialized = false;
};

} // namespace ppe
