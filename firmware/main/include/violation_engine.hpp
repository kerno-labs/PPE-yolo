#pragma once

#include "app_config.hpp"

#include <cstdint>
#include <vector>

namespace ppe {

struct ViolationConfig {
    float conf_thresh = 0.40f;
    bool enable_person_fallback = true;
    int debounce_frames = 2;
    int64_t wecom_cooldown_ms = 30000;
    int64_t voice_cooldown_ms = 8000;
};

struct ViolationDecision {
    InferenceSnapshot snapshot;
    bool trigger_short_beep = false;
    bool trigger_voice_alert = false;
    bool trigger_wecom_alert = false;
};

class ViolationEngine {
public:
    explicit ViolationEngine(const ViolationConfig &cfg);

    void reset();

    ViolationDecision evaluate(const std::vector<Detection> &detections,
                               int src_width,
                               int src_height,
                               int64_t now_ms,
                               const InferenceTiming &timing);

private:
    ViolationConfig m_cfg;
    int m_consecutive_violations = 0;
    int64_t m_last_voice_ms = -1000000;
    int64_t m_last_wecom_ms = -1000000;
};

} // namespace ppe
