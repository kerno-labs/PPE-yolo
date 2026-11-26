#pragma once

#include "ppe_types.hpp"

#include <cstdint>
#include <vector>

namespace ppe {

enum AlertStage : uint8_t {
    ALERT_NONE = 0,
    ALERT_WARNING_STAGE1 = 1,   // Frame 1 violation: local warning banner + short beep
    ALERT_CONFIRMED_STAGE2 = 2, // >= N consecutive frames: voice alert + WeCom push
};

struct AnnotatedBox {
    int class_id;
    float score;
    int x1;
    int y1;
    int x2;
    int y2;
    bool is_violation; // True if no_helmet OR fallback person without helmet
    bool is_fallback;  // True specifically for person-without-helmet fallback
};

struct InferenceSnapshot {
    int src_width = 0;
    int src_height = 0;
    std::vector<AnnotatedBox> boxes;
    int helmet_count = 0;
    int no_helmet_count = 0;
    int fallback_violation_count = 0;
    int person_count = 0;
    int total_violations = 0;
    float max_violation_score = 0.0f;
    int consecutive_violations = 0;
    AlertStage stage = ALERT_NONE;
    InferenceTiming timing = {0, 0, 0};
    int64_t timestamp_ms = 0;
};

struct UiRuntimeStatus {
    bool wifi_connected = false;
    bool time_synced = false;
    bool muted = false;
    bool paused = false;
    bool wecom_sending = false;
    uint32_t wecom_sent_count = 0;
    bool last_wecom_ok = false;
    int64_t last_wecom_ms = 0;
};

} // namespace ppe
