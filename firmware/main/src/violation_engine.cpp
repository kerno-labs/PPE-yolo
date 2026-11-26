#include "violation_engine.hpp"

#include <algorithm>

namespace ppe {

namespace {

struct RectF {
    float x1, y1, x2, y2;
};

bool head_matches_box(const RectF &head, const Detection &det)
{
    const float cx = 0.5f * static_cast<float>(det.x1 + det.x2);
    const float cy = 0.5f * static_cast<float>(det.y1 + det.y2);
    if (cx >= head.x1 && cx <= head.x2 && cy >= head.y1 && cy <= head.y2) {
        return true;
    }

    const float ix1 = std::max(head.x1, static_cast<float>(det.x1));
    const float iy1 = std::max(head.y1, static_cast<float>(det.y1));
    const float ix2 = std::min(head.x2, static_cast<float>(det.x2));
    const float iy2 = std::min(head.y2, static_cast<float>(det.y2));
    const float iw = std::max(0.0f, ix2 - ix1);
    const float ih = std::max(0.0f, iy2 - iy1);
    const float inter = iw * ih;
    const float det_area = static_cast<float>(std::max(1, (det.x2 - det.x1) * (det.y2 - det.y1)));
    return (inter / det_area) >= 0.30f;
}

} // namespace

ViolationEngine::ViolationEngine(const ViolationConfig &cfg) : m_cfg(cfg) {}

void ViolationEngine::reset()
{
    m_consecutive_violations = 0;
}

ViolationDecision ViolationEngine::evaluate(const std::vector<Detection> &detections,
                                            int src_width,
                                            int src_height,
                                            int64_t now_ms,
                                            const InferenceTiming &timing)
{
    ViolationDecision decision{};
    InferenceSnapshot &snap = decision.snapshot;
    snap.src_width = src_width;
    snap.src_height = src_height;
    snap.timing = timing;
    snap.timestamp_ms = now_ms;

    std::vector<const Detection *> helmets;
    std::vector<const Detection *> no_helmets;
    std::vector<const Detection *> persons;

    for (const auto &d : detections) {
        if (d.score < m_cfg.conf_thresh) {
            continue;
        }
        if (d.class_id == CLASS_HELMET) {
            helmets.push_back(&d);
        } else if (d.class_id == CLASS_NO_HELMET) {
            no_helmets.push_back(&d);
        } else if (d.class_id == CLASS_PERSON) {
            persons.push_back(&d);
        }
    }

    snap.helmet_count = static_cast<int>(helmets.size());
    snap.no_helmet_count = static_cast<int>(no_helmets.size());
    snap.person_count = static_cast<int>(persons.size());

    // 1. Add helmet boxes (compliant)
    for (const auto *h : helmets) {
        snap.boxes.push_back({h->class_id, h->score, h->x1, h->y1, h->x2, h->y2, false, false});
    }

    // 2. Add direct no_helmet detections (violations)
    for (const auto *nh : no_helmets) {
        snap.boxes.push_back({nh->class_id, nh->score, nh->x1, nh->y1, nh->x2, nh->y2, true, false});
        if (nh->score > snap.max_violation_score) {
            snap.max_violation_score = nh->score;
        }
    }

    // 3. Evaluate person boxes + optional head-region fallback
    const int top_margin = std::max(4, static_cast<int>(src_height * 0.04f));
    for (const auto *p : persons) {
        bool is_fallback_violation = false;
        if (m_cfg.enable_person_fallback && p->y1 > top_margin) {
            const float pw = static_cast<float>(std::max(1, p->x2 - p->x1));
            const float ph = static_cast<float>(std::max(1, p->y2 - p->y1));

            // Head region: top 42% of person box, slightly expanded upward for hard hats
            const RectF head = {
                static_cast<float>(p->x1) - 0.10f * pw,
                std::max(0.0f, static_cast<float>(p->y1) - 0.12f * ph),
                static_cast<float>(p->x2) + 0.10f * pw,
                static_cast<float>(p->y1) + 0.42f * ph,
            };

            bool has_helmet = false;
            for (const auto *h : helmets) {
                if (head_matches_box(head, *h)) {
                    has_helmet = true;
                    break;
                }
            }

            bool already_no_helmet = false;
            if (!has_helmet) {
                for (const auto *nh : no_helmets) {
                    if (head_matches_box(head, *nh)) {
                        already_no_helmet = true;
                        break;
                    }
                }
            }

            if (!has_helmet && !already_no_helmet) {
                is_fallback_violation = true;
                snap.fallback_violation_count++;
                if (p->score > snap.max_violation_score) {
                    snap.max_violation_score = p->score;
                }
            }
        }

        snap.boxes.push_back({
            p->class_id,
            p->score,
            p->x1,
            p->y1,
            p->x2,
            p->y2,
            is_fallback_violation,
            is_fallback_violation,
        });
    }

    snap.total_violations = snap.no_helmet_count + snap.fallback_violation_count;

    // 4. Two-stage debounce & cooldown state machine
    if (snap.total_violations > 0) {
        m_consecutive_violations++;
        snap.consecutive_violations = m_consecutive_violations;

        if (m_consecutive_violations < m_cfg.debounce_frames) {
            snap.stage = ALERT_WARNING_STAGE1;
            decision.trigger_short_beep = true;
        } else {
            snap.stage = ALERT_CONFIRMED_STAGE2;
            if (now_ms - m_last_voice_ms >= m_cfg.voice_cooldown_ms) {
                decision.trigger_voice_alert = true;
                m_last_voice_ms = now_ms;
            }
            if (now_ms - m_last_wecom_ms >= m_cfg.wecom_cooldown_ms) {
                decision.trigger_wecom_alert = true;
                m_last_wecom_ms = now_ms;
            }
        }
    } else {
        m_consecutive_violations = 0;
        snap.consecutive_violations = 0;
        snap.stage = ALERT_NONE;
    }

    return decision;
}

} // namespace ppe
