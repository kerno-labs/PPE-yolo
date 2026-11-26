#include <cassert>
#include <cstdio>
#include <vector>

// Host-side standalone unit test for ViolationEngine (does not require ESP-IDF headers)
#include "violation_engine.hpp"

int main()
{
    using namespace ppe;

    ViolationConfig cfg{};
    cfg.conf_thresh = 0.40f;
    cfg.enable_person_fallback = true;
    cfg.debounce_frames = 2;
    cfg.wecom_cooldown_ms = 30000;
    cfg.voice_cooldown_ms = 8000;

    ViolationEngine engine(cfg);
    InferenceTiming t{20, 2350, 25};

    // Case 1: Person WITH helmet in head region -> compliant, no alarm
    {
        std::vector<Detection> dets = {
            {CLASS_PERSON, 0.90f, 100, 80, 260, 440},
            {CLASS_HELMET, 0.88f, 145, 72, 215, 130},
        };
        auto dec = engine.evaluate(dets, 640, 480, 1000, t);
        assert(dec.snapshot.total_violations == 0);
        assert(dec.snapshot.helmet_count == 1);
        assert(dec.snapshot.stage == ALERT_NONE);
        assert(!dec.trigger_short_beep && !dec.trigger_voice_alert && !dec.trigger_wecom_alert);
    }

    // Case 2: Person cropped at top edge (y1 <= 4% of 480 = 19) -> NOT flagged as fallback
    {
        std::vector<Detection> dets = {
            {CLASS_PERSON, 0.85f, 200, 5, 380, 460},
        };
        auto dec = engine.evaluate(dets, 640, 480, 3500, t);
        assert(dec.snapshot.total_violations == 0);
        assert(dec.snapshot.stage == ALERT_NONE);
    }

    // Case 3: Direct no_helmet frame 1 -> Stage 1 (short beep only, no WeCom)
    {
        std::vector<Detection> dets = {
            {CLASS_NO_HELMET, 0.78f, 180, 90, 240, 150},
            {CLASS_PERSON, 0.85f, 140, 85, 290, 430}, // overlaps no_helmet, shouldn't double-count!
        };
        auto dec = engine.evaluate(dets, 640, 480, 6000, t);
        assert(dec.snapshot.no_helmet_count == 1);
        assert(dec.snapshot.fallback_violation_count == 0);
        assert(dec.snapshot.total_violations == 1);
        assert(dec.snapshot.stage == ALERT_WARNING_STAGE1);
        assert(dec.trigger_short_beep);
        assert(!dec.trigger_voice_alert);
        assert(!dec.trigger_wecom_alert);
    }

    // Case 4: Frame 2 (fallback person without helmet) -> Stage 2 (voice + WeCom triggered!)
    {
        std::vector<Detection> dets = {
            {CLASS_PERSON, 0.82f, 150, 90, 300, 440},
        };
        auto dec = engine.evaluate(dets, 640, 480, 8500, t);
        assert(dec.snapshot.fallback_violation_count == 1);
        assert(dec.snapshot.total_violations == 1);
        assert(dec.snapshot.stage == ALERT_CONFIRMED_STAGE2);
        assert(!dec.trigger_short_beep);
        assert(dec.trigger_voice_alert);
        assert(dec.trigger_wecom_alert);
    }

    // Case 5: Frame 3 within cooldowns (t=11000ms) -> Stage 2 stays active, neither voice nor WeCom re-fires
    {
        std::vector<Detection> dets = {
            {CLASS_NO_HELMET, 0.91f, 180, 90, 240, 150},
        };
        auto dec = engine.evaluate(dets, 640, 480, 11000, t);
        assert(dec.snapshot.stage == ALERT_CONFIRMED_STAGE2);
        assert(!dec.trigger_voice_alert);
        assert(!dec.trigger_wecom_alert);
    }

    // Case 6: Frame 4 after voice cooldown (t=17000ms) -> voice re-fires, WeCom still in 30s cooldown
    {
        std::vector<Detection> dets = {
            {CLASS_NO_HELMET, 0.91f, 180, 90, 240, 150},
        };
        auto dec = engine.evaluate(dets, 640, 480, 17000, t);
        assert(dec.trigger_voice_alert);
        assert(!dec.trigger_wecom_alert);
    }

    // Case 7: Frame 5 after WeCom cooldown (t=39000ms) -> both voice & WeCom fire again
    {
        std::vector<Detection> dets = {
            {CLASS_NO_HELMET, 0.91f, 180, 90, 240, 150},
        };
        auto dec = engine.evaluate(dets, 640, 480, 39000, t);
        assert(dec.trigger_voice_alert);
        assert(dec.trigger_wecom_alert);
    }

    // Case 8: Clean frame resets debounce streak
    {
        std::vector<Detection> dets = {};
        auto dec = engine.evaluate(dets, 640, 480, 41500, t);
        assert(dec.snapshot.stage == ALERT_NONE);
        assert(dec.snapshot.consecutive_violations == 0);
    }

    std::printf("ALL VIOLATION ENGINE UNIT TESTS PASSED!\n");
    return 0;
}
