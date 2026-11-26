#pragma once

#include <cstdint>

namespace ppe {

enum ClassId : int {
    CLASS_HELMET = 0,
    CLASS_NO_HELMET = 1,
    CLASS_PERSON = 2,
    CLASS_COUNT = 3,
};

struct Detection {
    int class_id;
    float score;
    int x1; // Source image coordinates [0 .. src_w - 1]
    int y1; // Source image coordinates [0 .. src_h - 1]
    int x2;
    int y2;
};

struct InferenceTiming {
    uint32_t pre_ms;
    uint32_t inf_ms;
    uint32_t post_ms;
};

} // namespace ppe
