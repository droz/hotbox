#pragma once
// Stub for ESP32Encoder — native compilation.
// The harness injects encoder counts directly via hotbox_hal_set_encoder().

#include <cstdint>

// Two encoder instances: index 0 = azimuth, index 1 = elevation.
// Defined in hal.cpp; set by hotbox_cil_set_encoder() from the Python harness.
extern volatile long g_encoder_counts[2];

class ESP32Encoder {
public:
    int axis_idx_ = -1;  // set by attachFullQuad based on pin

    // pin_a values match electrical/pinouts.txt via config.h:
    // D2 = azimuth (horiz), D5 = elevation (vert).
    void attachFullQuad(int pin_a, int /*pin_b*/) {
        // D2 = 2 → azimuth (0), D5 = 5 → elevation (1)
        axis_idx_ = (pin_a == 2) ? 0 : 1;
    }

    void setCount(long val) {
        if (axis_idx_ >= 0) g_encoder_counts[axis_idx_] = val;
    }

    long getCount() const {
        return (axis_idx_ >= 0) ? g_encoder_counts[axis_idx_] : 0;
    }
};
