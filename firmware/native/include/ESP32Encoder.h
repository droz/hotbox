#pragma once
// Stub for ESP32Encoder — native compilation.
// The harness injects encoder counts directly via hotbox_hal_set_encoder().

#include <cstdint>

#include "hotbox_geometry.h"

// Two encoder instances: index 0 = azimuth, index 1 = elevation.
// Defined in hal.cpp; set by hotbox_cil_set_encoder() from the Python harness.
extern volatile long g_encoder_counts[2];

enum class puType { up, down, none };

class ESP32Encoder {
public:
    int axis_idx_ = -1;  // set by attachFullQuad based on pin
    static puType useInternalWeakPullResistors;

    // Identify axis from the A-channel Arduino pin in config/system.yaml.
    void attachFullQuad(int pin_a, int /*pin_b*/) {
        axis_idx_ = (pin_a == HOTBOX_PIN_AZIMUTH_ENC_A) ? 0 : 1;
    }

    void setCount(long val) {
        if (axis_idx_ >= 0) g_encoder_counts[axis_idx_] = val;
    }

    long getCount() const {
        return (axis_idx_ >= 0) ? g_encoder_counts[axis_idx_] : 0;
    }
};
