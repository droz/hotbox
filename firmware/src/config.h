#pragma once

#include <Arduino.h>
#include "hotbox_geometry.h"

namespace hotbox {

// ── Pin assignments ──────────────────────────────────────────────────────────
constexpr int kCanTxPin = D10;
constexpr int kCanRxPin = D9;

constexpr int kVertMotorP = A0;
constexpr int kVertMotorM = A1;
constexpr int kVertEncA = D2;
constexpr int kVertEncB = D3;
constexpr int kVertHall = D4;

constexpr int kHorizMotorP = A2;
constexpr int kHorizMotorM = A3;
constexpr int kHorizEncA = D5;
constexpr int kHorizEncB = D6;
constexpr int kHorizHall = D7;

#ifndef HOTBOX_NODE_ID
#define HOTBOX_NODE_ID 0
#endif

// ── Actuator constants (generated from config/system.yaml) ───────────────────
// Regenerate with: uv run hotbox-gen-firmware-geometry
constexpr float kControlPeriodS         = HOTBOX_CONTROL_PERIOD_S;
constexpr float kMaxVelocityDegS        = HOTBOX_MAX_VELOCITY_DEG_S;
constexpr float kMaxAccelDegS2          = HOTBOX_MAX_ACCEL_DEG_S2;
constexpr float kHomingVelocityDegS     = HOTBOX_HOMING_VELOCITY_DEG_S;
constexpr float kTicksPerDegree         = HOTBOX_TICKS_PER_DEGREE;
constexpr float kPidKp                  = HOTBOX_PID_KP;
constexpr float kPidKi                  = HOTBOX_PID_KI;
constexpr float kPidKd                  = HOTBOX_PID_KD;
constexpr float kStallVelocityThreshDegS = HOTBOX_STALL_VELOCITY_THRESHOLD_DEG_S;
constexpr float kStallTimeoutS          = HOTBOX_STALL_TIMEOUT_S;

}  // namespace hotbox
