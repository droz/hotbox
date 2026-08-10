#pragma once

#include <Arduino.h>
#include "hotbox_geometry.h"

namespace hotbox {

// ── Pin assignments (from config/system.yaml via hotbox_geometry.h) ───────────
constexpr int kCanTxPin = HOTBOX_PIN_CAN_TX;
constexpr int kCanRxPin = HOTBOX_PIN_CAN_RX;

constexpr int kVertMotorP = HOTBOX_PIN_ELEVATION_MOTOR_P;
constexpr int kVertMotorM = HOTBOX_PIN_ELEVATION_MOTOR_M;
constexpr int kVertEncA = HOTBOX_PIN_ELEVATION_ENC_A;
constexpr int kVertEncB = HOTBOX_PIN_ELEVATION_ENC_B;
constexpr int kVertHall = HOTBOX_PIN_ELEVATION_HALL;

constexpr int kHorizMotorP = HOTBOX_PIN_AZIMUTH_MOTOR_P;
constexpr int kHorizMotorM = HOTBOX_PIN_AZIMUTH_MOTOR_M;
constexpr int kHorizEncA = HOTBOX_PIN_AZIMUTH_ENC_A;
constexpr int kHorizEncB = HOTBOX_PIN_AZIMUTH_ENC_B;
constexpr int kHorizHall = HOTBOX_PIN_AZIMUTH_HALL;

// H-bridge PWM carrier (Arduino-ESP32 default is 1 kHz; 20 kHz is above hearing).
constexpr int kMotorPwmHz = HOTBOX_MOTOR_PWM_HZ;

#ifndef HOTBOX_NODE_ID
#define HOTBOX_NODE_ID 0
#endif

// ESP32Encoder / TWAI take Espressif GPIO numbers. On Nano ESP32 the Arduino
// D# labels are remapped (D5 → GPIO 8, etc.), so convert before driving
// peripherals that bypass the Arduino pin API.
#if defined(NATIVE_CIL)
inline int pinToGpio(int arduino_pin) { return arduino_pin; }
#else
inline int pinToGpio(int arduino_pin) {
  return static_cast<int>(digitalPinToGPIONumber(static_cast<int8_t>(arduino_pin)));
}
#endif

// ── Actuator constants (generated from config/system.yaml) ───────────────────
// Regenerate with: uv run hotbox-gen-firmware-geometry
constexpr float kControlPeriodS         = HOTBOX_CONTROL_PERIOD_S;
constexpr float kMaxVelocityDegS        = HOTBOX_MAX_VELOCITY_DEG_S;
constexpr float kMaxAccelDegS2          = HOTBOX_MAX_ACCEL_DEG_S2;
constexpr float kHomingSearchVelocityDegS = HOTBOX_HOMING_SEARCH_VELOCITY_DEG_S;
constexpr float kHomingCreepVelocityDegS  = HOTBOX_HOMING_CREEP_VELOCITY_DEG_S;
constexpr float kHomingBackoffDeg         = HOTBOX_HOMING_BACKOFF_DEG;
constexpr float kHomingSettleTolDeg       = HOTBOX_HOMING_SETTLE_TOL_DEG;
constexpr float kTicksPerDegree         = HOTBOX_TICKS_PER_DEGREE;
constexpr float kPidKp                  = HOTBOX_PID_KP;
constexpr float kPidKi                  = HOTBOX_PID_KI;
constexpr float kPidKd                  = HOTBOX_PID_KD;
constexpr float kPwmDeadband            = HOTBOX_PWM_DEADBAND;
constexpr float kPositionDeadbandDeg    = HOTBOX_POSITION_DEADBAND_DEG;
constexpr float kStallVelocityThreshDegS = HOTBOX_STALL_VELOCITY_THRESHOLD_DEG_S;
constexpr float kStallTimeoutS          = HOTBOX_STALL_TIMEOUT_S;

// Joint limits (from config/system.yaml). Azimuth limits are relative to oven-facing.
constexpr float kElevationMinDeg = HOTBOX_ELEVATION_MIN_DEG;
constexpr float kElevationMaxDeg = HOTBOX_ELEVATION_MAX_DEG;
constexpr float kAzimuthMinDeg = HOTBOX_AZIMUTH_MIN_DEG;
constexpr float kAzimuthMaxDeg = HOTBOX_AZIMUTH_MAX_DEG;

#if HOTBOX_NODE_ID == 0
constexpr float kOvenFacingAzimuthDeg = HOTBOX_OVEN_FACING_AZIMUTH_DEG_NODE_0;
#elif HOTBOX_NODE_ID == 1
constexpr float kOvenFacingAzimuthDeg = HOTBOX_OVEN_FACING_AZIMUTH_DEG_NODE_1;
#elif HOTBOX_NODE_ID == 2
constexpr float kOvenFacingAzimuthDeg = HOTBOX_OVEN_FACING_AZIMUTH_DEG_NODE_2;
#else
constexpr float kOvenFacingAzimuthDeg = 0.0f;
#endif

}  // namespace hotbox
