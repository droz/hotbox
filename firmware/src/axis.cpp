#include "axis.h"

#include <ESP32Encoder.h>
#include <cstring>

namespace hotbox {
namespace {

ESP32Encoder g_az_encoder;
ESP32Encoder g_el_encoder;

float clampf(float value, float min_value, float max_value) {
  if (value < min_value) return min_value;
  if (value > max_value) return max_value;
  return value;
}

float wrap360(float deg) {
  while (deg < 0.0f) deg += 360.0f;
  while (deg >= 360.0f) deg -= 360.0f;
  return deg;
}

float wrap180(float deg) {
  while (deg > 180.0f) deg -= 360.0f;
  while (deg <= -180.0f) deg += 360.0f;
  return deg;
}

void clamp_joint_targets(float* azimuth_deg, float* elevation_deg) {
  *elevation_deg = clampf(*elevation_deg, kElevationMinDeg, kElevationMaxDeg);
  const float rel = wrap180(*azimuth_deg - kOvenFacingAzimuthDeg);
  const float rel_clamped = clampf(rel, kAzimuthMinDeg, kAzimuthMaxDeg);
  // Keep continuous oven+rel (may be outside [0,360)) so the servo can travel
  // through 0° inside the valid window instead of the forbidden back side.
  *azimuth_deg = kOvenFacingAzimuthDeg + rel_clamped;
}

// Azimuth error that stays inside the joint travel window (no shortest-path wrap
// through ±180° relative to oven-facing).
float limited_azimuth_error_deg(float target_deg, float position_deg) {
  const float target_rel = wrap180(target_deg - kOvenFacingAzimuthDeg);
  const float position_rel = wrap180(position_deg - kOvenFacingAzimuthDeg);
  return target_rel - position_rel;
}

constexpr float kHomingPwm = 0.15f;
constexpr float kHomingBackoffTimeoutS = 8.0f;
constexpr float kHomingSeekTimeoutS = 30.0f;

}  // namespace

BrushedAxis::BrushedAxis(int motor_p, int motor_m, int enc_a, int enc_b, int hall_pin)
    : motor_p_(motor_p), motor_m_(motor_m), enc_a_(enc_a), enc_b_(enc_b), hall_pin_(hall_pin) {}

void BrushedAxis::begin() {
  pinMode(motor_p_, OUTPUT);
  pinMode(motor_m_, OUTPUT);
  // Arduino-ESP32 on this core: global frequency for subsequent analogWrite().
  analogWriteFrequency(static_cast<uint32_t>(kMotorPwmHz));
  analogWrite(motor_p_, 0);
  analogWrite(motor_m_, 0);
  // Active-low halls: keep internal pull-up so open wires read not-triggered.
  pinMode(hall_pin_, INPUT_PULLUP);
  // ESP32Encoder configures PCNT with raw GPIO numbers (not Arduino D# labels).
  ESP32Encoder::useInternalWeakPullResistors = puType::up;
  const int gpio_a = pinToGpio(enc_a_);
  const int gpio_b = pinToGpio(enc_b_);
  if (enc_a_ == kHorizEncA) {
    g_az_encoder.attachFullQuad(gpio_a, gpio_b);
    g_az_encoder.setCount(0);
    encoder_ticks_ = g_az_encoder.getCount();
  } else {
    g_el_encoder.attachFullQuad(gpio_a, gpio_b);
    g_el_encoder.setCount(0);
    encoder_ticks_ = g_el_encoder.getCount();
  }
  last_encoder_ticks_ = encoder_ticks_;
  position_deg_ = static_cast<float>(encoder_ticks_) / kTicksPerDegree;
}

// Hall sensors are active-low open-collector (pulled up; magnet → LOW).
bool BrushedAxis::hallTriggered() const { return digitalRead(hall_pin_) == LOW; }

void BrushedAxis::startHoming() {
  homed_ = false;
  mode_ = AxisMode::Homing;
  homing_phase_s_ = 0.0f;
  // If already on the switch (or the pin is stuck low), back off first so we
  // don't "succeed" instantly with motors never driven.
  homing_backoff_ = hallTriggered();
  resetPidState();
  clearFault();
}

void BrushedAxis::finishHoming() {
  if (enc_a_ == kHorizEncA) {
    g_az_encoder.setCount(0);
  } else {
    g_el_encoder.setCount(0);
  }
  encoder_ticks_ = 0;
  last_encoder_ticks_ = 0;
  position_deg_ = 0.0f;
  velocity_deg_s_ = 0.0f;
  homed_ = true;
  mode_ = AxisMode::Idle;
  homing_backoff_ = false;
  resetPidState();
  driveMotor(0.0f);
}

void BrushedAxis::setTargetDeg(float target_deg) {
  target_deg_ = target_deg;
  mode_ = AxisMode::Position;
  clearFault();
}

void BrushedAxis::stop() {
  mode_ = AxisMode::Idle;
  command_velocity_deg_s_ = 0.0f;
  homing_backoff_ = false;
  resetPidState();
  driveMotor(0.0f);
}

void BrushedAxis::clearFault() {
  fault_text_ = nullptr;
  stall_timer_s_ = 0.0f;
}

void BrushedAxis::setPidGains(float kp, float ki, float kd) {
  kp_ = kp;
  ki_ = ki;
  kd_ = kd;
}

void BrushedAxis::resetPidState() {
  integral_ = 0.0f;
  last_error_deg_ = 0.0f;
  pid_has_last_error_ = false;
}

void BrushedAxis::setFault(const char* text) {
  fault_text_ = text;
  mode_ = AxisMode::Fault;
  homing_backoff_ = false;
  resetPidState();
  driveMotor(0.0f);
}

void BrushedAxis::driveMotor(float command) {
  command = clampf(command, -1.0f, 1.0f);
  // Below deadband: both legs fully off (no PWM edge activity → lower quiescent draw).
  if (fabs(command) < kPwmDeadband) {
    analogWrite(motor_p_, 0);
    analogWrite(motor_m_, 0);
    return;
  }
  int pwm = static_cast<int>(fabs(command) * 255.0f);
  if (command > 0.0f) {
    analogWrite(motor_p_, pwm);
    analogWrite(motor_m_, 0);
  } else {
    analogWrite(motor_p_, 0);
    analogWrite(motor_m_, pwm);
  }
}

void BrushedAxis::update(float dt_s) {
  if (enc_a_ == kHorizEncA) {
    encoder_ticks_ = g_az_encoder.getCount();
  } else {
    encoder_ticks_ = g_el_encoder.getCount();
  }

  const long delta_ticks = encoder_ticks_ - last_encoder_ticks_;
  last_encoder_ticks_ = encoder_ticks_;
  position_deg_ = static_cast<float>(encoder_ticks_) / kTicksPerDegree;
  velocity_deg_s_ = static_cast<float>(delta_ticks) / kTicksPerDegree / dt_s;
  if (delta_ticks != 0) {
    encoder_alive_ = true;
  }

  if (mode_ == AxisMode::Homing) {
    homing_phase_s_ += dt_s;
    if (homing_backoff_) {
      // Leave the hall switch (or a stuck-low pin) before searching for home.
      if (!hallTriggered()) {
        homing_backoff_ = false;
        homing_phase_s_ = 0.0f;
        driveMotor(0.0f);
        return;
      }
      if (homing_phase_s_ > kHomingBackoffTimeoutS) {
        setFault("hall_stuck");
        return;
      }
      command_velocity_deg_s_ = -kHomingVelocityDegS;
      driveMotor(-kHomingPwm);
      return;
    }

    // Seek toward the hall edge.
    if (hallTriggered()) {
      finishHoming();
      return;
    }
    if (homing_phase_s_ > kHomingSeekTimeoutS) {
      setFault("hall_not_found");
      return;
    }
    command_velocity_deg_s_ = kHomingVelocityDegS;
    driveMotor(kHomingPwm);
    return;
  }

  if (mode_ == AxisMode::Fault) {
    driveMotor(0.0f);
    return;
  }

  if (mode_ == AxisMode::Position) {
    float error_deg = target_deg_ - position_deg_;
    if (enc_a_ == kHorizEncA) {
      error_deg = limited_azimuth_error_deg(target_deg_, position_deg_);
    }

    // Close enough: fully coast. Freeze I (don't clear) so small re-entries
    // don't have to re-wind the integrator from zero (that felt jittery).
    // Reset D history only so de/dt isn't a spike when leaving the band.
    if (kPositionDeadbandDeg > 0.0f && fabs(error_deg) < kPositionDeadbandDeg) {
      last_error_deg_ = error_deg;
      pid_has_last_error_ = false;
      driveMotor(0.0f);
      command_velocity_deg_s_ = 0.0f;
      stall_timer_s_ = 0.0f;
      return;
    }

    float d_error = 0.0f;
    if (pid_has_last_error_ && dt_s > 1e-6f) {
      d_error = (error_deg - last_error_deg_) / dt_s;
    }
    last_error_deg_ = error_deg;
    pid_has_last_error_ = true;

    // Duty fraction u ∈ [-1, 1]: kp*e + ki*∫e + kd*de/dt with integrator anti-windup.
    float u = kp_ * error_deg + ki_ * integral_ + kd_ * d_error;
    const bool saturated = u > 1.0f || u < -1.0f;
    if (!saturated || (error_deg * integral_ <= 0.0f)) {
      integral_ += error_deg * dt_s;
    }
    u = kp_ * error_deg + ki_ * integral_ + kd_ * d_error;
    const float pwm_command = clampf(u, -1.0f, 1.0f);
    driveMotor(pwm_command);
    command_velocity_deg_s_ = pwm_command * kMaxVelocityDegS;
    if (kStallTimeoutS > 0.0f) {
      if (encoder_alive_ && fabs(command_velocity_deg_s_) > 1.0f &&
          fabs(velocity_deg_s_) < kStallVelocityThreshDegS) {
        stall_timer_s_ += dt_s;
      } else {
        stall_timer_s_ = 0.0f;
      }
      if (stall_timer_s_ > kStallTimeoutS) {
        setFault("stalled");
      }
    } else {
      stall_timer_s_ = 0.0f;
    }
    return;
  }

  driveMotor(0.0f);
}

MirrorMount::MirrorMount()
    : azimuth_(kHorizMotorP, kHorizMotorM, kHorizEncA, kHorizEncB, kHorizHall),
      elevation_(kVertMotorP, kVertMotorM, kVertEncA, kVertEncB, kVertHall) {}

void MirrorMount::begin() {
  applyPidGains();
  azimuth_.begin();
  elevation_.begin();
}

void MirrorMount::applyPidGains() {
  azimuth_.setPidGains(pid_kp_, pid_ki_, pid_kd_);
  elevation_.setPidGains(pid_kp_, pid_ki_, pid_kd_);
}

void MirrorMount::home() {
  azimuth_.startHoming();
  elevation_.startHoming();
  refreshModeText();
  Serial.print("{\"hotbox\":\"home_start\",\"az_hall\":");
  Serial.print(azimuth_.hallTriggered() ? "true" : "false");
  Serial.print(",\"el_hall\":");
  Serial.print(elevation_.hallTriggered() ? "true" : "false");
  Serial.println("}");
}

void MirrorMount::stop() {
  azimuth_.stop();
  elevation_.stop();
  refreshModeText();
}

void MirrorMount::setTarget(float azimuth_deg, float elevation_deg) {
  clamp_joint_targets(&azimuth_deg, &elevation_deg);
  azimuth_.setTargetDeg(azimuth_deg);
  elevation_.setTargetDeg(elevation_deg);
  refreshModeText();
}

void MirrorMount::clearError() {
  azimuth_.clearFault();
  elevation_.clearFault();
  azimuth_.stop();
  elevation_.stop();
  refreshModeText();
}

void MirrorMount::setPid(float kp, float ki, float kd) {
  pid_kp_ = kp;
  pid_ki_ = ki;
  pid_kd_ = kd;
  applyPidGains();
}

void MirrorMount::reset() {
  azimuth_.clearFault();
  elevation_.clearFault();
  azimuth_.stop();
  elevation_.stop();
  azimuth_.resetPidState();
  elevation_.resetPidState();
  refreshModeText();
#if !defined(NATIVE_CIL)
  Serial.println("{\"hotbox\":\"reset\",\"via\":\"soft\"}");
  Serial.flush();
  delay(20);
  ESP.restart();
#endif
}

void MirrorMount::update(float dt_s) {
  azimuth_.update(dt_s);
  elevation_.update(dt_s);
  refreshModeText();
}

void MirrorMount::refreshModeText() {
  if (azimuth_.mode() == AxisMode::Fault || elevation_.mode() == AxisMode::Fault) {
    mode_text_ = "fault";
  } else if (azimuth_.mode() == AxisMode::Homing || elevation_.mode() == AxisMode::Homing) {
    mode_text_ = "homing";
  } else if (azimuth_.mode() == AxisMode::Position || elevation_.mode() == AxisMode::Position) {
    mode_text_ = "position";
  } else {
    mode_text_ = "idle";
  }
}

const char* MirrorMount::faultText() const {
  if (azimuth_.faultText() != nullptr) return azimuth_.faultText();
  if (elevation_.faultText() != nullptr) return elevation_.faultText();
  return nullptr;
}

}  // namespace hotbox
