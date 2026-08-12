#include "axis.h"

#include <ESP32Encoder.h>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace hotbox {
namespace {

ESP32Encoder g_az_encoder;
ESP32Encoder g_el_encoder;

void hallGpioIsr(void* arg) {
  static_cast<BrushedAxis*>(arg)->onHallEdgeIsr();
}

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

constexpr float kHomingSeekTimeoutS = 30.0f;
/** Generous upper bound on magnet window width for leave-time budgeting [°]. */
constexpr float kHomingLeaveMaxWindowDeg = 20.0f;
/** Extra wall-clock slack on top of (window + clear) / velocity [s]. */
constexpr float kHomingLeaveTimeoutMarginS = 5.0f;

float homingLeaveTimeoutS() {
  const float v = fmaxf(kHomingVelocityDegS, 0.25f);
  return (kHomingLeaveMaxWindowDeg + kHomingClearDistanceDeg) / v +
         kHomingLeaveTimeoutMarginS;
}

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

  // Latch encoder count on hall edges so homing edges are not delayed by the
  // 50 Hz control poll. Active-low: LOW = magnet present.
  clearHallEdgeLatches();
  attachInterruptArg(
      digitalPinToInterrupt(hall_pin_), hallGpioIsr, this, CHANGE);
}

// Hall sensors are active-low open-collector (pulled up; magnet → LOW).
bool BrushedAxis::hallTriggered() const { return digitalRead(hall_pin_) == LOW; }

long BrushedAxis::encoderCountNow() const {
  return (enc_a_ == kHorizEncA) ? static_cast<long>(g_az_encoder.getCount())
                                : static_cast<long>(g_el_encoder.getCount());
}

void BrushedAxis::onHallEdgeIsr() {
  const long ticks = encoderCountNow();
  if (digitalRead(hall_pin_) == LOW) {
    hall_assert_ticks_ = ticks;
    hall_assert_pending_ = true;
  } else {
    hall_clear_ticks_ = ticks;
    hall_clear_pending_ = true;
  }
}

void BrushedAxis::clearHallEdgeLatches() {
  noInterrupts();
  hall_assert_pending_ = false;
  hall_clear_pending_ = false;
  interrupts();
}

bool BrushedAxis::takeHallAssertEdge(long* ticks_out) {
  noInterrupts();
  const bool pending = hall_assert_pending_;
  const long ticks = hall_assert_ticks_;
  hall_assert_pending_ = false;
  interrupts();
  if (!pending || ticks_out == nullptr) {
    return false;
  }
  *ticks_out = ticks;
  return true;
}

bool BrushedAxis::takeHallClearEdge(long* ticks_out) {
  noInterrupts();
  const bool pending = hall_clear_pending_;
  const long ticks = hall_clear_ticks_;
  hall_clear_pending_ = false;
  interrupts();
  if (!pending || ticks_out == nullptr) {
    return false;
  }
  *ticks_out = ticks;
  return true;
}

const char* BrushedAxis::homeStateText() const {
  if (mode_ == AxisMode::Fault) {
    return "fault";
  }
  if (mode_ == AxisMode::Homing) {
    return "homing";
  }
  if (homed_) {
    return "homed";
  }
  return "unhomed";
}

void BrushedAxis::enterHomingPhase(HomingPhase phase) {
  homing_phase_ = phase;
  homing_phase_s_ = 0.0f;
  homing_leave_cleared_ = false;
  homing_leave_clear_deg_ = 0.0f;
  // Drop stale edges so Seek/Across/Leave only see transitions in this phase.
  clearHallEdgeLatches();
  target_deg_ = position_deg_;
  command_velocity_deg_s_ = 0.0f;
  resetPidState();
  driveMotor(0.0f);
}

void BrushedAxis::startHoming() {
  homed_ = false;
  hall_width_deg_ = -1.0f;
  mode_ = AxisMode::Homing;
  clearFault();
  // If already on the switch, leave it first so the rising edge is a true edge.
  enterHomingPhase(hallTriggered() ? HomingPhase::LeaveSwitch : HomingPhase::Seek);
}

float BrushedAxis::homeAngleDeg() const {
  // Elevation home = 90° (face-up). Azimuth home = oven-facing so relative az is
  // 0° (center of the ±150° travel window) — not absolute north (0°), which sits
  // on the relative ±180° discontinuity and makes the post-home servo flee.
  return (enc_a_ == kHorizEncA) ? kOvenFacingAzimuthDeg : 90.0f;
}

void BrushedAxis::finishHoming(float mid_deg) {
  // Redefine the encoder so the hall-window midpoint maps to home_deg, without
  // physically jumping: current pose becomes home + (pos − mid). Then servo home.
  const float width_deg = fabs(homing_edge2_deg_ - homing_edge1_deg_);
  hall_width_deg_ = width_deg;
  const float home_deg = homeAngleDeg();
  const float new_pos = home_deg + (position_deg_ - mid_deg);
  const long home_at_mid_ticks = lroundf(new_pos * kTicksPerDegree);
  const bool is_azimuth = (enc_a_ == kHorizEncA);
  if (is_azimuth) {
    g_az_encoder.setCount(home_at_mid_ticks);
  } else {
    g_el_encoder.setCount(home_at_mid_ticks);
  }
  encoder_ticks_ = home_at_mid_ticks;
  last_encoder_ticks_ = home_at_mid_ticks;
  position_deg_ = new_pos;
  velocity_deg_s_ = 0.0f;
  homed_ = true;
  homing_phase_ = HomingPhase::Seek;
  resetPidState();
  Serial.print("{\"hotbox\":\"home_done\",\"axis\":\"");
  Serial.print(is_azimuth ? "az" : "el");
  Serial.print("\",\"hall_width_deg\":");
  Serial.print(width_deg, 3);
  Serial.println("}");
  // Drive back to the new zero (hall midpoint).
  setTargetDeg(home_deg);
}

void BrushedAxis::setTargetDeg(float target_deg) {
  target_deg_ = target_deg;
  command_velocity_deg_s_ = 0.0f;
  mode_ = AxisMode::Position;
  // Switching loops: clear shared-ish state so leftover I doesn't punch.
  resetPidState();
  clearFault();
}

void BrushedAxis::setVelocityDegS(float velocity_deg_s) {
  command_velocity_deg_s_ = velocity_deg_s;
  mode_ = AxisMode::Velocity;
  resetPidState();
  clearFault();
}

void BrushedAxis::stop() {
  mode_ = AxisMode::Idle;
  command_velocity_deg_s_ = 0.0f;
  homing_phase_ = HomingPhase::Seek;
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

void BrushedAxis::setVelocityPidGains(float kp, float ki, float kd) {
  kp_vel_ = kp;
  ki_vel_ = ki;
  kd_vel_ = kd;
}

void BrushedAxis::resetPidState() {
  integral_ = 0.0f;
  last_error_deg_ = 0.0f;
  pid_has_last_error_ = false;
  vel_integral_ = 0.0f;
  last_vel_error_ = 0.0f;
  vel_pid_has_last_error_ = false;
}

void BrushedAxis::setFault(const char* text) {
  fault_text_ = text;
  mode_ = AxisMode::Fault;
  homing_phase_ = HomingPhase::Seek;
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

float BrushedAxis::computePositionPidDuty(float error_deg, float dt_s, bool apply_position_deadband) {
  if (apply_position_deadband && kPositionDeadbandDeg > 0.0f &&
      fabs(error_deg) < kPositionDeadbandDeg) {
    // Coast near target; freeze I (don't clear) so small re-entries aren't jittery.
    last_error_deg_ = error_deg;
    pid_has_last_error_ = false;
    return 0.0f;
  }

  float d_error = 0.0f;
  if (pid_has_last_error_ && dt_s > 1e-6f) {
    d_error = (error_deg - last_error_deg_) / dt_s;
  }
  last_error_deg_ = error_deg;
  pid_has_last_error_ = true;

  float u = kp_ * error_deg + ki_ * integral_ + kd_ * d_error;
  const bool saturated = u > 1.0f || u < -1.0f;
  if (!saturated || (error_deg * integral_ <= 0.0f)) {
    integral_ += error_deg * dt_s;
  }
  // Hard-cap |ki·I| so a long slew cannot bank an I-term that drives overshoot.
  if (fabs(ki_) > 1e-12f && kPidIntegralLimit > 0.0f) {
    const float i_max = kPidIntegralLimit / fabs(ki_);
    integral_ = clampf(integral_, -i_max, i_max);
  }
  u = kp_ * error_deg + ki_ * integral_ + kd_ * d_error;
  return clampf(u, -1.0f, 1.0f);
}

float BrushedAxis::computeVelocityPidDuty(float target_velocity_deg_s, float dt_s) {
  const float error = target_velocity_deg_s - velocity_deg_s_;
  float d_error = 0.0f;
  if (vel_pid_has_last_error_ && dt_s > 1e-6f) {
    d_error = (error - last_vel_error_) / dt_s;
  }
  last_vel_error_ = error;
  vel_pid_has_last_error_ = true;

  float u = kp_vel_ * error + ki_vel_ * vel_integral_ + kd_vel_ * d_error;
  const bool saturated = u > 1.0f || u < -1.0f;
  if (!saturated || (error * vel_integral_ <= 0.0f)) {
    vel_integral_ += error * dt_s;
  }
  // Reuse the same duty-fraction I-cap as the position loop.
  if (fabs(ki_vel_) > 1e-12f && kPidIntegralLimit > 0.0f) {
    const float i_max = kPidIntegralLimit / fabs(ki_vel_);
    vel_integral_ = clampf(vel_integral_, -i_max, i_max);
  }
  u = kp_vel_ * error + ki_vel_ * vel_integral_ + kd_vel_ * d_error;
  return clampf(u, -1.0f, 1.0f);
}

float BrushedAxis::limitAwareVelocityCommand(float commanded_deg_s) const {
  if (enc_a_ == kHorizEncA) {
    const float rel = wrap180(position_deg_ - kOvenFacingAzimuthDeg);
    if (rel >= kAzimuthMaxDeg && commanded_deg_s > 0.0f) {
      return 0.0f;
    }
    if (rel <= kAzimuthMinDeg && commanded_deg_s < 0.0f) {
      return 0.0f;
    }
    return commanded_deg_s;
  }
  if (position_deg_ >= kElevationMaxDeg && commanded_deg_s > 0.0f) {
    return 0.0f;
  }
  if (position_deg_ <= kElevationMinDeg && commanded_deg_s < 0.0f) {
    return 0.0f;
  }
  return commanded_deg_s;
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
  velocity_deg_s_ = (dt_s > 1e-6f) ? static_cast<float>(delta_ticks) / kTicksPerDegree / dt_s : 0.0f;
  if (delta_ticks != 0) {
    encoder_alive_ = true;
  }

  if (mode_ == AxisMode::Homing) {
    homing_phase_s_ += dt_s;

    switch (homing_phase_) {
      case HomingPhase::LeaveSwitch: {
        // Leave an already-asserted hall, then continue clear-distance so Seek
        // approaches with established +velocity (not from a dead stop on the edge).
        long clear_ticks = 0;
        if (!homing_leave_cleared_) {
          if (takeHallClearEdge(&clear_ticks)) {
            if (!hallTriggered()) {
              homing_leave_cleared_ = true;
              homing_leave_clear_deg_ =
                  static_cast<float>(clear_ticks) / kTicksPerDegree;
            }
            // else bounce/noise — wait for a stable clear
          } else if (!hallTriggered()) {
            // Poll fallback if the clear edge ISR was missed.
            homing_leave_cleared_ = true;
            homing_leave_clear_deg_ = position_deg_;
          }
        }
        if (homing_leave_cleared_ &&
            fabs(position_deg_ - homing_leave_clear_deg_) >=
                kHomingClearDistanceDeg) {
          enterHomingPhase(HomingPhase::Seek);
          return;
        }
        if (homing_phase_s_ > homingLeaveTimeoutS()) {
          // Still on hall → true stuck; after clear this is usually too-slow leave.
          setFault(homing_leave_cleared_ ? "homing_timeout" : "hall_stuck");
          return;
        }
        // Opposite of Seek so we clear the magnet before driving back through it.
        command_velocity_deg_s_ = -kHomingVelocityDegS;
        break;
      }

      case HomingPhase::Seek: {
        // Rising edge into the magnet while seeking more-positive encoder.
        long assert_ticks = 0;
        if (takeHallAssertEdge(&assert_ticks)) {
          if (hallTriggered()) {
            homing_edge1_deg_ =
                static_cast<float>(assert_ticks) / kTicksPerDegree;
            enterHomingPhase(HomingPhase::Across);
            return;
          }
          // else bounce/noise — ignore
        } else if (hallTriggered()) {
          // Poll fallback if the assert edge ISR was missed.
          homing_edge1_deg_ = position_deg_;
          enterHomingPhase(HomingPhase::Across);
          return;
        }
        if (homing_phase_s_ > kHomingSeekTimeoutS) {
          setFault("hall_not_found");
          return;
        }
        command_velocity_deg_s_ = kHomingVelocityDegS;
        break;
      }

      case HomingPhase::Across: {
        // Falling edge: continue the same way across the magnet window.
        long clear_ticks = 0;
        if (takeHallClearEdge(&clear_ticks)) {
          if (!hallTriggered()) {
            homing_edge2_deg_ =
                static_cast<float>(clear_ticks) / kTicksPerDegree;
            const float mid_deg = 0.5f * (homing_edge1_deg_ + homing_edge2_deg_);
            finishHoming(mid_deg);
            return;
          }
          // else bounce/noise — ignore
        } else if (!hallTriggered()) {
          // Poll fallback if the clear edge ISR was missed.
          homing_edge2_deg_ = position_deg_;
          const float mid_deg = 0.5f * (homing_edge1_deg_ + homing_edge2_deg_);
          finishHoming(mid_deg);
          return;
        }
        if (homing_phase_s_ > kHomingSeekTimeoutS) {
          setFault("hall_stuck");
          return;
        }
        command_velocity_deg_s_ = kHomingVelocityDegS;
        break;
      }
    }

    // Constant-speed stages: regulate measured shaft rate with the velocity PID.
    const float pwm_command = computeVelocityPidDuty(command_velocity_deg_s_, dt_s);
    driveMotor(pwm_command);
    return;
  }

  if (mode_ == AxisMode::Fault) {
    driveMotor(0.0f);
    return;
  }

  if (mode_ == AxisMode::Velocity) {
    float limited_cmd = limitAwareVelocityCommand(command_velocity_deg_s_);
    if (fabs(limited_cmd) < 1e-6f && fabs(command_velocity_deg_s_) > 1e-6f) {
      // Hard stop: don't wind the velocity integrator into the bumper.
      vel_integral_ = 0.0f;
      vel_pid_has_last_error_ = false;
    }
    const float pwm_command = computeVelocityPidDuty(limited_cmd, dt_s);
    driveMotor(pwm_command);
    if (kStallTimeoutS > 0.0f) {
      if (encoder_alive_ && fabs(limited_cmd) > 1.0f &&
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

  if (mode_ == AxisMode::Position) {
    float error_deg = target_deg_ - position_deg_;
    if (enc_a_ == kHorizEncA) {
      error_deg = limited_azimuth_error_deg(target_deg_, position_deg_);
    }
    const float pwm_command = computePositionPidDuty(error_deg, dt_s, /*apply_position_deadband=*/true);
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
  azimuth_.setVelocityPidGains(pid_velocity_kp_, pid_velocity_ki_, pid_velocity_kd_);
  elevation_.setVelocityPidGains(pid_velocity_kp_, pid_velocity_ki_, pid_velocity_kd_);
}

void MirrorMount::home() {
  homeAzimuth();
  homeElevation();
}

void MirrorMount::homeAzimuth() {
  azimuth_.startHoming();
  refreshModeText();
  Serial.print("{\"hotbox\":\"home_start\",\"axis\":\"az\",\"az_hall\":");
  Serial.print(azimuth_.hallTriggered() ? "true" : "false");
  Serial.println("}");
}

void MirrorMount::homeElevation() {
  elevation_.startHoming();
  refreshModeText();
  Serial.print("{\"hotbox\":\"home_start\",\"axis\":\"el\",\"el_hall\":");
  Serial.print(elevation_.hallTriggered() ? "true" : "false");
  Serial.println("}");
}

void MirrorMount::stop() {
  azimuth_.stop();
  elevation_.stop();
  refreshModeText();
}

bool MirrorMount::setTarget(float azimuth_deg, float elevation_deg) {
  if (!isHomed()) {
    return false;
  }
  clamp_joint_targets(&azimuth_deg, &elevation_deg);
  azimuth_.setTargetDeg(azimuth_deg);
  elevation_.setTargetDeg(elevation_deg);
  refreshModeText();
  return true;
}

bool MirrorMount::setVelocity(float azimuth_deg_s, float elevation_deg_s) {
  if (!isHomed()) {
    return false;
  }
  azimuth_.setVelocityDegS(azimuth_deg_s);
  elevation_.setVelocityDegS(elevation_deg_s);
  refreshModeText();
  return true;
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

void MirrorMount::setVelocityPid(float kp, float ki, float kd) {
  pid_velocity_kp_ = kp;
  pid_velocity_ki_ = ki;
  pid_velocity_kd_ = kd;
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
  } else if (azimuth_.mode() == AxisMode::Velocity || elevation_.mode() == AxisMode::Velocity) {
    mode_text_ = "velocity";
  } else if (azimuth_.mode() == AxisMode::Position || elevation_.mode() == AxisMode::Position) {
    mode_text_ = "position";
  } else {
    mode_text_ = "idle";
  }
}

const char* MirrorMount::faultText() const {
  const char* az = azimuth_.faultText();
  const char* el = elevation_.faultText();
  if (az == nullptr && el == nullptr) {
    return nullptr;
  }
  // Single-threaded; emitStatus/CAN readers copy before the next update.
  static char buf[80];
  if (az != nullptr && el != nullptr) {
    snprintf(buf, sizeof(buf), "az:%s;el:%s", az, el);
  } else if (az != nullptr) {
    snprintf(buf, sizeof(buf), "az:%s", az);
  } else {
    snprintf(buf, sizeof(buf), "el:%s", el);
  }
  return buf;
}

}  // namespace hotbox
