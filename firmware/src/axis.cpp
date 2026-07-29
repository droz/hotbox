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

float wrapped_error_deg(float target_deg, float position_deg) {
  float error = target_deg - position_deg;
  while (error > 180.0f) error -= 360.0f;
  while (error <= -180.0f) error += 360.0f;
  return error;
}

}  // namespace

BrushedAxis::BrushedAxis(int motor_p, int motor_m, int enc_a, int enc_b, int hall_pin)
    : motor_p_(motor_p), motor_m_(motor_m), enc_a_(enc_a), enc_b_(enc_b), hall_pin_(hall_pin) {}

void BrushedAxis::begin() {
  pinMode(motor_p_, OUTPUT);
  pinMode(motor_m_, OUTPUT);
  pinMode(hall_pin_, INPUT);
  if (enc_a_ == kHorizEncA) {
    g_az_encoder.attachFullQuad(enc_a_, enc_b_);
    g_az_encoder.setCount(0);
    encoder_ticks_ = g_az_encoder.getCount();
  } else {
    g_el_encoder.attachFullQuad(enc_a_, enc_b_);
    g_el_encoder.setCount(0);
    encoder_ticks_ = g_el_encoder.getCount();
  }
  last_encoder_ticks_ = encoder_ticks_;
  position_deg_ = static_cast<float>(encoder_ticks_) / kTicksPerDegree;
}

bool BrushedAxis::hallTriggered() const { return digitalRead(hall_pin_) == HIGH; }

void BrushedAxis::startHoming() {
  homed_ = false;
  mode_ = AxisMode::Homing;
  clearFault();
}

void BrushedAxis::setTargetDeg(float target_deg) {
  target_deg_ = target_deg;
  mode_ = AxisMode::Tracking;
  clearFault();
}

void BrushedAxis::setJogRateDegS(float rate_deg_s) {
  jog_rate_deg_s_ = clampf(rate_deg_s, -kMaxVelocityDegS, kMaxVelocityDegS);
  mode_ = AxisMode::Jog;
  clearFault();
}

void BrushedAxis::stop() {
  mode_ = AxisMode::Idle;
  command_velocity_deg_s_ = 0.0f;
  driveMotor(0.0f);
}

void BrushedAxis::clearFault() { fault_text_ = nullptr; }

void BrushedAxis::setFault(const char* text) {
  fault_text_ = text;
  mode_ = AxisMode::Fault;
  driveMotor(0.0f);
}

void BrushedAxis::driveMotor(float command) {
  command = clampf(command, -1.0f, 1.0f);
  int pwm = static_cast<int>(fabs(command) * 255.0f);
  if (command > 0.01f) {
    analogWrite(motor_p_, pwm);
    analogWrite(motor_m_, 0);
  } else if (command < -0.01f) {
    analogWrite(motor_p_, 0);
    analogWrite(motor_m_, pwm);
  } else {
    analogWrite(motor_p_, 0);
    analogWrite(motor_m_, 0);
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

  if (mode_ == AxisMode::Homing) {
    if (hallTriggered()) {
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
      driveMotor(0.0f);
      return;
    }
    command_velocity_deg_s_ = kHomingVelocityDegS;
    driveMotor(0.15f);
    return;
  }

  if (mode_ == AxisMode::Fault) {
    driveMotor(0.0f);
    return;
  }

  if (mode_ == AxisMode::Jog) {
    command_velocity_deg_s_ = jog_rate_deg_s_;
    driveMotor(command_velocity_deg_s_ / kMaxVelocityDegS);
    return;
  }

  if (mode_ == AxisMode::Tracking) {
    float error_deg = target_deg_ - position_deg_;
    if (enc_a_ == kHorizEncA) {
      error_deg = wrapped_error_deg(target_deg_, position_deg_);
    }
    const float pwm_command = clampf(error_deg / 10.0f, -1.0f, 1.0f);
    driveMotor(pwm_command);
    command_velocity_deg_s_ = pwm_command * kMaxVelocityDegS;
    if (fabs(command_velocity_deg_s_) > 1.0f && fabs(velocity_deg_s_) < kStallVelocityThreshDegS) {
      stall_timer_s_ += dt_s;
    } else {
      stall_timer_s_ = 0.0f;
    }
    if (stall_timer_s_ > kStallTimeoutS) {
      setFault("stalled");
    }
    return;
  }

  driveMotor(0.0f);
}

MirrorMount::MirrorMount()
    : azimuth_(kHorizMotorP, kHorizMotorM, kHorizEncA, kHorizEncB, kHorizHall),
      elevation_(kVertMotorP, kVertMotorM, kVertEncA, kVertEncB, kVertHall) {}

void MirrorMount::begin() {
  azimuth_.begin();
  elevation_.begin();
}

void MirrorMount::home() {
  mode_text_ = "homing";
  azimuth_.startHoming();
  elevation_.startHoming();
}

void MirrorMount::stop() {
  mode_text_ = "idle";
  azimuth_.stop();
  elevation_.stop();
}

void MirrorMount::setTarget(float azimuth_deg, float elevation_deg, const char* mode_text) {
  mode_text_ = mode_text;
  azimuth_.setTargetDeg(azimuth_deg);
  elevation_.setTargetDeg(elevation_deg);
}

void MirrorMount::jog(float azimuth_rate_deg_s, float elevation_rate_deg_s) {
  mode_text_ = "jog";
  azimuth_.setJogRateDegS(azimuth_rate_deg_s);
  elevation_.setJogRateDegS(elevation_rate_deg_s);
}

void MirrorMount::clearError() {
  mode_text_ = "idle";
  azimuth_.clearFault();
  elevation_.clearFault();
  azimuth_.stop();
  elevation_.stop();
}

void MirrorMount::update(float dt_s) {
  azimuth_.update(dt_s);
  elevation_.update(dt_s);
  if (azimuth_.mode() == AxisMode::Fault || elevation_.mode() == AxisMode::Fault) {
    mode_text_ = "fault";
  } else if (strcmp(mode_text_, "homing") == 0 && isHomed()) {
    mode_text_ = "idle";
  }
}

const char* MirrorMount::faultText() const {
  if (azimuth_.faultText() != nullptr) return azimuth_.faultText();
  if (elevation_.faultText() != nullptr) return elevation_.faultText();
  return nullptr;
}

}  // namespace hotbox
