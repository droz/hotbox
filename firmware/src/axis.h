#pragma once

#include "config.h"

namespace hotbox {

enum class AxisMode { Idle, Homing, Tracking, Jog, Fault };

class BrushedAxis {
 public:
  BrushedAxis(int motor_p, int motor_m, int enc_a, int enc_b, int hall_pin);

  void begin();
  void startHoming();
  void setTargetDeg(float target_deg);
  void setJogRateDegS(float rate_deg_s);
  void stop();
  void clearFault();
  void update(float dt_s);

  float positionDeg() const { return position_deg_; }
  bool isHomed() const { return homed_; }
  bool hallTriggered() const;
  AxisMode mode() const { return mode_; }
  const char* faultText() const { return fault_text_; }

 private:
  void driveMotor(float command);
  void setFault(const char* text);

  int motor_p_;
  int motor_m_;
  int enc_a_;
  int enc_b_;
  int hall_pin_;
  long encoder_ticks_ = 0;
  long last_encoder_ticks_ = 0;
  float position_deg_ = 0.0f;
  float velocity_deg_s_ = 0.0f;
  float target_deg_ = 0.0f;
  float jog_rate_deg_s_ = 0.0f;
  float command_velocity_deg_s_ = 0.0f;
  float stall_timer_s_ = 0.0f;
  bool homed_ = false;
  AxisMode mode_ = AxisMode::Idle;
  const char* fault_text_ = nullptr;
};

class MirrorMount {
 public:
  MirrorMount();

  void begin();
  void home();
  void stop();
  void setTarget(float azimuth_deg, float elevation_deg, const char* mode_text);
  void jog(float azimuth_rate_deg_s, float elevation_rate_deg_s);
  void clearError();
  void update(float dt_s);

  float azimuthDeg() const { return azimuth_.positionDeg(); }
  float elevationDeg() const { return elevation_.positionDeg(); }
  bool isHomed() const { return azimuth_.isHomed() && elevation_.isHomed(); }
  const char* modeText() const { return mode_text_; }
  const char* faultText() const;

 private:
  BrushedAxis azimuth_;
  BrushedAxis elevation_;
  const char* mode_text_ = "idle";
};

}  // namespace hotbox
