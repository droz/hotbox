#pragma once

#include "config.h"

namespace hotbox {

// Minimal axis states. Product modes (track/park/jog) live on the host;
// firmware only servos a position, homes, idles, or faults.
enum class AxisMode { Idle, Homing, Position, Fault };

class BrushedAxis {
 public:
  BrushedAxis(int motor_p, int motor_m, int enc_a, int enc_b, int hall_pin);

  void begin();
  void startHoming();
  void setTargetDeg(float target_deg);
  void stop();
  void clearFault();
  void update(float dt_s);

  void setPidGains(float kp, float ki, float kd);
  void resetPidState();

  float positionDeg() const { return position_deg_; }
  /** Integral term contribution to duty command (ki * ∫error dt), before clamp. */
  float integralTerm() const { return ki_ * integral_; }
  bool isHomed() const { return homed_; }
  bool hallTriggered() const;
  AxisMode mode() const { return mode_; }
  const char* faultText() const { return fault_text_; }

 private:
  void driveMotor(float command);
  void setFault(const char* text);
  void finishHoming();

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
  float command_velocity_deg_s_ = 0.0f;
  float stall_timer_s_ = 0.0f;
  float homing_phase_s_ = 0.0f;
  bool homed_ = false;
  // Stall detect only after the encoder has moved at least once — otherwise a
  // missing encoder would immediately fault any open-loop PWM bring-up.
  bool encoder_alive_ = false;
  // True while leaving an already-asserted hall before the seek toward home.
  bool homing_backoff_ = false;
  AxisMode mode_ = AxisMode::Idle;
  const char* fault_text_ = nullptr;

  float kp_ = kPidKp;
  float ki_ = kPidKi;
  float kd_ = kPidKd;
  float integral_ = 0.0f;
  float last_error_deg_ = 0.0f;
  bool pid_has_last_error_ = false;
};

class MirrorMount {
 public:
  MirrorMount();

  void begin();
  void home();
  void stop();
  void setTarget(float azimuth_deg, float elevation_deg);
  void clearError();
  /** Soft reset: stop, clear faults/PID state. On device also reboots via ESP.restart(). */
  void reset();
  void setPid(float kp, float ki, float kd);
  void update(float dt_s);

  float azimuthDeg() const { return azimuth_.positionDeg(); }
  float elevationDeg() const { return elevation_.positionDeg(); }
  float azimuthIntegralTerm() const { return azimuth_.integralTerm(); }
  float elevationIntegralTerm() const { return elevation_.integralTerm(); }
  float pidKp() const { return pid_kp_; }
  float pidKi() const { return pid_ki_; }
  float pidKd() const { return pid_kd_; }
  bool isHomed() const { return azimuth_.isHomed() && elevation_.isHomed(); }
  bool azimuthHallTriggered() const { return azimuth_.hallTriggered(); }
  bool elevationHallTriggered() const { return elevation_.hallTriggered(); }
  const char* modeText() const { return mode_text_; }
  const char* faultText() const;

 private:
  void refreshModeText();
  void applyPidGains();

  BrushedAxis azimuth_;
  BrushedAxis elevation_;
  float pid_kp_ = kPidKp;
  float pid_ki_ = kPidKi;
  float pid_kd_ = kPidKd;
  const char* mode_text_ = "idle";
};

}  // namespace hotbox
