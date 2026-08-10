#pragma once

#include "config.h"

namespace hotbox {

// Minimal axis states. Product modes (track/park/jog) live on the host;
// firmware servos position, velocity, homes, idles, or faults.
enum class AxisMode { Idle, Homing, Position, Velocity, Fault };

/** Homing sub-states while AxisMode::Homing. */
enum class HomingPhase {
  LeaveSwitch,  // started on hall — reverse until clear, then Seek
  Seek,         // constant-speed (positive encoder) until rising hall edge
  Across,       // continue until falling hall edge, then zero + go home
};

class BrushedAxis {
 public:
  BrushedAxis(int motor_p, int motor_m, int enc_a, int enc_b, int hall_pin);

  void begin();
  void startHoming();
  void setTargetDeg(float target_deg);
  void setVelocityDegS(float velocity_deg_s);
  void stop();
  void clearFault();
  void update(float dt_s);

  void setPidGains(float kp, float ki, float kd);
  void resetPidState();

  float positionDeg() const { return position_deg_; }
  float velocityDegS() const { return velocity_deg_s_; }
  /** Integral term contribution to duty command (position loop), before clamp. */
  float integralTerm() const { return ki_ * integral_; }
  bool isHomed() const { return homed_; }
  /** unhomed | homing | homed | fault */
  const char* homeStateText() const;
  bool hallTriggered() const;
  AxisMode mode() const { return mode_; }
  const char* faultText() const { return fault_text_; }

 private:
  void driveMotor(float command);
  void setFault(const char* text);
  void finishHoming(float mid_deg);
  void enterHomingPhase(HomingPhase phase);
  /** Home joint angle for this axis (az = oven-facing, el = 90°). */
  float homeAngleDeg() const;
  /** Position PID → duty ∈ [-1,1]. When ``apply_position_deadband``, coast+freeze I near zero error. */
  float computePositionPidDuty(float error_deg, float dt_s, bool apply_position_deadband);
  /**
   * Velocity PID → duty ∈ [-1,1].
   *
   * Gains are derived from the position PID by matching SI units:
   *   kp_vel = kd_pos   [duty / (deg/s)]
   *   ki_vel = kp_pos   [duty / deg]  (∫ velocity error dt has units deg)
   *   kd_vel = 0
   */
  float computeVelocityPidDuty(float target_velocity_deg_s, float dt_s);
  void syncVelocityGainsFromPosition();
  /**
   * Zero any velocity that would drive further past joint limits (elevation box
   * or azimuth relative to oven-facing). Homing does not use this.
   */
  float limitAwareVelocityCommand(float commanded_deg_s) const;

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
  /** Hall window edges captured during Seek / Across (encoder degrees). */
  float homing_edge1_deg_ = 0.0f;
  float homing_edge2_deg_ = 0.0f;
  bool homed_ = false;
  // Stall detect only after the encoder has moved at least once — otherwise a
  // missing encoder would immediately fault any open-loop PWM bring-up.
  bool encoder_alive_ = false;
  HomingPhase homing_phase_ = HomingPhase::Seek;
  AxisMode mode_ = AxisMode::Idle;
  const char* fault_text_ = nullptr;

  // Position-loop gains (from config / set_pid).
  float kp_ = kPidKp;
  float ki_ = kPidKi;
  float kd_ = kPidKd;
  float integral_ = 0.0f;
  float last_error_deg_ = 0.0f;
  bool pid_has_last_error_ = false;

  // Velocity-loop gains (derived from position gains).
  float kp_vel_ = kPidKd;
  float ki_vel_ = kPidKp;
  float kd_vel_ = 0.0f;
  float vel_integral_ = 0.0f;
  float last_vel_error_ = 0.0f;
  bool vel_pid_has_last_error_ = false;
};

class MirrorMount {
 public:
  MirrorMount();

  void begin();
  void home();  // both axes
  void homeAzimuth();
  void homeElevation();
  void stop();
  /** Returns false if either axis is not yet homeed (command ignored). */
  bool setTarget(float azimuth_deg, float elevation_deg);
  /** Returns false if either axis is not yet homeed (command ignored). */
  bool setVelocity(float azimuth_deg_s, float elevation_deg_s);
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
  const char* azimuthHomeState() const { return azimuth_.homeStateText(); }
  const char* elevationHomeState() const { return elevation_.homeStateText(); }
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
