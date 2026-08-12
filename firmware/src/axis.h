#pragma once

#include "config.h"

namespace hotbox {

// Minimal axis states. Product modes (track/park/jog) live on the host;
// firmware servos position, velocity, homes, idles, or faults.
enum class AxisMode { Idle, Homing, Position, Velocity, Fault };

/** Homing sub-states while AxisMode::Homing. */
enum class HomingPhase {
  LeaveSwitch,  // started on hall — reverse until clear + clear-distance, then Seek
  Seek,         // constant-speed (positive encoder) until rising hall edge
  Across,       // continue until falling hall edge, then zero + go home
};

class BrushedAxis {
 public:
  BrushedAxis(int motor_p, int motor_m, int enc_a, int enc_b, int hall_pin);

  void begin();
  void startHoming();
  /** Update position setpoint only — does not engage the PID (see startPosition). */
  void setTargetDeg(float target_deg);
  /** Engage position PID toward the current setpoint (no-op if already in Position). */
  void startPosition();
  void setVelocityDegS(float velocity_deg_s);
  /** Disengage PID / coast. Does not modify the position setpoint. */
  void stop();
  void clearFault();
  void update(float dt_s);

  void setPidGains(float kp, float ki, float kd);
  void setVelocityPidGains(float kp, float ki, float kd);
  void resetPidState();

  float positionDeg() const { return position_deg_; }
  float targetDeg() const { return target_deg_; }
  float velocityDegS() const { return velocity_deg_s_; }
  /** Integral term contribution to duty command (position loop), before clamp. */
  float integralTerm() const { return ki_ * integral_; }
  bool isHomed() const { return homed_; }
  /** unhomed | homing | homed | fault */
  const char* homeStateText() const;
  bool hallTriggered() const;
  AxisMode mode() const { return mode_; }
  const char* faultText() const { return fault_text_; }
  /** Hall-window width from last successful home [°]; <0 if unknown. */
  float hallWidthDeg() const { return hall_width_deg_; }
  bool hasHallWidth() const { return hall_width_deg_ >= 0.0f; }

  /** GPIO CHANGE ISR trampoline target — latch encoder ticks on hall edge. */
  void onHallEdgeIsr();

 private:
  void driveMotor(float command);
  void setFault(const char* text);
  void finishHoming(float mid_deg);
  void enterHomingPhase(HomingPhase phase);
  /** Home joint angle for this axis (az = oven-facing, el = 90°). */
  float homeAngleDeg() const;
  /** Position PID → duty ∈ [-1,1]. When ``apply_position_deadband``, coast+freeze I near zero error. */
  float computePositionPidDuty(float error_deg, float dt_s, bool apply_position_deadband);
  /** Velocity PID → duty ∈ [-1,1] using ``kp_vel_`` / ``ki_vel_`` / ``kd_vel_``. */
  float computeVelocityPidDuty(float target_velocity_deg_s, float dt_s);
  /**
   * Zero any velocity that would drive further past joint limits (elevation box
   * or azimuth relative to oven-facing). Homing does not use this.
   */
  float limitAwareVelocityCommand(float commanded_deg_s) const;

  void clearHallEdgeLatches();
  /** Consume ISR assert latch (magnet entry). Returns false if none pending. */
  bool takeHallAssertEdge(long* ticks_out);
  /** Consume ISR clear latch (magnet exit). Returns false if none pending. */
  bool takeHallClearEdge(long* ticks_out);
  long encoderCountNow() const;

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
  /** LeaveSwitch: true once hall has cleared; clear mark is that pose. */
  bool homing_leave_cleared_ = false;
  float homing_leave_clear_deg_ = 0.0f;
  bool homed_ = false;
  /** Last measured magnet window width [°]; -1 = not measured yet. */
  float hall_width_deg_ = -1.0f;
  // Stall detect only after the encoder has moved at least once — otherwise a
  // missing encoder would immediately fault any open-loop PWM bring-up.
  bool encoder_alive_ = false;
  HomingPhase homing_phase_ = HomingPhase::Seek;
  AxisMode mode_ = AxisMode::Idle;
  const char* fault_text_ = nullptr;

  // Hall GPIO ISR latches (encoder ticks at edge). Written in ISR, read in update().
  volatile bool hall_assert_pending_ = false;
  volatile bool hall_clear_pending_ = false;
  volatile long hall_assert_ticks_ = 0;
  volatile long hall_clear_ticks_ = 0;

  // Position-loop gains (from config / set_pid_pos).
  float kp_ = kPidKp;
  float ki_ = kPidKi;
  float kd_ = kPidKd;
  float integral_ = 0.0f;
  float last_error_deg_ = 0.0f;
  bool pid_has_last_error_ = false;

  // Velocity-loop gains (from config / set_pid_vel).
  float kp_vel_ = kPidVelocityKp;
  float ki_vel_ = kPidVelocityKi;
  float kd_vel_ = kPidVelocityKd;
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
  /**
   * Update position setpoints only (does not start the PID).
   * When ``hold_current`` is true, ignore az/el and capture the live pose.
   * Returns false if either axis is not yet homed (command ignored).
   */
  bool setTarget(float azimuth_deg, float elevation_deg, bool hold_current = false);
  /** Engage position PID on both axes. Returns false if not yet homed. */
  bool start();
  /** Returns false if either axis is not yet homeed (command ignored). */
  bool setVelocity(float azimuth_deg_s, float elevation_deg_s);
  void clearError();
  /** Soft reset: stop, clear faults/PID state. On device also reboots via ESP.restart(). */
  void reset();
  void setPid(float kp, float ki, float kd);
  void setVelocityPid(float kp, float ki, float kd);
  void update(float dt_s);

  float azimuthDeg() const { return azimuth_.positionDeg(); }
  float elevationDeg() const { return elevation_.positionDeg(); }
  float targetAzimuthDeg() const { return azimuth_.targetDeg(); }
  float targetElevationDeg() const { return elevation_.targetDeg(); }
  float azimuthIntegralTerm() const { return azimuth_.integralTerm(); }
  float elevationIntegralTerm() const { return elevation_.integralTerm(); }
  float pidKp() const { return pid_kp_; }
  float pidKi() const { return pid_ki_; }
  float pidKd() const { return pid_kd_; }
  float pidVelocityKp() const { return pid_velocity_kp_; }
  float pidVelocityKi() const { return pid_velocity_ki_; }
  float pidVelocityKd() const { return pid_velocity_kd_; }
  bool isHomed() const { return azimuth_.isHomed() && elevation_.isHomed(); }
  const char* azimuthHomeState() const { return azimuth_.homeStateText(); }
  const char* elevationHomeState() const { return elevation_.homeStateText(); }
  bool azimuthHallTriggered() const { return azimuth_.hallTriggered(); }
  bool elevationHallTriggered() const { return elevation_.hallTriggered(); }
  bool azimuthHasHallWidth() const { return azimuth_.hasHallWidth(); }
  bool elevationHasHallWidth() const { return elevation_.hasHallWidth(); }
  float azimuthHallWidthDeg() const { return azimuth_.hallWidthDeg(); }
  float elevationHallWidthDeg() const { return elevation_.hallWidthDeg(); }
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
  float pid_velocity_kp_ = kPidVelocityKp;
  float pid_velocity_ki_ = kPidVelocityKi;
  float pid_velocity_kd_ = kPidVelocityKd;
  const char* mode_text_ = "idle";
};

}  // namespace hotbox
