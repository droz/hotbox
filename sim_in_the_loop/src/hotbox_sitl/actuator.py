from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ActuatorState:
    angle_deg: float = 0.0
    velocity_deg_s: float = 0.0
    encoder_ticks: int = 0
    hall_triggered: bool = False


@dataclass(slots=True)
class ActuatorModel:
    """First-order brushed-DC axis model for SITL.

    All tuneable constants default to the values in ``config/system.yaml``
    (``actuator`` section) but can be overridden for unit tests.  Pass
    ``ActuatorModel.from_constants(system.actuator)`` to build from the live
    config rather than hardcoding anything here.
    """

    ticks_per_degree: float = 35.56
    max_velocity_deg_s: float = 30.0
    # First-order lag time constant for velocity response [s].
    # Approximates motor electrical + mechanical inertia.
    velocity_time_constant_s: float = 0.2
    stall_velocity_threshold_deg_s: float = 0.05
    stall_timeout_s: float = 1.0
    hall_angle_deg: float = 0.0
    last_pwm: float = 0.0
    stall_timer_s: float = 0.0
    state: ActuatorState = field(default_factory=ActuatorState)

    @classmethod
    def from_constants(cls, ac: "hotbox_shared.ActuatorConstants", hall_angle_deg: float = 0.0) -> "ActuatorModel":  # type: ignore[name-defined]
        return cls(
            ticks_per_degree=ac.ticks_per_degree,
            max_velocity_deg_s=ac.max_velocity_deg_s,
            velocity_time_constant_s=ac.velocity_time_constant_s,
            stall_velocity_threshold_deg_s=ac.stall_velocity_threshold_deg_s,
            stall_timeout_s=ac.stall_timeout_s,
            hall_angle_deg=hall_angle_deg,
        )

    def step(self, pwm_command: float, dt_s: float) -> ActuatorState:
        pwm = max(-1.0, min(1.0, pwm_command))
        self.last_pwm = pwm
        commanded_velocity = pwm * self.max_velocity_deg_s
        # First-order lag: α = dt / τ  (clamped to 1 so we never overshoot)
        alpha = min(1.0, dt_s / self.velocity_time_constant_s)
        self.state.velocity_deg_s += (commanded_velocity - self.state.velocity_deg_s) * alpha
        self.state.angle_deg += self.state.velocity_deg_s * dt_s
        self.state.encoder_ticks = int(round(self.state.angle_deg * self.ticks_per_degree))
        self.state.hall_triggered = abs(self.state.angle_deg - self.hall_angle_deg) <= 1.0

        # Stall detection (mirrors firmware logic)
        if abs(pwm) > 0.8 and abs(self.state.velocity_deg_s) < self.stall_velocity_threshold_deg_s:
            self.stall_timer_s += dt_s
        else:
            self.stall_timer_s = 0.0

        return self.state
