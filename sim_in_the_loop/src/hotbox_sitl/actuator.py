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
    ticks_per_degree: float = 8.0
    max_velocity_deg_s: float = 30.0
    damping: float = 0.2
    hall_angle_deg: float = 0.0
    last_pwm: float = 0.0
    stall_timer_s: float = 0.0
    state: ActuatorState = field(default_factory=ActuatorState)

    def step(self, pwm_command: float, dt_s: float) -> ActuatorState:
        pwm = max(-1.0, min(1.0, pwm_command))
        self.last_pwm = pwm
        if abs(pwm) > 0.8 and abs(self.state.velocity_deg_s) < 0.05:
            self.stall_timer_s += dt_s
        else:
            self.stall_timer_s = 0.0
        commanded_velocity = pwm * self.max_velocity_deg_s
        self.state.velocity_deg_s += (commanded_velocity - self.state.velocity_deg_s) * min(1.0, dt_s * (1.0 + self.damping))
        self.state.angle_deg += self.state.velocity_deg_s * dt_s
        self.state.encoder_ticks = int(round(self.state.angle_deg * self.ticks_per_degree))
        self.state.hall_triggered = abs(self.state.angle_deg - self.hall_angle_deg) <= 1.0
        return self.state
