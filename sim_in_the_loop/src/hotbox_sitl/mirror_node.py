from __future__ import annotations

from dataclasses import dataclass, field

from hotbox_controller.protocol import CommandName, MirrorCommand, MirrorStatus

from .actuator import ActuatorModel


@dataclass(slots=True)
class SimulatedMirrorNode:
    node_id: int
    azimuth_axis: ActuatorModel = field(default_factory=ActuatorModel)
    altitude_axis: ActuatorModel = field(default_factory=ActuatorModel)
    homed: bool = False
    mode: str = "idle"
    fault: str | None = None
    target_azimuth_deg: float = 0.0
    target_elevation_deg: float = 0.0
    jog_az_rate_deg_s: float = 0.0
    jog_el_rate_deg_s: float = 0.0

    def handle_command(self, command: MirrorCommand) -> None:
        if command.node_id != self.node_id:
            return
        if command.command == CommandName.HOME:
            self._start_homing()
        elif command.command == CommandName.STOP:
            self.mode = "idle"
            self.jog_az_rate_deg_s = 0.0
            self.jog_el_rate_deg_s = 0.0
        elif command.command == CommandName.SET_TARGET:
            if self.mode == "homing":
                return
            self.target_azimuth_deg = float(command.payload.get("azimuth_deg", self.target_azimuth_deg))
            self.target_elevation_deg = float(command.payload.get("elevation_deg", self.target_elevation_deg))
            self.mode = str(command.payload.get("mode", "tracking"))
            self.fault = None
        elif command.command == CommandName.JOG:
            self.jog_az_rate_deg_s = float(command.payload.get("azimuth_rate_deg_s", 0.0))
            self.jog_el_rate_deg_s = float(command.payload.get("elevation_rate_deg_s", 0.0))
            self.mode = "jog"
            self.fault = None
        elif command.command == CommandName.CLEAR_ERROR:
            self.fault = None
            self.mode = "idle"
        elif command.command == CommandName.GET_STATUS:
            return

    def status(self) -> MirrorStatus:
        return MirrorStatus(
            node_id=self.node_id,
            homed=self.homed,
            fault=self.fault,
            azimuth_deg=self.azimuth_axis.state.angle_deg,
            elevation_deg=self.altitude_axis.state.angle_deg,
            mode=self.mode,
        )

    def _start_homing(self) -> None:
        self.mode = "homing"
        self.homed = False
        self.azimuth_axis.state.angle_deg = self.azimuth_axis.hall_angle_deg - 5.0
        self.altitude_axis.state.angle_deg = self.altitude_axis.hall_angle_deg - 5.0

    def step(self, dt_s: float) -> None:
        if self.mode == "homing":
            self._step_homing(dt_s)
            return

        if self.mode == "jog":
            pwm_az = max(-1.0, min(1.0, self.jog_az_rate_deg_s / self.azimuth_axis.max_velocity_deg_s))
            pwm_el = max(-1.0, min(1.0, self.jog_el_rate_deg_s / self.altitude_axis.max_velocity_deg_s))
            self.azimuth_axis.step(pwm_az, dt_s)
            self.altitude_axis.step(pwm_el, dt_s)
            return

        if self.mode in {"tracking", "parked"}:
            pwm_az = self._position_pwm(self.azimuth_axis, self.target_azimuth_deg)
            pwm_el = self._position_pwm(self.altitude_axis, self.target_elevation_deg)
            self.azimuth_axis.step(pwm_az, dt_s)
            self.altitude_axis.step(pwm_el, dt_s)
            if self._axis_stalled(self.azimuth_axis) or self._axis_stalled(self.altitude_axis):
                self.fault = "stalled"
                self.mode = "fault"

    def _step_homing(self, dt_s: float) -> None:
        for axis in (self.azimuth_axis, self.altitude_axis):
            if not axis.state.hall_triggered:
                axis.step(0.15, dt_s)
            else:
                axis.state.angle_deg = axis.hall_angle_deg
                axis.state.velocity_deg_s = 0.0
        if self.azimuth_axis.state.hall_triggered and self.altitude_axis.state.hall_triggered:
            self.homed = True
            self.mode = "idle"

    @staticmethod
    def _position_pwm(axis: ActuatorModel, target_deg: float) -> float:
        error = target_deg - axis.state.angle_deg
        command = max(-1.0, min(1.0, error / 10.0))
        return command

    @staticmethod
    def _axis_stalled(axis: ActuatorModel) -> bool:
        return abs(axis.state.velocity_deg_s) < 0.01 and abs(axis.last_pwm) > 0.8 and axis.stall_timer_s > 1.0
