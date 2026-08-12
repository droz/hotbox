from __future__ import annotations

from dataclasses import dataclass, field

from hotbox_controller.protocol import CommandName, MirrorCommand, MirrorStatus

from .actuator import ActuatorModel


def _parse_home_axis(payload: dict) -> str:
    axis = str(payload.get("axis", "both")).strip().lower()
    if axis in {"az", "azimuth"}:
        return "az"
    if axis in {"el", "elevation"}:
        return "el"
    return "both"


@dataclass(slots=True)
class _AxisHomeState:
    active: bool = False
    phase: str = "seek"  # leave | seek | across
    edge1_deg: float = 0.0
    edge2_deg: float = 0.0
    leave_cleared: bool = False
    leave_clear_deg: float = 0.0


@dataclass(slots=True)
class SimulatedMirrorNode:
    """Python plant+controller stand-in that mirrors the slim firmware API.

    Wire commands: home, stop, set_target, get_status, clear_error.
    Status modes: idle | homing | position | fault.
    Home states per axis: unhomed | homing | homed | fault.
    Homing: optional leave (+ clear distance) → constant-speed seek both hall
    edges → remap zero → drive to home.
    """

    node_id: int
    azimuth_axis: ActuatorModel = field(default_factory=ActuatorModel)
    altitude_axis: ActuatorModel = field(default_factory=ActuatorModel)
    azimuth_home: str = "unhomed"
    elevation_home: str = "unhomed"
    mode: str = "idle"
    fault: str | None = None
    target_azimuth_deg: float = 0.0
    target_elevation_deg: float = 0.0
    oven_facing_azimuth_deg: float = 0.0
    homing_velocity_deg_s: float = 2.0
    homing_clear_distance_deg: float = 5.0
    _home_az: _AxisHomeState = field(default_factory=_AxisHomeState)
    _home_el: _AxisHomeState = field(default_factory=_AxisHomeState)
    _vel_az: float = 0.0
    _vel_el: float = 0.0

    @property
    def homed(self) -> bool:
        return self.azimuth_home == "homed" and self.elevation_home == "homed"

    @classmethod
    def from_constants(
        cls,
        node_id: int,
        ac: "hotbox_shared.ActuatorConstants",  # type: ignore[name-defined]
        *,
        oven_facing_azimuth_deg: float = 0.0,
    ) -> "SimulatedMirrorNode":
        # Hall/home pose matches firmware: az = oven-facing (travel center),
        # el 90° (face-up / zenith).
        of = float(oven_facing_azimuth_deg)
        azimuth_axis = ActuatorModel.from_constants(ac, hall_angle_deg=of)
        azimuth_axis.state.angle_deg = of
        altitude_axis = ActuatorModel.from_constants(ac, hall_angle_deg=90.0)
        altitude_axis.state.angle_deg = 90.0
        return cls(
            node_id=node_id,
            azimuth_axis=azimuth_axis,
            altitude_axis=altitude_axis,
            oven_facing_azimuth_deg=of,
            target_azimuth_deg=of,
            target_elevation_deg=90.0,
            homing_velocity_deg_s=float(ac.homing_velocity_deg_s),
            homing_clear_distance_deg=float(ac.homing_clear_distance_deg),
        )

    def handle_command(self, command: MirrorCommand) -> None:
        if command.node_id != self.node_id:
            return
        if command.command == CommandName.HOME:
            self._start_homing(_parse_home_axis(command.payload))
        elif command.command == CommandName.STOP:
            self._home_az = _AxisHomeState()
            self._home_el = _AxisHomeState()
            if self.azimuth_home == "homing":
                self.azimuth_home = "unhomed"
            if self.elevation_home == "homing":
                self.elevation_home = "unhomed"
            self.mode = "idle"
        elif command.command == CommandName.SET_TARGET:
            if self.mode == "homing" or not self.homed:
                return
            self.target_azimuth_deg = float(command.payload.get("azimuth_deg", self.target_azimuth_deg))
            self.target_elevation_deg = float(command.payload.get("elevation_deg", self.target_elevation_deg))
            self.mode = "position"
            self.fault = None
            self._vel_az = 0.0
            self._vel_el = 0.0
        elif command.command == CommandName.SET_VELOCITY:
            if self.mode == "homing" or not self.homed:
                return
            self._vel_az = float(command.payload.get("azimuth_deg_s", 0.0))
            self._vel_el = float(command.payload.get("elevation_deg_s", 0.0))
            self.mode = "velocity"
            self.fault = None
        elif command.command == CommandName.CLEAR_ERROR:
            self.fault = None
            if self.azimuth_home == "fault":
                self.azimuth_home = "unhomed"
            if self.elevation_home == "fault":
                self.elevation_home = "unhomed"
            self.mode = "idle"
        elif command.command == CommandName.GET_STATUS:
            return

    def status(self) -> MirrorStatus:
        return MirrorStatus(
            node_id=self.node_id,
            azimuth_home=self.azimuth_home,
            elevation_home=self.elevation_home,
            fault=self.fault,
            azimuth_deg=self.azimuth_axis.state.angle_deg,
            elevation_deg=self.altitude_axis.state.angle_deg,
            mode=self.mode,
        )

    def _start_homing(self, axis: str) -> None:
        self.mode = "homing"
        self.fault = None
        if axis in {"az", "both"}:
            self.azimuth_home = "homing"
            self.azimuth_axis.step(0.0, 0.0)  # refresh hall from angle
            phase = "leave" if self.azimuth_axis.state.hall_triggered else "seek"
            self._home_az = _AxisHomeState(active=True, phase=phase)
        if axis in {"el", "both"}:
            self.elevation_home = "homing"
            self.altitude_axis.step(0.0, 0.0)
            phase = "leave" if self.altitude_axis.state.hall_triggered else "seek"
            self._home_el = _AxisHomeState(active=True, phase=phase)

    def step(self, dt_s: float) -> None:
        if self.mode == "homing":
            self._step_homing(dt_s)
            return

        if self.mode == "velocity":
            v_az = self._limit_aware_velocity(
                self.azimuth_axis.state.angle_deg, self._vel_az, azimuth=True
            )
            v_el = self._limit_aware_velocity(
                self.altitude_axis.state.angle_deg, self._vel_el, azimuth=False
            )
            pwm_az = self._pwm_for_velocity(self.azimuth_axis, v_az)
            pwm_el = self._pwm_for_velocity(self.altitude_axis, v_el)
            self.azimuth_axis.step(pwm_az, dt_s)
            self.altitude_axis.step(pwm_el, dt_s)
            return

        if self.mode == "position":
            pwm_az = self._azimuth_pwm(self.target_azimuth_deg)
            pwm_el = self._position_pwm(self.altitude_axis, self.target_elevation_deg)
            self.azimuth_axis.step(pwm_az, dt_s)
            self.altitude_axis.step(pwm_el, dt_s)
            if self._axis_stalled(self.azimuth_axis) or self._axis_stalled(self.altitude_axis):
                self.fault = "stalled"
                self.mode = "fault"
                if self._axis_stalled(self.azimuth_axis):
                    self.azimuth_home = "fault"
                if self._axis_stalled(self.altitude_axis):
                    self.elevation_home = "fault"

    def _limit_aware_velocity(self, position_deg: float, commanded_deg_s: float, *, azimuth: bool) -> float:
        """Match firmware: zero outward velocity at joint limits (homing is exempt)."""
        from hotbox_shared import relative_azimuth_deg

        if azimuth:
            rel = relative_azimuth_deg(position_deg, self.oven_facing_azimuth_deg)
            if rel >= 150.0 and commanded_deg_s > 0.0:
                return 0.0
            if rel <= -150.0 and commanded_deg_s < 0.0:
                return 0.0
            return commanded_deg_s
        if position_deg >= 90.0 and commanded_deg_s > 0.0:
            return 0.0
        if position_deg <= 0.0 and commanded_deg_s < 0.0:
            return 0.0
        return commanded_deg_s

    def _pwm_for_velocity(self, axis: ActuatorModel, velocity_deg_s: float) -> float:
        max_v = max(axis.max_velocity_deg_s, 1e-6)
        return max(-1.0, min(1.0, velocity_deg_s / max_v))

    def _step_homing(self, dt_s: float) -> None:
        if self._home_az.active:
            done = self._advance_home(self.azimuth_axis, self._home_az, dt_s)
            if done:
                self.azimuth_home = "homed"
                self.target_azimuth_deg = self.azimuth_axis.hall_angle_deg
                self._home_az = _AxisHomeState()
        elif self.azimuth_home == "homed":
            # Keep driving home while the other axis finishes seeking.
            self.azimuth_axis.step(self._azimuth_pwm(self.target_azimuth_deg), dt_s)

        if self._home_el.active:
            done = self._advance_home(self.altitude_axis, self._home_el, dt_s)
            if done:
                self.elevation_home = "homed"
                self.target_elevation_deg = self.altitude_axis.hall_angle_deg
                self._home_el = _AxisHomeState()
        elif self.elevation_home == "homed":
            self.altitude_axis.step(self._position_pwm(self.altitude_axis, self.target_elevation_deg), dt_s)

        if not self._home_az.active and not self._home_el.active:
            # Match firmware: after remap, servo home (position mode).
            self.mode = "position"

    def _advance_home(self, axis: ActuatorModel, home: _AxisHomeState, dt_s: float) -> bool:
        """Run one tick of single-speed home. Returns True once zero is remapped."""
        pwm = self._pwm_for_velocity(axis, self.homing_velocity_deg_s)

        if home.phase == "leave":
            if not axis.state.hall_triggered:
                if not home.leave_cleared:
                    home.leave_cleared = True
                    home.leave_clear_deg = axis.state.angle_deg
                if abs(axis.state.angle_deg - home.leave_clear_deg) >= self.homing_clear_distance_deg:
                    home.phase = "seek"
                    return False
            # Opposite of Seek so we clear the magnet before driving back through it.
            axis.step(-pwm, dt_s)
            return False

        if home.phase == "seek":
            if axis.state.hall_triggered:
                home.edge1_deg = axis.state.angle_deg
                home.phase = "across"
            else:
                axis.step(pwm, dt_s)
            return False

        # across: keep going positive until the falling edge, then redefine zero.
        if axis.state.hall_triggered:
            axis.step(pwm, dt_s)
            return False

        home.edge2_deg = axis.state.angle_deg
        mid = 0.5 * (home.edge1_deg + home.edge2_deg)
        home_deg = axis.hall_angle_deg
        # Remap so mid → home_deg without a physical jump; then drive home.
        axis.state.angle_deg = home_deg + (axis.state.angle_deg - mid)
        axis.state.velocity_deg_s = 0.0
        return True

    def _azimuth_pwm(self, target_deg: float) -> float:
        from hotbox_shared import limited_azimuth_error_deg

        error = limited_azimuth_error_deg(
            target_deg,
            self.azimuth_axis.state.angle_deg,
            oven_facing_azimuth_deg=self.oven_facing_azimuth_deg,
        )
        return max(-1.0, min(1.0, error / 10.0))

    @staticmethod
    def _position_pwm(axis: ActuatorModel, target_deg: float) -> float:
        error = target_deg - axis.state.angle_deg
        # Proportional position controller; gain chosen so 10° error → full PWM.
        return max(-1.0, min(1.0, error / 10.0))

    @staticmethod
    def _axis_stalled(axis: ActuatorModel) -> bool:
        return (
            abs(axis.state.velocity_deg_s) < axis.stall_velocity_threshold_deg_s
            and abs(axis.last_pwm) > 0.8
            and axis.stall_timer_s > axis.stall_timeout_s
        )
