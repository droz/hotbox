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
    phase: str = "search"  # leave | search | retract | creep | across | settle
    mark_deg: float = 0.0
    edge1_deg: float = 0.0
    edge2_deg: float = 0.0
    mid_deg: float = 0.0


@dataclass(slots=True)
class SimulatedMirrorNode:
    """Python plant+controller stand-in that mirrors the slim firmware API.

    Wire commands: home, stop, set_target, get_status, clear_error.
    Status modes: idle | homing | position | fault.
    Home states per axis: unhomed | homing | homed | fault.
    Homing: search → retract → creep both edges → settle to midpoint.
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
    homing_search_velocity_deg_s: float = 5.0
    homing_creep_velocity_deg_s: float = 0.5
    homing_backoff_deg: float = 1.0
    homing_settle_tol_deg: float = 0.05
    _home_az: _AxisHomeState = field(default_factory=_AxisHomeState)
    _home_el: _AxisHomeState = field(default_factory=_AxisHomeState)

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
        # Hall/home pose matches firmware: az 0° (N), el 90° (face-up / zenith).
        azimuth_axis = ActuatorModel.from_constants(ac, hall_angle_deg=0.0)
        altitude_axis = ActuatorModel.from_constants(ac, hall_angle_deg=90.0)
        altitude_axis.state.angle_deg = 90.0
        return cls(
            node_id=node_id,
            azimuth_axis=azimuth_axis,
            altitude_axis=altitude_axis,
            oven_facing_azimuth_deg=float(oven_facing_azimuth_deg),
            target_elevation_deg=90.0,
            homing_search_velocity_deg_s=float(ac.homing_search_velocity_deg_s),
            homing_creep_velocity_deg_s=float(ac.homing_creep_velocity_deg_s),
            homing_backoff_deg=float(ac.homing_backoff_deg),
            homing_settle_tol_deg=float(ac.homing_settle_tol_deg),
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
            if self.mode == "homing":
                return
            self.target_azimuth_deg = float(command.payload.get("azimuth_deg", self.target_azimuth_deg))
            self.target_elevation_deg = float(command.payload.get("elevation_deg", self.target_elevation_deg))
            self.mode = "position"
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
        # Match firmware: seek toward more-negative encoder angle (hall placed ahead).
        if axis in {"az", "both"}:
            self.azimuth_home = "homing"
            self.azimuth_axis.state.angle_deg = self.azimuth_axis.hall_angle_deg + 5.0
            self.azimuth_axis.state.hall_triggered = False
            self._home_az = _AxisHomeState(active=True, phase="search")
        if axis in {"el", "both"}:
            self.elevation_home = "homing"
            self.altitude_axis.state.angle_deg = self.altitude_axis.hall_angle_deg + 5.0
            self.altitude_axis.state.hall_triggered = False
            self._home_el = _AxisHomeState(active=True, phase="search")

    def step(self, dt_s: float) -> None:
        if self.mode == "homing":
            self._step_homing(dt_s)
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

    def _pwm_for_velocity(self, axis: ActuatorModel, velocity_deg_s: float) -> float:
        max_v = max(axis.max_velocity_deg_s, 1e-6)
        return max(-1.0, min(1.0, velocity_deg_s / max_v))

    def _step_homing(self, dt_s: float) -> None:
        if self._home_az.active:
            done = self._advance_home(self.azimuth_axis, self._home_az, dt_s)
            if done:
                self.azimuth_home = "homed"
                self._home_az = _AxisHomeState()
        if self._home_el.active:
            done = self._advance_home(self.altitude_axis, self._home_el, dt_s)
            if done:
                self.elevation_home = "homed"
                self._home_el = _AxisHomeState()
        if not self._home_az.active and not self._home_el.active:
            self.mode = "idle"

    def _advance_home(self, axis: ActuatorModel, home: _AxisHomeState, dt_s: float) -> bool:
        """Run one tick of multi-phase home. Returns True when homed at midpoint."""
        search_pwm = self._pwm_for_velocity(axis, self.homing_search_velocity_deg_s)
        creep_pwm = self._pwm_for_velocity(axis, self.homing_creep_velocity_deg_s)

        if home.phase == "leave":
            if not axis.state.hall_triggered:
                home.phase = "search"
            else:
                axis.step(search_pwm, dt_s)
            return False

        if home.phase == "search":
            if axis.state.hall_triggered:
                home.mark_deg = axis.state.angle_deg
                home.phase = "retract"
            else:
                axis.step(-search_pwm, dt_s)
            return False

        if home.phase == "retract":
            backed = axis.state.angle_deg - home.mark_deg
            if backed >= self.homing_backoff_deg and not axis.state.hall_triggered:
                home.phase = "creep"
            else:
                axis.step(search_pwm, dt_s)
            return False

        if home.phase == "creep":
            if axis.state.hall_triggered:
                home.edge1_deg = axis.state.angle_deg
                home.phase = "across"
            else:
                axis.step(-creep_pwm, dt_s)
            return False

        if home.phase == "across":
            if not axis.state.hall_triggered:
                home.edge2_deg = axis.state.angle_deg
                home.mid_deg = 0.5 * (home.edge1_deg + home.edge2_deg)
                home.phase = "settle"
            else:
                axis.step(creep_pwm, dt_s)
            return False

        # settle to midpoint, then zero encoder at hall home pose
        error = home.mid_deg - axis.state.angle_deg
        if abs(error) <= self.homing_settle_tol_deg:
            axis.state.angle_deg = axis.hall_angle_deg
            axis.state.velocity_deg_s = 0.0
            return True
        axis.step(max(-1.0, min(1.0, error / 10.0)), dt_s)
        return False

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
