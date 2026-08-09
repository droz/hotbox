from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(slots=True, frozen=True)
class SiteConstants:
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    # IANA timezone for civil-day work (plots, sunrise/sunset). Never use the host TZ.
    timezone_id: str


@dataclass(slots=True, frozen=True)
class AbsorberConstants:
    width_m: float
    height_m: float
    center_height_m: float
    normal_angle_from_x_deg: float

    @property
    def center_world(self) -> np.ndarray:
        # O is the world origin in XY; Z is the absorber center height.
        return np.array([0.0, 0.0, self.center_height_m], dtype=float)


@dataclass(slots=True, frozen=True)
class MirrorConstants:
    grid_nx: int
    grid_ny: int
    tile_side_m: float
    pitch_m: float
    mount_offset_d_m: float
    radius_of_curvature_m: float
    default_oa_distance_m: float
    default_mount_height_m: float

    @property
    def facet_count(self) -> int:
        return int(self.grid_nx * self.grid_ny)


@dataclass(slots=True, frozen=True)
class MountDesign:
    """One mirror assembly placement relative to the absorber / oven back.

    ``bearing_deg`` is a horizontal angle relative to the absorber normal
    (``AbsorberConstants.normal_angle_from_x_deg``): ``0`` is straight out along
    the oven back normal; positive is CCW about world +Z (up). Typical fleet:
    ``-30``, ``0``, ``+30``.
    """

    node_id: int
    bearing_deg: float
    oa_distance_m: float
    mount_height_m: float

    def mount_world(self, *, normal_angle_from_x_deg: float) -> np.ndarray:
        """World ENU position of mount pivot An from oven orientation + relative bearing."""
        ang = math.radians(float(normal_angle_from_x_deg) + float(self.bearing_deg))
        return np.array(
            [
                self.oa_distance_m * math.cos(ang),
                self.oa_distance_m * math.sin(ang),
                self.mount_height_m,
            ],
            dtype=float,
        )

    def oa_bearing_from_north_deg(self, *, normal_angle_from_x_deg: float) -> float:
        """Absolute OA bearing from north toward east (calibration / legacy storage)."""
        mount = self.mount_world(normal_angle_from_x_deg=normal_angle_from_x_deg)
        return float(math.degrees(math.atan2(mount[0], mount[1])) % 360.0)


@dataclass(slots=True, frozen=True)
class FleetConstants:
    assembly_count: int
    assembly_spacing_m: float
    mounts: tuple[MountDesign, ...]

    def mount_by_id(self, node_id: int) -> MountDesign:
        for mount in self.mounts:
            if mount.node_id == node_id:
                return mount
        raise KeyError(f"no mount design for node_id={node_id}")


@dataclass(slots=True, frozen=True)
class ControlConstants:
    safe_park_azimuth_deg: float
    safe_park_elevation_deg: float
    # When the oven is not requesting heat, Track mirrors aim this far above the absorber
    # (world +Z) rather than at the absorber center.
    idle_aim_height_above_absorber_m: float = 2.0
    # When true, aiming solves for mount_offset so the center facet reflects onto the
    # absorber (least squares after the bisector seed). Set false to skip (testing).
    solve_for_mount_offset: bool = True
    # Physical joint limits. Azimuth is relative to oven-facing (0 = aim at absorber at high el).
    elevation_min_deg: float = 0.0
    elevation_max_deg: float = 90.0
    azimuth_min_deg: float = -150.0
    azimuth_max_deg: float = 150.0

    def mount_joint_limits(self) -> "MountJointLimits":
        from .mount import MountJointLimits

        return MountJointLimits(
            elevation_min_deg=self.elevation_min_deg,
            elevation_max_deg=self.elevation_max_deg,
            azimuth_min_deg=self.azimuth_min_deg,
            azimuth_max_deg=self.azimuth_max_deg,
        )


@dataclass(slots=True, frozen=True)
class ActuatorConstants:
    """Mechanical and control constants for one brushed-DC alt-az axis pair.

    These values are shared across firmware, SITL, and (via the generated header)
    the on-board microcontroller.  They describe the *output* shaft (mount axis)
    unless noted otherwise.

    Encoder note
    ------------
    The encoder sits on the motor shaft *before* the gearbox.  The relationship is:

        ticks_per_degree = (encoder_ppr * 4) * gear_ratio / 360

    All three values are stored so the gear ratio is explicit and auditable.
    """

    # --- Encoder / gearbox ---
    encoder_ppr: int = 64
    """Encoder pulses per revolution of the *motor* shaft (single-channel edges)."""
    gear_ratio: float = 50.0
    """Output shaft turns per motor shaft turn (> 1 means speed reduction)."""

    @property
    def ticks_per_degree(self) -> float:
        """Quadrature encoder ticks per degree of output (mount) shaft rotation.

        Derived from encoder_ppr × 4 (quadrature edges) × gear_ratio / 360.
        """
        return self.encoder_ppr * 4.0 * self.gear_ratio / 360.0

    # --- Velocity limits ---
    max_velocity_deg_s: float = 30.0
    """Maximum commanded output-shaft angular velocity [°/s]."""
    max_accel_deg_s2: float = 120.0
    """Maximum commanded output-shaft angular acceleration [°/s²] (firmware ramp limiter)."""

    # --- Homing ---
    homing_velocity_deg_s: float = 5.0
    """Slow creep speed used during hall-sensor homing [°/s]."""

    # --- PID (position loop, output shaft degrees) ---
    pid_kp: float = 1.2
    """Proportional gain (applied in encoder-tick space, output ±255 PWM units)."""
    pid_ki: float = 0.05
    """Integral gain."""
    pid_kd: float = 0.01
    """Derivative gain."""

    # --- SITL physics model ---
    velocity_time_constant_s: float = 0.2
    """First-order lag time constant for motor velocity response in the SITL model [s].
    Approximates motor electrical + mechanical inertia without a full torque model."""

    # --- Stall detection ---
    stall_velocity_threshold_deg_s: float = 0.05
    """Mirror is considered stalled when |velocity| < this while |command| is large [°/s]."""
    stall_timeout_s: float = 1.0
    """Time after which a stalled axis triggers a fault [s]."""

    # --- Control loop ---
    control_period_s: float = 0.02
    """Firmware control-loop period [s] (50 Hz)."""


_ARDUINO_PIN_NAMES = frozenset(
    {*(f"D{i}" for i in range(14)), *(f"A{i}" for i in range(8))}
)


@dataclass(slots=True, frozen=True)
class PinConstants:
    """Arduino Nano ESP32 pin roles baked into firmware via the generated header.

    Values are Arduino pin labels (``D5``, ``A0``, …), not raw ESP32 GPIO numbers.
    Swap ``*_enc_a`` / ``*_enc_b`` on an axis to invert quadrature direction.
    """

    can_tx: str = "D10"
    can_rx: str = "D9"
    elevation_motor_p: str = "A0"
    elevation_motor_m: str = "A1"
    elevation_enc_a: str = "D6"
    elevation_enc_b: str = "D5"
    elevation_hall: str = "D7"
    azimuth_motor_p: str = "A2"
    azimuth_motor_m: str = "A3"
    azimuth_enc_a: str = "D3"
    azimuth_enc_b: str = "D2"
    azimuth_hall: str = "D4"
    motor_pwm_hz: int = 20000

    def __post_init__(self) -> None:
        for field_name in (
            "can_tx",
            "can_rx",
            "elevation_motor_p",
            "elevation_motor_m",
            "elevation_enc_a",
            "elevation_enc_b",
            "elevation_hall",
            "azimuth_motor_p",
            "azimuth_motor_m",
            "azimuth_enc_a",
            "azimuth_enc_b",
            "azimuth_hall",
        ):
            value = getattr(self, field_name)
            if value not in _ARDUINO_PIN_NAMES:
                raise ValueError(
                    f"pins.{field_name}={value!r} is not a valid Arduino pin label "
                    f"(expected one of {sorted(_ARDUINO_PIN_NAMES)})"
                )
        if int(self.motor_pwm_hz) <= 0:
            raise ValueError(f"pins.motor_pwm_hz must be positive, got {self.motor_pwm_hz}")


@dataclass(slots=True, frozen=True)
class SystemConstants:
    default_site: SiteConstants
    absorber: AbsorberConstants
    mirror: MirrorConstants
    fleet: FleetConstants
    control: ControlConstants
    actuator: ActuatorConstants = ActuatorConstants()
    pins: PinConstants = PinConstants()

    def mount_world(self, node_id: int) -> np.ndarray:
        return self.fleet.mount_by_id(node_id).mount_world(
            normal_angle_from_x_deg=self.absorber.normal_angle_from_x_deg
        )