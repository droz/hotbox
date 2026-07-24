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
class SystemConstants:
    default_site: SiteConstants
    absorber: AbsorberConstants
    mirror: MirrorConstants
    fleet: FleetConstants
    control: ControlConstants

    def mount_world(self, node_id: int) -> np.ndarray:
        return self.fleet.mount_by_id(node_id).mount_world(
            normal_angle_from_x_deg=self.absorber.normal_angle_from_x_deg
        )