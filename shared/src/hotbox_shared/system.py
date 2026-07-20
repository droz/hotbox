from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(slots=True, frozen=True)
class SiteConstants:
    latitude_deg: float
    longitude_deg: float
    altitude_m: float


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
    focal_length_m: float
    default_oa_distance_m: float
    default_mount_height_m: float

    @property
    def facet_count(self) -> int:
        return int(self.grid_nx * self.grid_ny)


@dataclass(slots=True, frozen=True)
class MountDesign:
    node_id: int
    bearing_deg: float
    oa_distance_m: float
    mount_height_m: float

    def mount_world(self) -> np.ndarray:
        bearing = math.radians(self.bearing_deg)
        return np.array(
            [
                self.oa_distance_m * math.sin(bearing),
                self.oa_distance_m * math.cos(bearing),
                self.mount_height_m,
            ],
            dtype=float,
        )


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


@dataclass(slots=True, frozen=True)
class SystemConstants:
    default_site: SiteConstants
    absorber: AbsorberConstants
    mirror: MirrorConstants
    fleet: FleetConstants
    control: ControlConstants
