from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hotbox_shared import MirrorGridSpec, MountJointLimits, solve_tracking_for_grid

from .config import OvenConfig
from .sun import SunVector


@dataclass(slots=True)
class TrackingTarget:
    azimuth_deg: float
    elevation_deg: float
    mode: str


def track_absorber(
    sun: SunVector,
    mirror_position_world: np.ndarray,
    absorber_world: np.ndarray,
    *,
    grid_nx: int = 3,
    grid_ny: int = 5,
    pitch_m: float = 0.26035,
    radius_of_curvature_m: float = 5.5,
    mount_offset_d_m: float = 0.0,
    solve_for_mount_offset: bool = True,
    joint_limits: MountJointLimits | None = None,
) -> TrackingTarget:
    grid = MirrorGridSpec(
        grid_nx=grid_nx,
        grid_ny=grid_ny,
        pitch_m=pitch_m,
        radius_of_curvature_m=radius_of_curvature_m,
        mount_offset_d_m=mount_offset_d_m,
    )
    angles = solve_tracking_for_grid(
        sun_direction_toward_scene=-np.asarray(sun.world_vector, dtype=float).reshape(3),
        mount_world=mirror_position_world,
        target_world=absorber_world,
        grid=grid,
        solve_for_mount_offset=solve_for_mount_offset,
        joint_limits=joint_limits,
    )
    mode = "parked" if angles.night_stow else "tracking"
    return TrackingTarget(azimuth_deg=angles.azimuth_deg, elevation_deg=angles.elevation_deg, mode=mode)


def safe_park(config: OvenConfig) -> TrackingTarget:
    return TrackingTarget(
        azimuth_deg=config.safe_park_azimuth_deg,
        elevation_deg=config.safe_park_elevation_deg,
        mode="parked",
    )
