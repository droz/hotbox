from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hotbox_shared import MirrorGridSpec, solve_bisector_tracking_for_grid

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
) -> TrackingTarget:
    grid = MirrorGridSpec(
        grid_nx=grid_nx,
        grid_ny=grid_ny,
        pitch_m=pitch_m,
        radius_of_curvature_m=radius_of_curvature_m,
    )
    angles = solve_bisector_tracking_for_grid(
        sun_direction_toward_scene=-np.asarray(sun.world_vector, dtype=float).reshape(3),
        mount_world=mirror_position_world,
        target_world=absorber_world,
        grid=grid,
    )
    return TrackingTarget(azimuth_deg=angles.azimuth_deg, elevation_deg=angles.elevation_deg, mode="tracking")


def safe_park(config: OvenConfig) -> TrackingTarget:
    return TrackingTarget(
        azimuth_deg=config.safe_park_azimuth_deg,
        elevation_deg=config.safe_park_elevation_deg,
        mode="parked",
    )
