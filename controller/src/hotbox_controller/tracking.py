from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import OvenConfig
from .geometry import (
    mirror_normal_for_reflection,
    mount_az_el_align_body_normal_to_world,
    normalize,
    pivot_facet_normal_body,
)
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
    incoming = -normalize(sun.world_vector)
    mount = np.asarray(mirror_position_world, dtype=float).reshape(3)
    outgoing = normalize(np.asarray(absorber_world, dtype=float).reshape(3) - mount)
    bisector = mirror_normal_for_reflection(incoming, outgoing)
    pivot_normal_body = pivot_facet_normal_body(
        grid_nx=grid_nx,
        grid_ny=grid_ny,
        pitch_m=pitch_m,
        radius_of_curvature_m=radius_of_curvature_m,
    )
    azimuth_deg, elevation_deg = mount_az_el_align_body_normal_to_world(pivot_normal_body, bisector)
    return TrackingTarget(azimuth_deg=azimuth_deg, elevation_deg=elevation_deg, mode="tracking")


def safe_park(config: OvenConfig) -> TrackingTarget:
    return TrackingTarget(
        azimuth_deg=config.safe_park_azimuth_deg,
        elevation_deg=config.safe_park_elevation_deg,
        mode="parked",
    )
