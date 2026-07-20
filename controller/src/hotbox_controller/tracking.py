from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import OvenConfig
from .geometry import az_el_from_normal, mirror_normal_for_reflection, normalize
from .sun import SunVector


@dataclass(slots=True)
class TrackingTarget:
    azimuth_deg: float
    elevation_deg: float
    mode: str


def track_absorber(sun: SunVector, mirror_position_world: np.ndarray, absorber_world: np.ndarray) -> TrackingTarget:
    incoming = -normalize(sun.world_vector)
    outgoing = normalize(np.asarray(absorber_world, dtype=float).reshape(3) - np.asarray(mirror_position_world, dtype=float).reshape(3))
    normal = mirror_normal_for_reflection(incoming, outgoing)
    azimuth_deg, elevation_deg = az_el_from_normal(normal)
    return TrackingTarget(azimuth_deg=azimuth_deg, elevation_deg=elevation_deg, mode="tracking")


def safe_park(config: OvenConfig) -> TrackingTarget:
    return TrackingTarget(
        azimuth_deg=config.safe_park_azimuth_deg,
        elevation_deg=config.safe_park_elevation_deg,
        mode="parked",
    )
