from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hotbox_shared import (
    bisector_normal,
    facet_normal_world,
    mount_az_el_align_body_normal_to_world,
    mount_rotation_matrix,
    normalize,
    normalize_mount_az_el,
    pivot_facet_normal_body,
)

# Backward-compatible alias used throughout the controller.
mirror_normal_for_reflection = bisector_normal

__all__ = [
    "MirrorCalibration",
    "az_el_from_normal",
    "facet_normal_world",
    "mirror_normal_for_reflection",
    "mount_az_el_align_body_normal_to_world",
    "mount_rotation_matrix",
    "normalize",
    "normalize_mount_az_el",
    "pivot_facet_normal_body",
]


def az_el_from_normal(normal_world: np.ndarray) -> tuple[float, float]:
    """Spherical heading of a unit direction (not inverse of alt-az mount kinematics)."""
    n = normalize(np.asarray(normal_world, dtype=float).reshape(3))
    azimuth_deg = float(np.rad2deg(np.arctan2(n[0], n[1]))) % 360.0
    elevation_deg = float(np.rad2deg(np.arcsin(np.clip(n[2], -1.0, 1.0))))
    return azimuth_deg, elevation_deg


@dataclass(slots=True)
class MirrorCalibration:
    node_id: int
    oa_bearing_deg: float
    oa_height_delta_m: float
    home_azimuth_offset_deg: float
    home_elevation_offset_deg: float
    oa_distance_m: float
    mirror_offset_d_m: float
