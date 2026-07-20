from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("cannot normalize zero-length vector")
    return vector / norm


def mirror_normal_for_reflection(incoming_toward_mirror: np.ndarray, outgoing_from_mirror: np.ndarray) -> np.ndarray:
    incoming = normalize(np.asarray(incoming_toward_mirror, dtype=float).reshape(3))
    outgoing = normalize(np.asarray(outgoing_from_mirror, dtype=float).reshape(3))
    normal = normalize(incoming + outgoing)
    if float(np.dot(normal, incoming)) < 0.0:
        normal = -normal
    return normal


def az_el_from_normal(normal_world: np.ndarray) -> tuple[float, float]:
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
    focal_length_m: float
