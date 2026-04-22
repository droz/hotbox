from __future__ import annotations

import numpy as np


def normalize(v: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    norm = np.maximum(norm, eps)
    return v / norm


def az_el_to_unit(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """World frame: x east, y north, z up."""
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    return np.array(
        [
            np.cos(el) * np.sin(az),
            np.cos(el) * np.cos(az),
            np.sin(el),
        ],
        dtype=float,
    )


def unit_to_az_el(direction: np.ndarray) -> tuple[float, float]:
    d = normalize(direction.reshape(1, 3))[0]
    elevation_deg = float(np.rad2deg(np.arcsin(np.clip(d[2], -1.0, 1.0))))
    azimuth_deg = float(np.rad2deg(np.arctan2(d[0], d[1]))) % 360.0
    return azimuth_deg, elevation_deg


def orthonormal_basis_from_direction(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return two unit vectors ``(u, v)`` spanning the plane perpendicular to ``direction``.

    Used by ``SunModel.sample_parallel_bundle`` and mirror-specific ray footprints; keep this
    implementation stable so sampling planes match across modules.
    """
    d = normalize(direction.reshape(1, 3))[0]
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(d, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    u = normalize(np.cross(d, ref).reshape(1, 3))[0]
    v = normalize(np.cross(d, u).reshape(1, 3))[0]
    return u, v
