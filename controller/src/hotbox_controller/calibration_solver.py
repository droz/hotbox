from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .geometry import MirrorCalibration, az_el_from_normal, mirror_normal_for_reflection, normalize


@dataclass(slots=True)
class CalibrationSample:
    look_at_oven_az_deg: float
    look_at_oven_el_deg: float
    focus_on_oven_az_deg: float
    focus_on_oven_el_deg: float
    sun_vector_world: np.ndarray


def solve_mirror_calibration(
    sample: CalibrationSample,
    *,
    node_id: int,
    oa_distance_m: float,
    mirror_offset_d_m: float,
    focal_length_m: float,
) -> MirrorCalibration:
    sun = normalize(np.asarray(sample.sun_vector_world, dtype=float).reshape(3))

    def residuals(params: np.ndarray) -> np.ndarray:
        bearing_deg, height_delta_m, home_az_offset, home_el_offset = params
        bearing = np.deg2rad(bearing_deg)
        mount_world = np.array(
            [
                oa_distance_m * np.sin(bearing),
                oa_distance_m * np.cos(bearing),
                height_delta_m,
            ],
            dtype=float,
        )
        az1 = sample.look_at_oven_az_deg - home_az_offset
        el1 = sample.look_at_oven_el_deg - home_el_offset
        normal1 = _mount_normal(az1, el1)
        look_dir = normalize(mount_world + mirror_offset_d_m * normal1 - np.zeros(3))
        err1 = look_dir - normalize(-mount_world)

        az2 = sample.focus_on_oven_az_deg - home_az_offset
        el2 = sample.focus_on_oven_el_deg - home_el_offset
        normal2 = _mount_normal(az2, el2)
        incoming = -sun
        outgoing = normalize(-mount_world)
        bisector = mirror_normal_for_reflection(incoming, outgoing)
        err2 = normal2 - bisector
        return np.concatenate([err1, err2])

    guess = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    result = least_squares(residuals, guess)
    bearing_deg, height_delta_m, home_az_offset, home_el_offset = result.x
    return MirrorCalibration(
        node_id=node_id,
        oa_bearing_deg=float(bearing_deg),
        oa_height_delta_m=float(height_delta_m),
        home_azimuth_offset_deg=float(home_az_offset),
        home_elevation_offset_deg=float(home_el_offset),
        oa_distance_m=oa_distance_m,
        mirror_offset_d_m=mirror_offset_d_m,
        focal_length_m=focal_length_m,
    )


def _mount_normal(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    body_z = np.array([0.0, 0.0, 1.0], dtype=float)
    cx, sx = np.cos(el), np.sin(el)
    cz, sz = np.cos(az), np.sin(az)
    r_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    r_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return normalize(r_z @ r_x @ body_z)
