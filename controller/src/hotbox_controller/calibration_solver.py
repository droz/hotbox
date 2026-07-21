from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .geometry import MirrorCalibration, facet_normal_world, mirror_normal_for_reflection, normalize, pivot_facet_normal_body


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

    pivot_normal_body = pivot_facet_normal_body(
        grid_nx=3,
        grid_ny=5,
        pitch_m=0.26035,
        radius_of_curvature_m=5.5,
    )

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
        normal1 = facet_normal_world(az1, el1, pivot_normal_body)
        look_dir = normalize(mount_world + mirror_offset_d_m * normal1 - np.zeros(3))
        err1 = look_dir - normalize(-mount_world)

        az2 = sample.focus_on_oven_az_deg - home_az_offset
        el2 = sample.focus_on_oven_el_deg - home_el_offset
        normal2 = facet_normal_world(az2, el2, pivot_normal_body)
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
