from __future__ import annotations

from typing import Any

import numpy as np

from .geometry import MirrorCalibration, normalize
from .protocol import MirrorStatus
from .sun import SunVector


def mount_rotation_matrix(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    cx, sx = np.cos(el), np.sin(el)
    cz, sz = np.cos(az), np.sin(az)
    r_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    r_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return r_z @ r_x


def mount_world_from_calibration(calibration: MirrorCalibration) -> np.ndarray:
    bearing = np.deg2rad(calibration.oa_bearing_deg)
    return np.array(
        [
            calibration.oa_distance_m * np.sin(bearing),
            calibration.oa_distance_m * np.cos(bearing),
            calibration.oa_height_delta_m,
        ],
        dtype=float,
    )


def default_mount_world(node_id: int, absorber_height_m: float) -> np.ndarray:
    # Temporary layout until calibration fills in true OA geometry.
    return np.array([2.0, float(node_id - 1), absorber_height_m], dtype=float)


def facet_center_world(
    mount_world: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
    mirror_offset_d_m: float,
) -> np.ndarray:
    r = mount_rotation_matrix(azimuth_deg, elevation_deg)
    return np.asarray(mount_world, dtype=float).reshape(3) + r @ np.array([0.0, 0.0, mirror_offset_d_m], dtype=float)


def reflect_ray(incoming_toward_mirror: np.ndarray, normal: np.ndarray) -> np.ndarray:
    d = normalize(incoming_toward_mirror)
    n = normalize(normal)
    return normalize(d - 2.0 * float(np.dot(d, n)) * n)


def build_mirror_scene_entry(
    *,
    node_id: int,
    mount_world: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
    mirror_offset_d_m: float,
    sun: SunVector,
    absorber_world: np.ndarray,
    upstream_distance_m: float = 4.0,
) -> dict[str, Any]:
    mount = np.asarray(mount_world, dtype=float).reshape(3)
    facet = facet_center_world(mount, azimuth_deg, elevation_deg, mirror_offset_d_m)
    normal = mount_rotation_matrix(azimuth_deg, elevation_deg) @ np.array([0.0, 0.0, 1.0], dtype=float)
    incoming = -normalize(sun.world_vector)
    reflected = reflect_ray(incoming, normal)
    absorber = np.asarray(absorber_world, dtype=float).reshape(3)
    to_absorber = normalize(absorber - facet)
    miss_m = float(np.linalg.norm(np.cross(to_absorber, reflected)))
    sun_start = facet - normalize(sun.world_vector) * upstream_distance_m
    reflected_end = facet + reflected * float(np.linalg.norm(absorber - facet))
    return {
        "node_id": node_id,
        "mount": mount.tolist(),
        "facet_center": facet.tolist(),
        "normal": normal.tolist(),
        "azimuth_deg": float(azimuth_deg),
        "elevation_deg": float(elevation_deg),
        "incoming": {"start": sun_start.tolist(), "end": facet.tolist()},
        "reflected": {"start": facet.tolist(), "end": reflected_end.tolist()},
        "to_absorber": {"start": facet.tolist(), "end": absorber.tolist()},
        "miss_m": miss_m,
    }


def build_estimated_scene(
    *,
    sun: SunVector,
    absorber_world: np.ndarray,
    statuses: dict[int, MirrorStatus],
    calibrations: dict[int, MirrorCalibration],
    absorber_height_m: float,
) -> dict[str, Any]:
    mirrors = []
    for node_id, status in sorted(statuses.items()):
        calibration = calibrations.get(node_id)
        if calibration is not None:
            mount = mount_world_from_calibration(calibration)
            offset = calibration.mirror_offset_d_m
            az = status.azimuth_deg + calibration.home_azimuth_offset_deg
            el = status.elevation_deg + calibration.home_elevation_offset_deg
        else:
            mount = default_mount_world(node_id, absorber_height_m)
            offset = 0.2
            az = status.azimuth_deg
            el = status.elevation_deg
        mirrors.append(
            build_mirror_scene_entry(
                node_id=node_id,
                mount_world=mount,
                azimuth_deg=az,
                elevation_deg=el,
                mirror_offset_d_m=offset,
                sun=sun,
                absorber_world=absorber_world,
            )
        )
    return {
        "label": "estimated",
        "absorber": {"center": np.asarray(absorber_world, dtype=float).reshape(3).tolist()},
        "sun": {
            "azimuth_deg": sun.azimuth_deg,
            "elevation_deg": sun.elevation_deg,
            "world_vector": sun.world_vector.tolist(),
            "display_position": (normalize(sun.world_vector) * 8.0).tolist(),
        },
        "mirrors": mirrors,
    }
