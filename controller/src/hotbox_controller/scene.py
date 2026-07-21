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


def default_mount_world(node_id: int, absorber_height_m: float, oa_distance_m: float = 2.5) -> np.ndarray:
    return np.array([oa_distance_m, float(node_id - 1), absorber_height_m], dtype=float)


def tangent_basis_from_normal(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = normalize(normal)
    ref = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(n, ref))) > 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=float)
    u = normalize(np.cross(n, ref))
    v = normalize(np.cross(n, u))
    return u, v


def design_spherical_facet_grid(
    grid_nx: int,
    grid_ny: int,
    pitch_m: float,
    *,
    sphere_center_offset_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return centers, normals, u, v in mount body frame (z=0 plane, +z out of mirror)."""
    nx, ny = int(grid_nx), int(grid_ny)
    half_x, half_y = nx // 2, ny // 2
    sphere = np.array([0.0, 0.0, float(sphere_center_offset_m)], dtype=float)
    centers: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    us: list[np.ndarray] = []
    vs: list[np.ndarray] = []
    for iy in range(ny):
        for ix in range(nx):
            p = np.array([(ix - half_x) * pitch_m, (iy - half_y) * pitch_m, 0.0], dtype=float)
            n = normalize(sphere - p)
            u, v = tangent_basis_from_normal(n)
            centers.append(p)
            normals.append(n)
            us.append(u)
            vs.append(v)
    return (
        np.stack(centers, axis=0),
        np.stack(normals, axis=0),
        np.stack(us, axis=0),
        np.stack(vs, axis=0),
    )


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


def build_oven_scene(
    *,
    absorber_center: np.ndarray,
    absorber_width_m: float,
    absorber_height_m: float,
    normal_angle_from_x_deg: float,
    fleet_mounts: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    absorber = np.asarray(absorber_center, dtype=float).reshape(3)
    # Prefer facing the mirror fleet; fall back to configured absorber normal.
    if fleet_mounts:
        fleet_mean = np.mean(np.stack([np.asarray(m, dtype=float).reshape(3) for m in fleet_mounts], axis=0), axis=0)
        facing = fleet_mean - absorber
        facing[2] = 0.0
        if float(np.linalg.norm(facing)) < 1e-9:
            ang = np.deg2rad(normal_angle_from_x_deg)
            facing = np.array([np.cos(ang), np.sin(ang), 0.0], dtype=float)
        normal = normalize(facing)
    else:
        ang = np.deg2rad(normal_angle_from_x_deg)
        normal = np.array([np.cos(ang), np.sin(ang), 0.0], dtype=float)

    # Oven body slightly larger than absorber; absorber on the back face (toward mirrors).
    depth_m = max(absorber_width_m, absorber_height_m) * 0.85
    body_w = absorber_width_m * 1.25
    body_h = absorber_height_m * 1.35
    body_center = absorber - normal * (depth_m * 0.5)
    handle_center = body_center - normal * (depth_m * 0.55)
    return {
        "absorber_center": absorber.tolist(),
        "absorber_size": [absorber_width_m, 0.04, absorber_height_m],
        "absorber_normal": normal.tolist(),
        "body_center": body_center.tolist(),
        "body_size": [body_w, depth_m, body_h],
        "handle_center": handle_center.tolist(),
        "handle_size": [body_w * 0.35, 0.05, 0.05],
    }


def build_mirror_scene_entry(
    *,
    node_id: int,
    mount_world: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
    mirror_offset_d_m: float,
    sun: SunVector,
    absorber_world: np.ndarray,
    grid_nx: int = 3,
    grid_ny: int = 5,
    pitch_m: float = 0.26035,
    tile_side_m: float = 0.254,
    radius_of_curvature_m: float = 5.5,
    upstream_distance_m: float = 4.0,
) -> dict[str, Any]:
    mount = np.asarray(mount_world, dtype=float).reshape(3)
    r_mount = mount_rotation_matrix(azimuth_deg, elevation_deg)
    centers_b, normals_b, us_b, vs_b = design_spherical_facet_grid(
        grid_nx,
        grid_ny,
        pitch_m,
        sphere_center_offset_m=radius_of_curvature_m,
    )
    # Shift facet lattice along +Z body by mount_offset_d so the center facet sits at Mn.
    centers_b = centers_b + np.array([0.0, 0.0, mirror_offset_d_m], dtype=float)

    facets = []
    for i in range(centers_b.shape[0]):
        c_w = mount + r_mount @ centers_b[i]
        n_w = r_mount @ normals_b[i]
        u_w = r_mount @ us_b[i]
        v_w = r_mount @ vs_b[i]
        facets.append(
            {
                "center": c_w.tolist(),
                "normal": normalize(n_w).tolist(),
                "u": normalize(u_w).tolist(),
                "v": normalize(v_w).tolist(),
                "half_m": tile_side_m * 0.5,
            }
        )

    center_index = (grid_ny // 2) * grid_nx + (grid_nx // 2)
    facet = np.asarray(facets[center_index]["center"], dtype=float)
    normal = np.asarray(facets[center_index]["normal"], dtype=float)
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
        "rotation": r_mount.tolist(),
        "facets": facets,
        "grid_nx": grid_nx,
        "grid_ny": grid_ny,
        "incoming": {"start": sun_start.tolist(), "end": facet.tolist()},
        "reflected": {"start": facet.tolist(), "end": reflected_end.tolist()},
        "to_absorber": {"start": facet.tolist(), "end": absorber.tolist()},
        "miss_m": miss_m,
    }


def _mirror_params_from_system(system: Any | None) -> dict[str, float | int]:
    if system is None:
        return {
            "grid_nx": 3,
            "grid_ny": 5,
            "pitch_m": 0.26035,
            "tile_side_m": 0.254,
            "radius_of_curvature_m": 5.5,
            "mount_offset_d_m": 0.2,
        }
    return {
        "grid_nx": system.mirror.grid_nx,
        "grid_ny": system.mirror.grid_ny,
        "pitch_m": system.mirror.pitch_m,
        "tile_side_m": system.mirror.tile_side_m,
        "radius_of_curvature_m": system.mirror.radius_of_curvature_m,
        "mount_offset_d_m": system.mirror.mount_offset_d_m,
    }


def build_estimated_scene(
    *,
    sun: SunVector,
    absorber_world: np.ndarray,
    statuses: dict[int, MirrorStatus],
    calibrations: dict[int, MirrorCalibration],
    absorber_height_m: float,
    default_oa_distance_m: float = 2.5,
    default_mirror_offset_d_m: float = 0.2,
    system: Any | None = None,
) -> dict[str, Any]:
    params = _mirror_params_from_system(system)
    mirrors = []
    mounts_for_oven: list[np.ndarray] = []
    for node_id, status in sorted(statuses.items()):
        calibration = calibrations.get(node_id)
        if calibration is not None:
            mount = mount_world_from_calibration(calibration)
            offset = calibration.mirror_offset_d_m
            az = status.azimuth_deg + calibration.home_azimuth_offset_deg
            el = status.elevation_deg + calibration.home_elevation_offset_deg
        else:
            if system is not None:
                try:
                    mount = system.mount_world(node_id)
                except KeyError:
                    mount = default_mount_world(node_id, absorber_height_m, default_oa_distance_m)
            else:
                mount = default_mount_world(node_id, absorber_height_m, default_oa_distance_m)
            offset = float(params["mount_offset_d_m"])
            az = status.azimuth_deg
            el = status.elevation_deg
        mounts_for_oven.append(mount)
        mirrors.append(
            build_mirror_scene_entry(
                node_id=node_id,
                mount_world=mount,
                azimuth_deg=az,
                elevation_deg=el,
                mirror_offset_d_m=offset,
                sun=sun,
                absorber_world=absorber_world,
                grid_nx=int(params["grid_nx"]),
                grid_ny=int(params["grid_ny"]),
                pitch_m=float(params["pitch_m"]),
                tile_side_m=float(params["tile_side_m"]),
                radius_of_curvature_m=float(params["radius_of_curvature_m"]),
            )
        )

    absorber_width = float(getattr(getattr(system, "absorber", None), "width_m", 0.4))
    absorber_height = float(getattr(getattr(system, "absorber", None), "height_m", 0.4))
    normal_deg = float(getattr(getattr(system, "absorber", None), "normal_angle_from_x_deg", 90.0))
    oven = build_oven_scene(
        absorber_center=absorber_world,
        absorber_width_m=absorber_width,
        absorber_height_m=absorber_height,
        normal_angle_from_x_deg=normal_deg,
        fleet_mounts=mounts_for_oven,
    )

    sun_distance_m = 10.0
    sun_pos = normalize(sun.world_vector) * sun_distance_m
    return {
        "label": "estimated",
        "frame": {"x": "east", "y": "north", "z": "up"},
        "ground_z": 0.0,
        "absorber": {"center": np.asarray(absorber_world, dtype=float).reshape(3).tolist()},
        "oven": oven,
        "sun": {
            "azimuth_deg": sun.azimuth_deg,
            "elevation_deg": sun.elevation_deg,
            "world_vector": sun.world_vector.tolist(),
            "display_position": sun_pos.tolist(),
            "distance_m": sun_distance_m,
        },
        "mirrors": mirrors,
    }
