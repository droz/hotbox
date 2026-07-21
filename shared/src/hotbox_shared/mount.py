"""Alt-az mount kinematics shared by the live controller and raytrace simulation."""

from __future__ import annotations

import numpy as np

from .vectors import normalize


def mount_rotation_matrix(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """
    Active rotation **body → world**: ``p_W = R @ p_B``, ``R = R_z(az) @ R_x(el)``.

    Elevation is a right-handed rotation about **+world X** (east); azimuth about **+world Z** (up).
    """
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    cx, sx = np.cos(el), np.sin(el)
    cz, sz = np.cos(az), np.sin(az)
    r_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    r_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return r_z @ r_x


def normalize_mount_az_el(az_deg: float, el_deg: float) -> tuple[float, float]:
    """Clamp mount angles to ``azimuth in [0, 360)``, ``elevation in [-90, 90]``."""
    az = float(az_deg % 360.0)
    el = float(el_deg)
    if not np.isfinite(el) or abs(el) > 720.0:
        el = 45.0
    el = float(np.clip(el, -90.0, 90.0))
    return az, el


def pivot_facet_normal_body(
    *,
    grid_nx: int,
    grid_ny: int,
    pitch_m: float,
    radius_of_curvature_m: float,
) -> np.ndarray:
    """
    Unit reflective normal of the center facet in mount body frame at ``(az, el) = (0, 0)``.

    Assumes an odd facet grid with the center tile at the mount pivot and spherical design
    about ``(0, 0, R)`` in body coordinates.
    """
    _ = grid_nx, grid_ny, pitch_m  # grid dimensions affect off-center facets only
    center = np.array([0.0, 0.0, 0.0], dtype=float)
    sphere = np.array([0.0, 0.0, float(radius_of_curvature_m)], dtype=float)
    return normalize(sphere - center)


def mount_az_el_align_body_normal_to_world(
    body_normal: np.ndarray,
    target_normal_world: np.ndarray,
) -> tuple[float, float]:
    """
    Solve ``(azimuth_deg, elevation_deg)`` so ``R_mount(az, el) @ n_B ≈ n_W``.

    Closed-form inverse kinematics for ``R = R_z(az) @ R_x(el)``; no grid search in the
    typical case.
    """
    nb = normalize(body_normal)
    nw = normalize(target_normal_world)
    nx, ny, nz = float(nb[0]), float(nb[1]), float(nb[2])
    tx, ty, tz = float(nw[0]), float(nw[1]), float(nw[2])

    def sqerr(az_deg: float, el_deg: float) -> float:
        d = mount_rotation_matrix(az_deg, el_deg) @ nb - nw
        return float(np.dot(d, d))

    def finish(az_deg: float, el_deg: float) -> tuple[float, float]:
        return normalize_mount_az_el(float(az_deg), float(el_deg))

    r_yz = float(np.hypot(ny, nz))
    candidates: list[tuple[float, float]] = []

    if r_yz < 1e-12:
        vx, vy = nx, 0.0
        r_xy = float(np.hypot(tx, ty))
        if r_xy < 1e-12:
            return finish(0.0, 0.0)
        az_deg = float(np.rad2deg(np.arctan2(ty, tx) - np.arctan2(vy, vx)))
        return finish(az_deg, 0.0)

    phi = float(np.arctan2(nz, ny))
    arg = float(np.clip(tz / r_yz, -1.0, 1.0))
    for delta in (float(np.arcsin(arg)), float(np.pi - np.arcsin(arg))):
        el_rad = delta - phi
        el_deg = float(np.rad2deg(el_rad))
        if not (-90.0 - 1e-9 <= el_deg <= 90.0 + 1e-9):
            continue
        el_deg = float(np.clip(el_deg, -90.0, 90.0))
        elr = np.deg2rad(el_deg)
        cr, sr = np.cos(elr), np.sin(elr)
        vx = nx
        vy = cr * ny - sr * nz
        vz = sr * ny + cr * nz
        r_xy_v = float(np.hypot(vx, vy))
        r_xy_t = float(np.hypot(tx, ty))
        if r_xy_v < 1e-12 or r_xy_t < 1e-12:
            if abs(vz - tz) < 1e-6 and r_xy_t < 1e-12:
                candidates.append((0.0, el_deg))
            continue
        az_deg = float(np.rad2deg(np.arctan2(ty, tx) - np.arctan2(vy, vx)))
        candidates.append((az_deg, el_deg))

    if not candidates:
        el_cands = [-90.0, 90.0]
        el_crit = float(np.rad2deg(np.arctan2(ny, nz)))
        for shift in (-360.0, 0.0, 360.0):
            e = el_crit + shift
            if -90.0 <= e <= 90.0:
                el_cands.append(e)
        best_az_f, best_el_f, best_e_f = 0.0, 0.0, 1e30
        for el_deg in el_cands:
            elr = np.deg2rad(float(el_deg))
            cr, sr = np.cos(elr), np.sin(elr)
            vy = cr * ny - sr * nz
            vx = nx
            r_xy_v = float(np.hypot(vx, vy))
            r_xy_t = float(np.hypot(tx, ty))
            if r_xy_v < 1e-12:
                az_deg = 0.0
            else:
                az_deg = float(np.rad2deg(np.arctan2(ty, tx) - np.arctan2(vy, vx)))
            e = sqerr(az_deg, el_deg)
            if e < best_e_f:
                best_e_f, best_az_f, best_el_f = e, az_deg, el_deg
        return finish(best_az_f, best_el_f)

    best_az, best_el = candidates[0]
    best_e = sqerr(best_az, best_el)
    for az_deg, el_deg in candidates[1:]:
        e = sqerr(az_deg, el_deg)
        if e < best_e:
            best_e, best_az, best_el = e, az_deg, el_deg
    return finish(best_az, best_el)


def facet_normal_world(azimuth_deg: float, elevation_deg: float, pivot_normal_body: np.ndarray) -> np.ndarray:
    """World-frame pivot facet normal at the given mount angles."""
    return normalize(mount_rotation_matrix(azimuth_deg, elevation_deg) @ np.asarray(pivot_normal_body, dtype=float).reshape(3))


def heading_and_tilt_from_normal(normal_world: np.ndarray) -> tuple[float, float]:
    """
    Spherical heading of a unit direction in world frame.

    Returns ``(azimuth_deg, tilt_from_horizontal_deg)`` — display angles, not raw mount angles.
    """
    n = normalize(normal_world)
    azimuth_deg = float(np.rad2deg(np.arctan2(n[0], n[1]))) % 360.0
    tilt_deg = float(np.rad2deg(abs(np.arcsin(np.clip(n[2], -1.0, 1.0)))))
    return azimuth_deg, tilt_deg
