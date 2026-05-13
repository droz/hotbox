from __future__ import annotations

from datetime import datetime

import numpy as np
from scipy.optimize import least_squares

from src.absorber import SolarAbsorber
from src.flat_mirror_grid import (
    AltAzFlatMirrorGrid,
    _normalize_mount_az_el,
    coarse_mount_angles_align_lattice_normal,
)
from src.mirror_grid_design import unit_mirror_normal_at_point
from src.sun import SunModel

# Residuals are absorber (u, v) [m]; ~mm-scale aim.
_MOUNT_LS_FTOL = 1e-8
_MOUNT_LS_XTOL = 1e-6
_MOUNT_LS_GTOL = 1e-8
_MOUNT_LS_MAX_NFEV = 200


def mount_facet_absorber_uv_residuals(
    grid: AltAzFlatMirrorGrid,
    azimuth_deg: float,
    elevation_deg: float,
    d_sun: np.ndarray,
    absorber: SolarAbsorber,
) -> np.ndarray:
    """
    One ray per facet **center**: reflect ``d_sun`` off each facet plane, intersect the absorber
    plane, return stacked ``(u, v)`` residuals vs target 0. Length ``2 * n_facets``.
    """
    c_w, n_w, _, _ = grid._world_facets_from_angles(azimuth_deg, elevation_deg)
    d = np.asarray(d_sun, dtype=float).reshape(3)
    d = d / max(float(np.linalg.norm(d)), 1e-15)
    na = absorber.normal.reshape(3)
    ca = absorber.center.reshape(3)
    u_ax = absorber.horizontal_axis.reshape(3)
    v_ax = absorber.vertical_axis.reshape(3)

    dn = np.sum(n_w * d.reshape(1, 3), axis=1)
    bad = np.abs(dn) < 1e-12
    bad |= dn >= -1e-10
    r_dir = d.reshape(1, 3) - 2.0 * dn[:, None] * n_w
    denom = r_dir @ na
    bad |= np.abs(denom) < 1e-12
    t = np.sum((ca.reshape(1, 3) - c_w) * na.reshape(1, 3), axis=1) / np.where(
        np.abs(denom) < 1e-15, np.nan, denom
    )
    bad |= ~np.isfinite(t) | (t <= 1e-9)

    t_use = np.where(bad, 0.0, t)
    pt = c_w + t_use[:, None] * r_dir
    rel = pt - ca.reshape(1, 3)
    u = np.sum(rel * u_ax.reshape(1, 3), axis=1)
    v = np.sum(rel * v_ax.reshape(1, 3), axis=1)
    pen = 1e3
    u = np.where(bad, pen, u)
    v = np.where(bad, pen, v)
    return np.stack([u, v], axis=1).reshape(-1)


def solve_mount_angles_for_grid(
    grid: AltAzFlatMirrorGrid,
    when_utc: datetime,
    absorber_center: np.ndarray,
    absorber: SolarAbsorber,
) -> tuple[float, float]:
    """
    Alt-az joint angles minimizing facet-center reflected rays to the absorber ``(u,v)``.

    Initial guess: rotate the body so the **lattice-plane** normal matches the specular bisector
    at the **assembly pivot** between sun and absorber (same construction as a single flat
    heliostat aimed at the receiver). Refinement uses only the stored facet centers and normals.
    """
    d_sun = grid.sun.ray_direction(when_utc)
    d_sun = np.asarray(d_sun, dtype=float).reshape(3)
    d_sun = d_sun / max(float(np.linalg.norm(d_sun)), 1e-15)
    m = np.asarray(grid.mount_world, dtype=float).reshape(3)
    a = np.asarray(absorber_center, dtype=float).reshape(3)
    n_bisector = unit_mirror_normal_at_point(d_sun, m, a)
    az_seed, el_seed = coarse_mount_angles_align_lattice_normal(grid._lattice_plane_normal_body, n_bisector)
    x0 = np.array([az_seed, el_seed], dtype=float)

    def fun(x: np.ndarray) -> np.ndarray:
        az, el = _normalize_mount_az_el(float(x[0]), float(x[1]))
        return mount_facet_absorber_uv_residuals(grid, az, el, d_sun, absorber)

    bounds = (np.array([0.0, -90.0]), np.array([360.0, 90.0]))
    ls_kw: dict = {
        "bounds": bounds,
        "ftol": _MOUNT_LS_FTOL,
        "xtol": _MOUNT_LS_XTOL,
        "gtol": _MOUNT_LS_GTOL,
        "max_nfev": _MOUNT_LS_MAX_NFEV,
    }
    res = least_squares(fun, x0, **ls_kw)
    best_x = res.x
    best_cost = float(np.dot(res.fun, res.fun))

    alt_x0 = np.array([(float(x0[0]) + 180.0) % 360.0, float(x0[1])], dtype=float)
    alt_res = least_squares(fun, alt_x0, **ls_kw)
    alt_cost = float(np.dot(alt_res.fun, alt_res.fun))
    if alt_cost < best_cost:
        best_x = alt_res.x
    return _normalize_mount_az_el(float(best_x[0]), float(best_x[1]))


def mirror_orientations_for_time(
    when_utc: datetime,
    sun: SunModel,
    absorber_center: np.ndarray,
    mirrors: list[AltAzFlatMirrorGrid],
    absorber: SolarAbsorber,
) -> list[tuple[float, float]]:
    """
    Solve mount angles for each flat grid and return display angles:
    ``(physical azimuth [deg], lattice-plane tilt [deg])`` per mirror.
    """
    out: list[tuple[float, float]] = []
    a = np.asarray(absorber_center, dtype=float).reshape(3)
    for g in mirrors:
        az, el = solve_mount_angles_for_grid(g, when_utc, a, absorber)
        g.azimuth_deg, g.elevation_deg = az, el
        out.append((g.physical_mount_azimuth_deg(), g.physical_mount_tilt_deg()))
    return out
