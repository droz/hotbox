from __future__ import annotations

"""
Mount control uses the same **world** ``W`` and assembly **body** ``B`` frames as ``flat_mirror_grid``.

For each timestep: **flat heliostat at the pivot** — unit normal ``n_W`` that reflects sun toward
the absorber (``bisector_normal_world``), then find ``(azimuth, elevation)`` so the **pivot facet**
normal ``n_{pivot,B}`` maps to ``n_W`` under ``R_mount = R_z(az) R_x(el)`` (closed form in
``mount_az_el_align_body_normal_to_world``).
"""

from datetime import datetime

import numpy as np

from src.absorber import SolarAbsorber
from src.flat_mirror_grid import (
    AltAzFlatMirrorGrid,
    _normalize_mount_az_el,
    mount_az_el_align_body_normal_to_world,
)
from src.geometry import normalize
from src.sun import SunModel


def bisector_normal_world(
    incoming_toward_scene: np.ndarray,
    mount_world: np.ndarray,
    target_world: np.ndarray,
) -> np.ndarray:
    """
    Unit normal of a **flat** mirror at ``mount_world`` that reflects ``incoming_toward_scene``
    (unit, toward the mount) toward ``target_world`` (sign ``n·incoming < 0``).
    """
    d = normalize(np.asarray(incoming_toward_scene, dtype=float).reshape(1, 3))[0]
    u = normalize((np.asarray(target_world, dtype=float) - np.asarray(mount_world, dtype=float)).reshape(1, 3))[0]
    n = normalize((d - u).reshape(1, 3))[0]
    if float(np.dot(d, n)) > 0.0:
        n = -n
    return n.astype(float)


def solve_mount_angles_for_grid(
    grid: AltAzFlatMirrorGrid,
    when_utc: datetime,
    absorber_center: np.ndarray,
    absorber: SolarAbsorber,
) -> tuple[float, float]:
    """
    Alt-az angles for **bisector tracking at the pivot**: align the pivot facet normal in ``B`` with
    the specular bisector ``n_W`` at ``mount_world``. Other facets keep their fixed cants in ``B``.
    """
    _ = absorber  # kept for call-site compatibility; pivot bisector tracking does not use absorber geometry.
    d_sun = grid.sun.ray_direction(when_utc)
    d_sun = np.asarray(d_sun, dtype=float).reshape(3)
    d_sun = d_sun / max(float(np.linalg.norm(d_sun)), 1e-15)
    m = np.asarray(grid.mount_world, dtype=float).reshape(3)
    a = np.asarray(absorber_center, dtype=float).reshape(3)
    n_bisector = bisector_normal_world(d_sun, m, a)
    n_pivot_b = np.asarray(grid._pivot_facet_normal_body, dtype=float).reshape(3)
    az, el = mount_az_el_align_body_normal_to_world(n_pivot_b, n_bisector)
    return _normalize_mount_az_el(float(az), float(el))


def mirror_orientations_for_time(
    when_utc: datetime,
    sun: SunModel,
    absorber_center: np.ndarray,
    mirrors: list[AltAzFlatMirrorGrid],
    absorber: SolarAbsorber,
) -> list[tuple[float, float]]:
    """
    Bisector tracking for each grid; return display angles from the **pivot facet** normal in ``W``:
    ``(physical azimuth [deg], tilt from horizontal [deg])`` per mirror.
    """
    out: list[tuple[float, float]] = []
    a = np.asarray(absorber_center, dtype=float).reshape(3)
    for g in mirrors:
        az, el = solve_mount_angles_for_grid(g, when_utc, a, absorber)
        g.azimuth_deg, g.elevation_deg = az, el
        out.append((g.physical_mount_azimuth_deg(), g.physical_mount_tilt_deg()))
    return out
