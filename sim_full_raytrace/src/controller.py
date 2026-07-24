from __future__ import annotations

"""
Apply shared aiming to raytrace mirror grids.

Mount pointing is computed by :func:`hotbox_shared.solve_tracking` — the same
implementation used by the live controller (bisector seed + optional mount-offset refine).
"""

from datetime import datetime

import numpy as np

from hotbox_shared import MountJointLimits, bisector_normal_at_mount, solve_tracking

from src.absorber import SolarAbsorber
from src.flat_mirror_grid import AltAzFlatMirrorGrid
from src.sun import SunModel

# Backward-compatible alias for tests and callers.
bisector_normal_world = bisector_normal_at_mount


def solve_mount_angles_for_grid(
    grid: AltAzFlatMirrorGrid,
    when_utc: datetime,
    absorber_center: np.ndarray,
    absorber: SolarAbsorber,
    *,
    solve_for_mount_offset: bool = True,
    joint_limits: MountJointLimits | None = None,
) -> tuple[float, float]:
    """Alt-az angles for center-facet aiming (shared with the live controller)."""
    _ = absorber  # kept for call-site compatibility
    d_sun = np.asarray(grid.sun.ray_direction(when_utc), dtype=float).reshape(3)
    angles = solve_tracking(
        sun_direction_toward_scene=d_sun,
        mount_world=grid.mount_world,
        target_world=absorber_center,
        pivot_facet_normal_body=grid._pivot_facet_normal_body,
        mount_offset_d_m=float(grid.mount_offset_d_m),
        solve_for_mount_offset=solve_for_mount_offset,
        joint_limits=joint_limits,
    )
    return angles.azimuth_deg, angles.elevation_deg


def mirror_orientations_for_time(
    when_utc: datetime,
    sun: SunModel,
    absorber_center: np.ndarray,
    mirrors: list[AltAzFlatMirrorGrid],
    absorber: SolarAbsorber,
    *,
    solve_for_mount_offset: bool = True,
    joint_limits: MountJointLimits | None = None,
) -> list[tuple[float, float]]:
    """
    Aim each grid; return display angles from the pivot facet normal in W:
    ``(physical azimuth [deg], tilt from horizontal [deg])`` per mirror.
    """
    _ = sun
    out: list[tuple[float, float]] = []
    a = np.asarray(absorber_center, dtype=float).reshape(3)
    for g in mirrors:
        az, el = solve_mount_angles_for_grid(
            g,
            when_utc,
            a,
            absorber,
            solve_for_mount_offset=solve_for_mount_offset,
            joint_limits=joint_limits,
        )
        g.azimuth_deg, g.elevation_deg = az, el
        out.append((g.physical_mount_azimuth_deg(), g.physical_mount_tilt_deg()))
    return out
