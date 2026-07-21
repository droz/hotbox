from __future__ import annotations

"""
Apply bisector tracking to raytrace mirror grids.

Mount pointing is computed by :func:`hotbox_shared.solve_bisector_tracking` — the same
implementation used by the live controller.
"""

from datetime import datetime

import numpy as np

from hotbox_shared import bisector_normal_at_mount, solve_bisector_tracking

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
) -> tuple[float, float]:
    """Alt-az angles for bisector tracking at the mount pivot."""
    _ = absorber  # kept for call-site compatibility
    d_sun = np.asarray(grid.sun.ray_direction(when_utc), dtype=float).reshape(3)
    angles = solve_bisector_tracking(
        sun_direction_toward_scene=d_sun,
        mount_world=grid.mount_world,
        target_world=absorber_center,
        pivot_facet_normal_body=grid._pivot_facet_normal_body,
    )
    return angles.azimuth_deg, angles.elevation_deg


def mirror_orientations_for_time(
    when_utc: datetime,
    sun: SunModel,
    absorber_center: np.ndarray,
    mirrors: list[AltAzFlatMirrorGrid],
    absorber: SolarAbsorber,
) -> list[tuple[float, float]]:
    """
    Bisector tracking for each grid; return display angles from the pivot facet normal in W:
    ``(physical azimuth [deg], tilt from horizontal [deg])`` per mirror.
    """
    _ = sun, when_utc
    out: list[tuple[float, float]] = []
    a = np.asarray(absorber_center, dtype=float).reshape(3)
    for g in mirrors:
        az, el = solve_mount_angles_for_grid(g, when_utc, a, absorber)
        g.azimuth_deg, g.elevation_deg = az, el
        out.append((g.physical_mount_azimuth_deg(), g.physical_mount_tilt_deg()))
    return out
