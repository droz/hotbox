from __future__ import annotations

from datetime import datetime

import numpy as np

from src.absorber import SolarAbsorber
from src.flat_mirror_grid import AltAzFlatMirrorGrid
from src.sun import SunModel


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
        g.solve_mount_angles(when_utc, a, absorber)
        out.append((g.physical_mount_azimuth_deg(), g.physical_mount_tilt_deg()))
    return out
