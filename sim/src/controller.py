from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from src.absorber import SolarAbsorber
from src.flat_mirror_grid import AltAzFlatMirrorGrid
from src.geometry import normalize, unit_to_az_el
from src.mirror import CylindricalMirror
from src.sun import SunModel


@dataclass(slots=True)
class Controller:
    mirror_positions_relative_to_absorber: list[np.ndarray]
    absorber_orientation_relative_to_north_deg: float

    def compute_mirror_orientations(
        self,
        when_utc: datetime,
        sun: SunModel,
        absorber_center: np.ndarray,
        mirror_back_to_rotation_offsets_m: list[float],
        iterations: int = 6,
    ) -> list[tuple[float, float]]:
        sun_dir = sun.ray_direction(when_utc)
        orientations: list[tuple[float, float]] = []

        for rel_pos, back_offset in zip(
            self.mirror_positions_relative_to_absorber, mirror_back_to_rotation_offsets_m, strict=True
        ):
            rotation_point = absorber_center + rel_pos

            # Fixed-point solve: mirror center depends on mirror normal through back-offset.
            n = normalize((absorber_center - rotation_point).reshape(1, 3))[0]
            for _ in range(iterations):
                mirror_center = rotation_point + back_offset * n
                toward_absorber = normalize((absorber_center - mirror_center).reshape(1, 3))[0]
                bisector = toward_absorber - sun_dir
                n = normalize(bisector.reshape(1, 3))[0]

            az_deg, el_deg = unit_to_az_el(n)
            orientations.append((az_deg, el_deg))

        return orientations

    def apply_for_time(
        self,
        when_utc: datetime,
        sun: SunModel,
        absorber_center: np.ndarray,
        mirrors: list[Any],
        absorber: SolarAbsorber | None = None,
    ) -> list[tuple[float, float]]:
        if mirrors and all(isinstance(m, AltAzFlatMirrorGrid) for m in mirrors):
            if absorber is None:
                raise TypeError("absorber= is required for AltAzFlatMirrorGrid mount solve.")
            out: list[tuple[float, float]] = []
            for g in mirrors:
                g.solve_mount_angles(when_utc, absorber_center, absorber)
                # Report physical lattice-plane angles (not raw R_z @ R_x joint parameters).
                out.append((g.physical_mount_azimuth_deg(), g.physical_mount_tilt_deg()))
            return out

        back_offsets = [m.back_to_rotation_offset_m for m in mirrors]
        az_el = self.compute_mirror_orientations(
            when_utc=when_utc,
            sun=sun,
            absorber_center=absorber_center,
            mirror_back_to_rotation_offsets_m=back_offsets,
        )
        for mirror, (az, el) in zip(mirrors, az_el, strict=True):
            mirror.azimuth_deg = az
            mirror.elevation_deg = el
        return az_el
