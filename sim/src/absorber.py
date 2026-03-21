from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry import normalize
from src.rays import RayBundle


@dataclass(slots=True)
class SolarAbsorber:
    width_m: float
    height_m: float
    center_height_m: float
    normal_angle_from_x_deg: float

    @property
    def center(self) -> np.ndarray:
        return np.array([0.0, 0.0, self.center_height_m], dtype=float)

    @property
    def normal(self) -> np.ndarray:
        a = np.deg2rad(self.normal_angle_from_x_deg)
        return np.array([np.cos(a), np.sin(a), 0.0], dtype=float)

    @property
    def vertical_axis(self) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0], dtype=float)

    @property
    def horizontal_axis(self) -> np.ndarray:
        return normalize(np.cross(self.vertical_axis, self.normal).reshape(1, 3))[0]

    def corners(self) -> np.ndarray:
        h = 0.5 * self.height_m
        w = 0.5 * self.width_m
        c = self.center
        u = self.horizontal_axis
        v = self.vertical_axis
        return np.array(
            [
                c - w * u - h * v,
                c + w * u - h * v,
                c + w * u + h * v,
                c - w * u + h * v,
            ]
        )

    def intersect(self, rays: RayBundle) -> tuple[np.ndarray, np.ndarray]:
        n = self.normal
        denom = rays.directions @ n
        eps = 1e-10
        valid = np.abs(denom) > eps

        t = np.full(rays.origins.shape[0], np.nan, dtype=float)
        t[valid] = ((self.center - rays.origins[valid]) @ n) / denom[valid]
        valid &= t > 0.0

        points = rays.origins + t[:, None] * rays.directions
        rel = points - self.center
        u = rel @ self.horizontal_axis
        v = rel @ self.vertical_axis
        inside = (np.abs(u) <= 0.5 * self.width_m) & (np.abs(v) <= 0.5 * self.height_m)
        hit_mask = valid & inside
        return hit_mask, points
