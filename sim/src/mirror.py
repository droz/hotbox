from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry import az_el_to_unit, normalize
from src.rays import RayBundle


@dataclass(slots=True)
class CylindricalMirror:
    radius_of_curvature_m: float
    width_m: float
    height_m: float
    post_height_m: float
    back_to_rotation_offset_m: float
    position_xy_m: tuple[float, float]
    azimuth_deg: float
    elevation_deg: float

    @property
    def rotation_point(self) -> np.ndarray:
        return np.array(
            [self.position_xy_m[0], self.position_xy_m[1], self.post_height_m],
            dtype=float,
        )

    @property
    def normal(self) -> np.ndarray:
        return az_el_to_unit(self.azimuth_deg, self.elevation_deg)

    @property
    def center(self) -> np.ndarray:
        return self.rotation_point + self.back_to_rotation_offset_m * self.normal

    @property
    def axis(self) -> np.ndarray:
        """Cylinder axis, constrained parallel to the ground."""
        z = np.array([0.0, 0.0, 1.0])
        a = np.cross(z, self.normal)
        if np.linalg.norm(a) < 1e-10:
            return np.array([1.0, 0.0, 0.0], dtype=float)
        return normalize(a.reshape(1, 3))[0]

    @property
    def curved_dir(self) -> np.ndarray:
        return normalize(np.cross(self.axis, self.normal).reshape(1, 3))[0]

    @property
    def curvature_center_line_point(self) -> np.ndarray:
        return self.center + self.radius_of_curvature_m * self.normal

    @property
    def sampling_radius_m(self) -> float:
        return 0.6 * np.sqrt(self.width_m**2 + self.height_m**2)

    def surface_grid(self, nu: int = 25, nv: int = 35) -> np.ndarray:
        u = np.linspace(-0.5 * self.width_m, 0.5 * self.width_m, nu)
        s = np.linspace(-0.5 * self.height_m, 0.5 * self.height_m, nv)
        uu, ss = np.meshgrid(u, s, indexing="xy")
        theta = ss / self.radius_of_curvature_m
        pts = (
            self.center
            + uu[..., None] * self.axis
            + self.radius_of_curvature_m * np.sin(theta)[..., None] * self.curved_dir
            + self.radius_of_curvature_m * (1.0 - np.cos(theta))[..., None] * self.normal
        )
        return pts

    def intersect_and_reflect(
        self,
        rays: RayBundle,
        target_point: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, RayBundle]:
        c0 = self.curvature_center_line_point
        a = self.axis
        c = self.center
        b = self.curved_dir
        n0 = self.normal
        r = self.radius_of_curvature_m

        p = rays.origins
        d = rays.directions
        m = p - c0
        m_perp = m - (m @ a)[:, None] * a
        d_perp = d - (d @ a)[:, None] * a

        aa = np.sum(d_perp * d_perp, axis=1)
        bb = 2.0 * np.sum(m_perp * d_perp, axis=1)
        cc = np.sum(m_perp * m_perp, axis=1) - r**2
        disc = bb**2 - 4.0 * aa * cc

        valid = (aa > 1e-12) & (disc >= 0.0)
        t = np.full(p.shape[0], np.nan, dtype=float)
        sqrt_disc = np.zeros_like(disc)
        sqrt_disc[valid] = np.sqrt(disc[valid])
        t1 = np.full_like(t, np.nan)
        t2 = np.full_like(t, np.nan)
        t1[valid] = (-bb[valid] - sqrt_disc[valid]) / (2.0 * aa[valid])
        t2[valid] = (-bb[valid] + sqrt_disc[valid]) / (2.0 * aa[valid])

        t_candidate = np.stack([t1, t2], axis=1)
        positive = t_candidate > 1e-8
        has_pos = np.any(positive, axis=1)
        valid &= has_pos

        # Default: nearest physical intersection.
        t[has_pos] = np.min(np.where(positive[has_pos], t_candidate[has_pos], np.inf), axis=1)

        # If a target is provided, choose the root that best aims the reflected ray toward it.
        if target_point is not None and np.any(valid):
            idx = np.where(valid)[0]
            p_idx = p[idx]
            d_idx = d[idx]
            tc = t_candidate[idx]  # (M, 2)
            pos = positive[idx]

            tc_safe = np.where(pos, tc, np.nan)
            points_c = p_idx[:, None, :] + tc_safe[:, :, None] * d_idx[:, None, :]

            u_c = np.sum((points_c - c) * a, axis=2)
            axis_points_c = c0 + u_c[:, :, None] * a
            radial_c = points_c - axis_points_c
            radial_norm_c = np.linalg.norm(radial_c, axis=2)
            nonzero_c = radial_norm_c > 1e-10
            normals_c = np.zeros_like(radial_c)
            normals_c[nonzero_c] = radial_c[nonzero_c] / radial_norm_c[nonzero_c, None]

            dot_dn_c = np.sum(d_idx[:, None, :] * normals_c, axis=2)
            reflected_c = d_idx[:, None, :] - 2.0 * dot_dn_c[:, :, None] * normals_c
            reflected_c = normalize(reflected_c.reshape(-1, 3)).reshape(-1, 2, 3)

            to_target = target_point.reshape(1, 1, 3) - points_c
            to_target = normalize(to_target.reshape(-1, 3)).reshape(-1, 2, 3)
            score = np.sum(reflected_c * to_target, axis=2)
            score[~pos] = -np.inf

            choose_second = score[:, 1] > score[:, 0]
            chosen_t = np.where(choose_second, tc[:, 1], tc[:, 0])
            t[idx] = chosen_t

        points = p + t[:, None] * d
        rel = points - c
        u = rel @ a
        inside_width = np.abs(u) <= 0.5 * self.width_m

        axis_points = c0 + u[:, None] * a
        radial = points - axis_points
        radial_norm = np.linalg.norm(radial, axis=1)
        nonzero = radial_norm > 1e-10
        radial_unit = np.zeros_like(radial)
        radial_unit[nonzero] = radial[nonzero] / radial_norm[nonzero, None]

        sin_theta = radial_unit @ b
        cos_theta = -(radial_unit @ n0)
        s = r * np.arctan2(sin_theta, cos_theta)
        inside_height = np.abs(s) <= 0.5 * self.height_m

        hit_mask = valid & inside_width & inside_height & nonzero
        normals = radial_unit

        dot_dn = np.sum(d * normals, axis=1)
        reflected_dirs = d - 2.0 * dot_dn[:, None] * normals
        reflected_dirs = normalize(reflected_dirs)

        reflected = RayBundle(
            origins=points,
            directions=reflected_dirs,
            powers_w=rays.powers_w.copy(),
        )
        reflected.powers_w[~hit_mask] = 0.0
        return hit_mask, points, reflected
