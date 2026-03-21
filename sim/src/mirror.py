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

    def intersect_and_reflect(self, rays: RayBundle) -> tuple[np.ndarray, np.ndarray, RayBundle]:
        """
        Intersect parallel rays with the convex cylindrical patch and reflect.

        We use a right-handed cylinder frame with origin on the axis at the curvature
        center line (same as the infinite cylinder used for intersection):
          e_z = axis (along the cylinder),
          e_x = curved_dir (meridional tangent at patch center),
          e_y = e_z × e_x (completes the basis).

        In that frame the infinite cylinder is x² + y² = R² with generators || e_z.
        Rays are expressed in that frame for the quadratic solve, then candidate hits are
        checked in world coordinates against finite width/height and front-face lighting.

        Two positive roots mean the line meets the infinite cylinder twice; only roots that
        lie on the **finite mirror patch** count. If both do, we keep the smaller t (first
        encounter along the ray from the bundle origin, i.e. closest to the sun side).
        """
        r = self.radius_of_curvature_m
        c0 = self.curvature_center_line_point
        c = self.center
        a = self.axis
        b = self.curved_dir
        n0 = self.normal

        p = rays.origins
        d = rays.directions
        m = p - c0

        # --- Cylinder-local frame (origin c0 on axis): e_z || axis, e_x || meridian at patch center ---
        ez = a
        ex = b
        ey = normalize(np.cross(ez, ex).reshape(1, 3))[0]
        e = np.stack([ex, ey, ez], axis=1)

        # --- World → local: row i is (p·e_x, p·e_y, p·e_z) in the cylinder basis ---
        ml = m @ e
        dl = d @ e

        # --- Intersect ray (ml + t*dl) with infinite cylinder: (xy)^2 = r^2, z free ---
        a_quad = dl[:, 0] ** 2 + dl[:, 1] ** 2
        b_quad = 2.0 * (ml[:, 0] * dl[:, 0] + ml[:, 1] * dl[:, 1])
        c_quad = ml[:, 0] ** 2 + ml[:, 1] ** 2 - r**2
        disc = b_quad**2 - 4.0 * a_quad * c_quad

        eps_a = 1e-12
        eps_t = 1e-8
        eps_disc = 1e-10
        # Degenerate: ray parallel to axis in the cylinder wall plane, or tangent hit.
        can_solve = (a_quad > eps_a) & (disc > eps_disc)

        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
        t1 = np.full(p.shape[0], np.nan, dtype=float)
        t2 = np.full_like(t1, np.nan)
        t1[can_solve] = (-b_quad[can_solve] - sqrt_disc[can_solve]) / (2.0 * a_quad[can_solve])
        t2[can_solve] = (-b_quad[can_solve] + sqrt_disc[can_solve]) / (2.0 * a_quad[can_solve])

        g1 = can_solve & (t1 > eps_t)
        g2 = can_solve & (t2 > eps_t)
        pts1 = p + t1[:, None] * d
        pts2 = p + t2[:, None] * d

        ok1, ru1 = self._patch_front_mask_and_radial(
            pts1, d, r, c0, c, a, b, n0, self.width_m, self.height_m
        )
        ok2, ru2 = self._patch_front_mask_and_radial(
            pts2, d, r, c0, c, a, b, n0, self.width_m, self.height_m
        )
        ok1 &= g1
        ok2 &= g2

        # --- Choose valid root(s): prefer smallest t among patch-valid hits (nearest to sun) ---
        only1 = ok1 & ~ok2
        only2 = ok2 & ~ok1
        both = ok1 & ok2
        prefer1 = both & (t1 <= t2)
        prefer2 = both & (t1 > t2)

        t_hit = np.full(p.shape[0], np.nan, dtype=float)
        radial_unit = np.zeros_like(p)
        t_hit[only1] = t1[only1]
        t_hit[only2] = t2[only2]
        t_hit[prefer1] = t1[prefer1]
        t_hit[prefer2] = t2[prefer2]
        radial_unit[only1] = ru1[only1]
        radial_unit[only2] = ru2[only2]
        radial_unit[prefer1] = ru1[prefer1]
        radial_unit[prefer2] = ru2[prefer2]

        hit_mask = ok1 | ok2
        points = p + t_hit[:, None] * d

        # --- Reflect in world frame (normal = outward radial from axis) ---
        dot_dn = np.sum(d * radial_unit, axis=1)
        reflected_dirs = d.copy()
        reflected_dirs[hit_mask] = d[hit_mask] - 2.0 * dot_dn[hit_mask, None] * radial_unit[hit_mask]
        reflected_dirs = normalize(reflected_dirs)

        reflected = RayBundle(
            origins=points,
            directions=reflected_dirs,
            powers_w=rays.powers_w.copy(),
        )
        reflected.powers_w[~hit_mask] = 0.0
        return hit_mask, points, reflected

    @staticmethod
    def _patch_front_mask_and_radial(
        points: np.ndarray,
        d: np.ndarray,
        r: float,
        c0: np.ndarray,
        c: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        n0: np.ndarray,
        width_m: float,
        height_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (mask, unit_radial) for candidate hit points on the infinite cylinder:
        finite strip along a (width), meridional arc s (height), non-degenerate radius,
        and d·n < 0 for the oriented outward normal n.
        """
        u = (points - c) @ a
        inside_width = np.abs(u) <= 0.5 * width_m

        axis_points = c0 + ((points - c0) @ a)[:, None] * a
        radial = points - axis_points
        radial_norm = np.linalg.norm(radial, axis=1)
        nonzero_r = radial_norm > 1e-10
        radial_unit = np.zeros_like(radial)
        radial_geom = np.zeros_like(radial)
        radial_geom[nonzero_r] = radial[nonzero_r] / radial_norm[nonzero_r, None]

        # Meridional arc from patch normal (geometry only; sign of radial_geom arbitrary here).
        sin_theta = radial_geom @ b
        cos_theta = -(radial_geom @ n0)
        s = r * np.arctan2(sin_theta, cos_theta)
        inside_height = np.abs(s) <= 0.5 * height_m

        # Reflective normal: radial from axis, oriented so incident light hits the front (d·n < 0).
        dot_dn = np.sum(d * radial_geom, axis=1)
        n_out = np.where((dot_dn > 0.0)[:, None], -radial_geom, radial_geom)
        dot_dn = np.sum(d * n_out, axis=1)
        front = dot_dn < 0.0

        mask = nonzero_r & inside_width & inside_height & front
        return mask, n_out
