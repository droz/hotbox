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

    @property
    def meridional_half_angle_rad(self) -> float:
        """Half the meridional arc in radians: (height/2) / R (matches surface_grid theta range)."""
        return 0.5 * self.height_m / self.radius_of_curvature_m

    @property
    def patch_half_extent_curved_dir_m(self) -> float:
        """
        Half-width along curved_dir (b) of the axis-aligned box that encloses the meridional arc.
        For theta in [-alpha, alpha], R*sin(theta) lies in [-R*sin(alpha), R*sin(alpha)].
        """
        r = self.radius_of_curvature_m
        return r * np.sin(self.meridional_half_angle_rad)

    @property
    def patch_max_extent_normal_m(self) -> float:
        """
        Maximum offset along mirror normal (n0) from patch center for points on that arc.
        R*(1 - cos(theta)) is in [0, R*(1 - cos(alpha))] for theta in [-alpha, alpha].
        """
        r = self.radius_of_curvature_m
        return r * (1.0 - np.cos(self.meridional_half_angle_rad))

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

    def _incoming_hit_t_radial(
        self, p: np.ndarray, d: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        For each ray p + t d, smallest t > 0 where the line hits this mirror's finite
        illuminated patch. Returns (t_hit, hit_mask, radial_unit_for_reflection).
        t_hit is nan where there is no such hit.
        """
        r = self.radius_of_curvature_m
        c0 = self.curvature_center_line_point
        c = self.center
        a = self.axis
        b = self.curved_dir
        n0 = self.normal
        m = p - c0

        ez = a
        ex = b
        ey = normalize(np.cross(ez, ex).reshape(1, 3))[0]
        e = np.stack([ex, ey, ez], axis=1)

        ml = m @ e
        dl = d @ e

        a_quad = dl[:, 0] ** 2 + dl[:, 1] ** 2
        b_quad = 2.0 * (ml[:, 0] * dl[:, 0] + ml[:, 1] * dl[:, 1])
        c_quad = ml[:, 0] ** 2 + ml[:, 1] ** 2 - r**2
        disc = b_quad**2 - 4.0 * a_quad * c_quad

        eps_a = 1e-12
        eps_t = 1e-8
        eps_disc = 1e-10
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

        hb = self.patch_half_extent_curved_dir_m
        zn = self.patch_max_extent_normal_m
        ok1, ru1 = self._finite_patch_mask_and_outward_normal(
            pts1, d, c0, c, a, b, n0, self.width_m, hb, zn
        )
        ok2, ru2 = self._finite_patch_mask_and_outward_normal(
            pts2, d, c0, c, a, b, n0, self.width_m, hb, zn
        )
        ok1 &= g1
        ok2 &= g2

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
        return t_hit, hit_mask, radial_unit

    def incoming_first_patch_hit_t(self, origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
        """
        Distance along each ray from its origin (first encounter with this mirror's patch).
        np.inf if the ray misses this mirror. Used for mutual shadowing (closer hit = upstream).
        """
        t_hit, hit_mask, _ = self._incoming_hit_t_radial(origins, directions)
        out = np.full(origins.shape[0], np.inf, dtype=float)
        out[hit_mask] = t_hit[hit_mask]
        return out

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
        p = rays.origins
        d = rays.directions
        t_hit, hit_mask, radial_unit = self._incoming_hit_t_radial(p, d)
        points = p + t_hit[:, None] * d

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
    def _finite_patch_mask_and_outward_normal(
        points: np.ndarray,
        d: np.ndarray,
        c0: np.ndarray,
        c: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        n0: np.ndarray,
        width_m: float,
        patch_half_extent_b_m: float,
        patch_max_extent_n0_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Clip infinite-cylinder hits to the finite mirror and build the shading normal.

        Meridional bounds use a fixed axis-aligned box in (curved_dir b, normal n0), derived
        once from height_m and R (same parametrization as surface_grid). That box slightly
        over-covers the circular arc at the corners (conservative). Per ray we only take
        dot products — no atan2.

        - Width: |(p - c)·a| <= width/2 (along cylinder axis).
        - Meridional slab: |(p - c)·b| <= patch_half_extent_b_m,
          0 <= (p - c)·n0 <= patch_max_extent_n0_m.
        """
        eps = 1e-9
        rel = points - c

        # --- Width along cylinder axis (patch center c). ---
        u = rel @ a
        inside_width = np.abs(u) <= 0.5 * width_m + eps

        # --- Meridional AABB in world (b, n0), precomputed from arc half-angle. ---
        along_b = rel @ b
        along_n0 = rel @ n0
        inside_height = (
            (np.abs(along_b) <= patch_half_extent_b_m + eps)
            & (along_n0 >= -eps)
            & (along_n0 <= patch_max_extent_n0_m + eps)
        )

        # --- Unit radial from axis (for reflection normal); same as before. ---
        axis_points = c0 + ((points - c0) @ a)[:, None] * a
        radial = points - axis_points
        radial_norm = np.linalg.norm(radial, axis=1)
        nonzero_r = radial_norm > 1e-10
        radial_geom = np.zeros_like(radial)
        radial_geom[nonzero_r] = radial[nonzero_r] / radial_norm[nonzero_r, None]

        # --- Outward normal for reflection; flip so incident ray meets the front face (d·n < 0). ---
        dot_dn = np.sum(d * radial_geom, axis=1)
        n_out = np.where((dot_dn > 0.0)[:, None], -radial_geom, radial_geom)
        dot_dn = np.sum(d * n_out, axis=1)
        front = dot_dn < 0.0

        mask = nonzero_r & inside_width & inside_height & front
        return mask, n_out
