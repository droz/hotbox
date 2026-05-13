from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np

from src.geometry import normalize, orthonormal_basis_from_direction
from src.mirror_grid_design import (
    design_optimized_facet_grid,
    design_spherical_facet_grid,
    unit_facet_normal_toward_point,
    unit_mirror_normal_at_point,
)
from src.rays import RayBundle
from src.sun import SunModel

# Re-export for tests and callers that imported optics from this module.
__all__ = (
    "AltAzFlatMirrorGrid",
    "FacetDesignStrategy",
    "coarse_mount_angles_align_lattice_normal",
    "grid_mount_rotation_matrix",
    "unit_facet_normal_toward_point",
    "unit_mirror_normal_at_point",
)


FacetDesignStrategy = Literal["optimized", "spherical"]


def _normalize_mount_az_el(az_deg: float, el_deg: float) -> tuple[float, float]:
    """
    Mount convention (see ``grid_mount_rotation_matrix``):

    - **Elevation** ``[-90, 90]`` [deg]: right-handed rotation about **+world X** (east), then
    - **Azimuth** ``[0, 360)``: rotation about **+world Z** (up): ``R = R_z(az) @ R_x(el)``.
    """
    az = float(az_deg % 360.0)
    el = float(el_deg)
    if not np.isfinite(el) or abs(el) > 720.0:
        el = 45.0
    el = float(np.clip(el, -90.0, 90.0))
    return az, el


def grid_mount_rotation_matrix(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """
    Active rotation **body → world** (same frame: x east, y north, z up).

    ``p_w = R @ p_b``,  ``R = R_z(azimuth) @ R_x(elevation)``.
    """
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    cx, sx = np.cos(el), np.sin(el)
    r_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    cz, sz = np.cos(az), np.sin(az)
    r_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return r_z @ r_x


def coarse_mount_angles_align_lattice_normal(
    lattice_normal_body: np.ndarray,
    lattice_normal_target_world: np.ndarray,
) -> tuple[float, float]:
    """
    Approximate ``(azimuth_deg, elevation_deg)`` so ``R(az,el) @ n_body ≈ n_target`` (both unit).
    """
    nb = np.asarray(lattice_normal_body, dtype=float).reshape(3)
    nw = np.asarray(lattice_normal_target_world, dtype=float).reshape(3)
    nb = nb / max(float(np.linalg.norm(nb)), 1e-15)
    nw = nw / max(float(np.linalg.norm(nw)), 1e-15)

    def sqerr(az: float, el: float) -> float:
        r = grid_mount_rotation_matrix(az, el)
        d = r @ nb - nw
        return float(np.dot(d, d))

    best_az, best_el = 0.0, 45.0
    best_e = sqerr(best_az, best_el)

    for az in np.linspace(0.0, 359.0, 72, endpoint=True):
        for el in np.linspace(-90.0, 90.0, 37, endpoint=True):
            e = sqerr(float(az), float(el))
            if e < best_e:
                best_e, best_az, best_el = e, float(az), float(el)

    for span_az, span_el, step in ((14.0, 14.0, 2.0), (5.0, 5.0, 0.5)):
        for daz in np.arange(-span_az, span_az + 0.001, step):
            for del_ in np.arange(-span_el, span_el + 0.001, step):
                az = (best_az + float(daz)) % 360.0
                el = float(np.clip(best_el + float(del_), -90.0, 90.0))
                e = sqerr(az, el)
                if e < best_e:
                    best_e, best_az, best_el = e, az, el

    return _normalize_mount_az_el(best_az, best_el)


@dataclass(slots=True)
class AltAzFlatMirrorGrid:
    """
    Rigid ``grid_nx``×``grid_ny`` grid of square flat mirrors on one alt-az mount.

    Facet centers and normals in body frame are produced by ``mirror_grid_design`` at
    ``design_when_utc``. Mount orientation ``(azimuth_deg, elevation_deg)`` is solved by the
    controller from that rigid body model.
    """

    mount_world: np.ndarray
    design_when_utc: datetime
    absorber_center: np.ndarray
    grid_nx: int
    grid_ny: int
    pitch_m: float
    tile_half_m: float
    sun: SunModel
    facet_design: FacetDesignStrategy = "optimized"
    spherical_target_world: np.ndarray | None = None
    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0

    _c_body: np.ndarray = field(init=False)
    _n_body: np.ndarray = field(init=False)
    _u_body: np.ndarray = field(init=False)
    _v_body: np.ndarray = field(init=False)
    _center_facet: int = field(init=False)
    _lattice_plane_normal_body: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        m = np.asarray(self.mount_world, dtype=float).reshape(3)
        a = self.absorber_center.astype(float).reshape(3)
        if self.facet_design == "spherical":
            if self.spherical_target_world is None:
                raise ValueError("spherical_target_world is required when facet_design='spherical'.")
            body = design_spherical_facet_grid(
                m,
                self.design_when_utc,
                self.grid_nx,
                self.grid_ny,
                self.pitch_m,
                self.sun,
                np.asarray(self.spherical_target_world, dtype=float).reshape(3),
            )
        else:
            body = design_optimized_facet_grid(
                m,
                self.design_when_utc,
                a,
                self.grid_nx,
                self.grid_ny,
                self.pitch_m,
                self.sun,
            )
        self._c_body = body.centers_body
        self._n_body = body.normals_body
        self._u_body = body.u_body
        self._v_body = body.v_body
        self._center_facet = body.center_facet_index
        self._lattice_plane_normal_body = body.lattice_plane_normal_body

    @property
    def rotation_point(self) -> np.ndarray:
        return np.asarray(self.mount_world, dtype=float).copy()

    @property
    def center(self) -> np.ndarray:
        r = grid_mount_rotation_matrix(self.azimuth_deg, self.elevation_deg)
        return self.mount_world + r @ np.zeros(3, dtype=float)

    @property
    def back_to_rotation_offset_m(self) -> float:
        return 0.0

    def incoming_ray_bundle_extents(
        self, world_ray_direction: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        d = normalize(np.asarray(world_ray_direction, dtype=float).reshape(1, 3))[0]
        u_ax, v_ax = orthonormal_basis_from_direction(d)
        c_w, _, u_w, v_w = self._world_facets()
        h = self.tile_half_m
        mount = self.center.reshape(3)

        corners: list[np.ndarray] = []
        for f in range(c_w.shape[0]):
            for su in (-1.0, 1.0):
                for sv in (-1.0, 1.0):
                    corners.append(c_w[f] + su * h * u_w[f] + sv * h * v_w[f])
        p = np.stack(corners, axis=0)
        rel = p - mount.reshape(1, 3)
        s = rel @ u_ax
        t = rel @ v_ax
        span_u = float(np.ptp(s))
        span_t = float(np.ptp(t))
        margin = max(1e-5, 0.02 * max(span_u, span_t, 1e-9))
        s0, s1 = float(np.min(s)) - margin, float(np.max(s)) + margin
        t0, t1 = float(np.min(t)) - margin, float(np.max(t)) + margin
        mid_s = 0.5 * (s0 + s1)
        mid_t = 0.5 * (t0 + t1)
        bundle_c = mount + mid_s * u_ax + mid_t * v_ax
        hu = max(0.5 * (s1 - s0), 1e-6)
        hv = max(0.5 * (t1 - t0), 1e-6)
        return bundle_c.astype(float), float(hu), float(hv)

    def incoming_ray_bundle_facet_grid(
        self,
        when_utc: datetime,
        samples_u: int,
        samples_v: int,
        *,
        upstream_distance_m: float = 50.0,
    ) -> RayBundle:
        nu = max(int(samples_u), 1)
        nv = max(int(samples_v), 1)
        d = normalize(np.asarray(self.sun.ray_direction(when_utc), dtype=float).reshape(1, 3))[0]
        dni = self.sun.clear_sky_dni_w_per_m2(when_utc)
        c_w, n_w, u_w, v_w = self._world_facets()
        h = float(self.tile_half_m)

        u_edges = np.linspace(-h, h, nu + 1)
        v_edges = np.linspace(-h, h, nv + 1)
        uc = 0.5 * (u_edges[:-1] + u_edges[1:])
        vc = 0.5 * (v_edges[:-1] + v_edges[1:])
        uu, vv = np.meshgrid(uc, vc, indexing="xy")
        du = (2.0 * h) / nu
        dv = (2.0 * h) / nv

        origins_parts: list[np.ndarray] = []
        powers_parts: list[np.ndarray] = []
        facet_parts: list[np.ndarray] = []

        n_facets = int(c_w.shape[0])
        for f in range(n_facets):
            cos_inc = float(-np.dot(n_w[f], d))
            if cos_inc <= 1e-15:
                continue
            pw = dni * du * dv * cos_inc
            pts = c_w[f] + uu[..., None] * u_w[f] + vv[..., None] * v_w[f]
            pts_r = pts.reshape(-1, 3)
            n_pts = int(pts_r.shape[0])
            origins_parts.append(pts_r - upstream_distance_m * d)
            powers_parts.append(np.full(n_pts, pw, dtype=float))
            facet_parts.append(np.full(n_pts, f, dtype=np.int32))

        if not origins_parts:
            return RayBundle(
                origins=np.zeros((0, 3), dtype=float),
                directions=np.zeros((0, 3), dtype=float),
                powers_w=np.zeros((0,), dtype=float),
                target_facet=np.zeros((0,), dtype=np.int32),
            )

        origins = np.vstack(origins_parts)
        powers_w = np.concatenate(powers_parts)
        target_facet = np.concatenate(facet_parts)
        directions = np.repeat(d.reshape(1, 3), origins.shape[0], axis=0)
        return RayBundle(
            origins=origins,
            directions=directions,
            powers_w=powers_w,
            target_facet=target_facet,
        )

    def _world_facets_from_angles(
        self, azimuth_deg: float, elevation_deg: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        az, el = _normalize_mount_az_el(float(azimuth_deg), float(elevation_deg))
        r = grid_mount_rotation_matrix(az, el)
        c_w = self.mount_world + (r @ self._c_body.T).T
        n_w = (r @ self._n_body.T).T
        u_w = (r @ self._u_body.T).T
        v_w = (r @ self._v_body.T).T
        return c_w, n_w, u_w, v_w

    def _world_facets(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self._world_facets_from_angles(self.azimuth_deg, self.elevation_deg)

    def physical_mount_tilt_deg(self) -> float:
        r = grid_mount_rotation_matrix(self.azimuth_deg, self.elevation_deg)
        n_w = r @ self._lattice_plane_normal_body.reshape(3)
        nz = float(np.clip(n_w[2], -1.0, 1.0))
        return float(np.rad2deg(abs(np.arcsin(nz))))

    def physical_mount_azimuth_deg(self) -> float:
        r = grid_mount_rotation_matrix(self.azimuth_deg, self.elevation_deg)
        n_w = r @ self._lattice_plane_normal_body.reshape(3)
        return float(np.rad2deg(np.arctan2(float(n_w[0]), float(n_w[1]))) % 360.0)

    def tile_surface_grids(self, nu: int = 7, nv: int = 7) -> list[np.ndarray]:
        c_w, n_w, u_w, v_w = self._world_facets()
        h = self.tile_half_m
        su = np.linspace(-h, h, nu)
        sv = np.linspace(-h, h, nv)
        uu, vv = np.meshgrid(su, sv, indexing="xy")
        out: list[np.ndarray] = []
        for f in range(c_w.shape[0]):
            pts = c_w[f] + uu[..., None] * u_w[f] + vv[..., None] * v_w[f]
            out.append(pts.astype(float))
        return out

    def _ray_plane_hits(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        c: np.ndarray,
        n: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        p, d = origins, directions
        n_dot_p = np.sum(p * n, axis=1)
        n_dot_c = float(np.dot(n, c))
        n_dot_d = np.sum(d * n, axis=1)
        eps = 1e-10
        parallel = np.abs(n_dot_d) < eps
        t = np.full(p.shape[0], np.nan, dtype=float)
        ok = ~parallel
        t[ok] = (n_dot_c - n_dot_p[ok]) / n_dot_d[ok]
        ok &= t > 1e-8
        t[~ok] = np.nan

        q = p + np.where(np.isfinite(t), t, 0.0)[:, None] * d
        rel = q - c
        au = np.sum(rel * u, axis=1)
        av = np.sum(rel * v, axis=1)
        h = self.tile_half_m
        inside = (np.abs(au) <= h + 1e-9) & (np.abs(av) <= h + 1e-9)
        front = n_dot_d < -1e-10
        hit = ok & inside & front
        t[~hit] = np.nan
        return t, hit

    def incoming_first_patch_hit_t(self, origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
        c_w, n_w, u_w, v_w = self._world_facets()
        n_rays = origins.shape[0]
        t_best = np.full(n_rays, np.inf, dtype=float)
        for f in range(c_w.shape[0]):
            t_hit, hit = self._ray_plane_hits(origins, directions, c_w[f], n_w[f], u_w[f], v_w[f])
            cand = hit & np.isfinite(t_hit)
            t_best[cand] = np.minimum(t_best[cand], t_hit[cand])
        return t_best

    def intersect_and_reflect(self, rays: RayBundle) -> tuple[np.ndarray, np.ndarray, RayBundle]:
        p = rays.origins
        d = rays.directions
        n_rays = p.shape[0]
        c_w, n_w, u_w, v_w = self._world_facets()

        t_best = np.full(n_rays, np.inf, dtype=float)
        facet_idx = np.full(n_rays, -1, dtype=np.int32)
        tf = rays.target_facet
        use_hint = tf is not None and int(tf.shape[0]) == n_rays and n_rays > 0
        if use_hint:
            for f in range(c_w.shape[0]):
                m = tf == f
                if not np.any(m):
                    continue
                t_hit, hit = self._ray_plane_hits(p[m], d[m], c_w[f], n_w[f], u_w[f], v_w[f])
                idx = np.flatnonzero(m)
                ok = hit & np.isfinite(t_hit)
                t_best[idx[ok]] = t_hit[ok]
                facet_idx[idx[ok]] = f
        else:
            for f in range(c_w.shape[0]):
                t_hit, hit = self._ray_plane_hits(p, d, c_w[f], n_w[f], u_w[f], v_w[f])
                better = hit & np.isfinite(t_hit) & (t_hit < t_best)
                t_best[better] = t_hit[better]
                facet_idx[better] = f

        hit_mask = facet_idx >= 0
        t_use = t_best.copy()
        t_use[~hit_mask] = 0.0
        points = p + t_use[:, None] * d

        n_hit = np.zeros_like(p)
        for f in range(c_w.shape[0]):
            m = facet_idx == f
            if np.any(m):
                n_hit[m] = n_w[f]

        dot_dn = np.sum(d * n_hit, axis=1)
        reflected_dirs = d.copy()
        reflected_dirs[hit_mask] = d[hit_mask] - 2.0 * dot_dn[hit_mask, None] * n_hit[hit_mask]
        reflected_dirs = normalize(reflected_dirs)

        reflected = RayBundle(
            origins=points,
            directions=reflected_dirs,
            powers_w=rays.powers_w.copy(),
            target_facet=None,
        )
        reflected.powers_w[~hit_mask] = 0.0
        return hit_mask, points, reflected
