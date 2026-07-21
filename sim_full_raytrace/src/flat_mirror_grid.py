"""
Frames (all right-handed, lengths in meters):

- **World** ``W``: ENU fixed to the site — +x east, +y north, +z up.
- **Assembly / mount body** ``B``: rigid with the mirror. At ``(azimuth_deg, elevation_deg) = (0, 0)``,
  ``B`` is aligned with ``W`` (same basis vectors). The mount pivot is ``mount_world`` in ``W``.
- **Facet data** is stored in ``B`` at identity mount: ``_centers_local``, ``_normals_local``, … are
  assembly coordinates (flat grid in ``xy``, ``z = 0``; normals from design).

**World placement** of a body point ``p_B``:

    ``p_W = mount_world + R_mount(az, el) @ p_B``

where ``R_mount`` is **body → world** (``grid_mount_rotation_matrix``). Facet world data uses
``R_chain = R_mount @ R_{B←L}`` with ``R_{B←L} = local_to_mount_body_rotation`` (identity for the
current spherical layout).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from hotbox_shared import (
    heading_and_tilt_from_normal,
    mount_az_el_align_body_normal_to_world,
    mount_rotation_matrix,
    normalize_mount_az_el,
)

from src.geometry import normalize, orthonormal_basis_from_direction
from src.mirror_grid_design import (
    FacetGridInLocalFrame,
    design_spherical_facet_grid,
)
from src.rays import RayBundle
from src.sun import SunModel

__all__ = (
    "AltAzFlatMirrorGrid",
    "FacetGridInLocalFrame",
    "mount_az_el_align_body_normal_to_world",
    "grid_mount_rotation_matrix",
)

# Shared mount kinematics (same implementation as the live controller).
grid_mount_rotation_matrix = mount_rotation_matrix
_normalize_mount_az_el = normalize_mount_az_el


@dataclass(slots=True)
class AltAzFlatMirrorGrid:
    """
    Rigid ``grid_nx``×``grid_ny`` mirror on one alt-az mount.

    Facet centers on a flat **xy** grid at **z = mount_offset_d_m** in assembly frame ``B``
    (default ``d = 0`` keeps the lattice in ``z = 0``); sphere at
    ``(0, 0, sphere_center_offset_m)_B``; each facet unit normal is ``normalize(O_B - P_B)``
    before the ``d`` shift (same as the controller scene).

    See module docstring for ``W`` / ``B`` and ``p_W = M + R_mount @ p_B``.
    """

    mount_world: np.ndarray
    grid_nx: int
    grid_ny: int
    pitch_m: float
    tile_half_m: float
    sun: SunModel
    sphere_center_offset_m: float
    mount_offset_d_m: float = 0.0
    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0

    _centers_local: np.ndarray = field(init=False)
    _normals_local: np.ndarray = field(init=False)
    _facet_u_local: np.ndarray = field(init=False)
    _facet_v_local: np.ndarray = field(init=False)
    _R_local_to_mount_body: np.ndarray = field(init=False)
    _center_facet: int = field(init=False)
    # +z_B: normal to the z=0 plane of facet centers (not the facet reflective normal).
    _lattice_plane_normal_body: np.ndarray = field(init=False)
    # Pivot facet n_B at (az,el)=(0,0); bisector tracking aligns this with the world target, not +z_B.
    _pivot_facet_normal_body: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        design = design_spherical_facet_grid(
            self.grid_nx,
            self.grid_ny,
            self.pitch_m,
            sphere_center_offset_m=float(self.sphere_center_offset_m),
        )
        self._ingest_local_design(design)

    def _ingest_local_design(self, design: FacetGridInLocalFrame) -> None:
        self._centers_local = np.asarray(design.centers_local, dtype=float).copy()
        # Shift facet lattice along +Z body by mount_offset_d so the center facet sits at Mn.
        d = float(self.mount_offset_d_m)
        if abs(d) > 0.0:
            self._centers_local = self._centers_local + np.array([0.0, 0.0, d], dtype=float)
        self._normals_local = np.asarray(design.normals_local, dtype=float).copy()
        self._facet_u_local = np.asarray(design.facet_u_local, dtype=float).copy()
        self._facet_v_local = np.asarray(design.facet_v_local, dtype=float).copy()
        self._R_local_to_mount_body = np.asarray(design.local_to_mount_body_rotation, dtype=float).copy()
        self._center_facet = int(design.center_facet_index)
        n_pi = self._R_local_to_mount_body[:, 2].astype(float).reshape(3)
        self._lattice_plane_normal_body = n_pi / max(float(np.linalg.norm(n_pi)), 1e-15)
        n0 = self._normals_local[self._center_facet].astype(float).reshape(3)
        self._pivot_facet_normal_body = n0 / max(float(np.linalg.norm(n0)), 1e-15)

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
        """Map facet centers/normals from ``B`` to ``W`` via ``R_mount @ R_{B←L}`` (see module docstring)."""
        az, el = _normalize_mount_az_el(float(azimuth_deg), float(elevation_deg))
        r = grid_mount_rotation_matrix(az, el)
        r_chain = r @ self._R_local_to_mount_body
        c_w = self.mount_world + (r_chain @ self._centers_local.T).T
        n_raw = (r_chain @ self._normals_local.T).T
        n_w = n_raw / np.maximum(np.linalg.norm(n_raw, axis=1, keepdims=True), 1e-15)
        u_raw = (r_chain @ self._facet_u_local.T).T
        u_w = u_raw / np.maximum(np.linalg.norm(u_raw, axis=1, keepdims=True), 1e-15)
        v_raw = (r_chain @ self._facet_v_local.T).T
        v_w = v_raw / np.maximum(np.linalg.norm(v_raw, axis=1, keepdims=True), 1e-15)
        return c_w, n_w, u_w, v_w

    def _world_facets(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self._world_facets_from_angles(self.azimuth_deg, self.elevation_deg)

    def physical_mount_tilt_deg(self) -> float:
        r = grid_mount_rotation_matrix(self.azimuth_deg, self.elevation_deg)
        n_w = r @ self._pivot_facet_normal_body.reshape(3)
        _, tilt_deg = heading_and_tilt_from_normal(n_w)
        return tilt_deg

    def physical_mount_azimuth_deg(self) -> float:
        r = grid_mount_rotation_matrix(self.azimuth_deg, self.elevation_deg)
        n_w = r @ self._pivot_facet_normal_body.reshape(3)
        azimuth_deg, _ = heading_and_tilt_from_normal(n_w)
        return azimuth_deg

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
