from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from scipy.optimize import least_squares

from src.absorber import SolarAbsorber
from src.geometry import normalize, orthonormal_basis_from_direction
from src.rays import RayBundle
from src.sun import SunModel


def unit_mirror_normal_at_point(
    d_incoming: np.ndarray,
    point: np.ndarray,
    absorber_center: np.ndarray,
) -> np.ndarray:
    """
    Unit normal **n** of a flat mirror at ``point`` that specularly reflects propagation
    direction ``d_incoming`` (unit) toward ``absorber_center``.

    Vector construction only: ``n ∝ normalize(d_incoming - u)`` with ``u`` the unit vector
    from ``point`` toward the absorber, sign chosen so ``dot(d_incoming, n) < 0`` (incoming
    toward the reflective face).
    """
    d = normalize(np.asarray(d_incoming, dtype=float).reshape(1, 3))[0]
    u = normalize((np.asarray(absorber_center, dtype=float) - np.asarray(point, dtype=float)).reshape(1, 3))[0]
    n = normalize((d - u).reshape(1, 3))[0]
    if float(np.dot(d, n)) > 0.0:
        n = -n
    return n.astype(float)


def _normalize_mount_az_el(az_deg: float, el_deg: float) -> tuple[float, float]:
    """
    Mount convention (see ``grid_mount_rotation_matrix``):

    - **Elevation** ``[-90, 90]`` [deg]: right-handed rotation about **+world X** (east), then
    - **Azimuth** ``[0, 360)``: rotation about **+world Z** (up): ``R = R_z(az) @ R_x(el)``.

    At ``(0, 0)`` the grid has the **design** attitude built in ``__post_init__`` (facet-center
    plane through the pivot with normal from ``unit_mirror_normal_at_point`` at the mount).
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

    Kinematic order: rotate the grid about **+X** by ``elevation_deg``, then about **+Z** by
    ``azimuth_deg``. For a column body vector ``p_b``,

        ``p_w = R @ p_b``,  ``R = R_z(azimuth) @ R_x(elevation)``.

    At ``(az, el) = (0, 0)``, ``R = I`` and body vectors match the design built from sun /
    absorber geometry (facet-center plane through the pivot, normal from specular bisector).
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

    Assumes a **rigid** mount: the lattice-plane normal in body is ``lattice_normal_body``; the
    desired **world** normal for a flat mirror at the pivot (ignoring facet offsets) is
    ``lattice_normal_target_world`` — typically ``unit_mirror_normal_at_point(d_sun, mount, A)``.

    Coarse global search plus local refinements; no dependence on the previous timestep, so
    day-long sweeps stay stable.
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

    for az in np.linspace(0.0, 359.0, 72, endpoint=True):  # ~5°
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


def _tangent_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal u, v spanning the plane perpendicular to unit n."""
    ref = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(n, ref))) > 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=float)
    u = normalize(np.cross(n, ref).reshape(1, 3))[0]
    v = normalize(np.cross(n, u).reshape(1, 3))[0]
    return u, v


@dataclass(slots=True)
class AltAzFlatMirrorGrid:
    """
    Rigid N×N grid of square flat mirrors on one alt-az mount.

    **Design (``design_when_utc``, mount joints 0)** — vector construction, no angles:

    - **Lattice plane** (facet centers): through ``mount_world`` with unit normal ``n_π``
      such that a **single** flat mirror with normal ``n_π`` would reflect the sun ray toward
      ``absorber_center`` (same rule as ``unit_mirror_normal_at_point`` at the pivot).
    - Centers form a ``pitch_m`` square grid in that plane (two orthonormal in-plane axes from
      ``n_π`` via cross products).
    - Each tile still has its own unit normal (small cant) so the ray through **that** center
      reflects to the absorber at the design instant.

    **World placement:** ``p_w = mount_world + R @ p_body`` with
    ``R = grid_mount_rotation_matrix(azimuth_deg, elevation_deg)``. Joint angles are only for
    the 2-DOF mount solver and for ``physical_mount_*`` display helpers.
    """

    mount_world: np.ndarray  # pivot M (world); body vectors below are design-time ENU offsets
    design_when_utc: datetime
    absorber_center: np.ndarray
    grid_n: int
    pitch_m: float
    tile_half_m: float
    sun: SunModel
    # Joint angles for R_z(az) @ R_x(el) (deg). Plots use physical_mount_*().
    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0

    _c_body: np.ndarray = field(init=False)  # (F, 3)
    _n_body: np.ndarray = field(init=False)  # (F, 3)
    _u_body: np.ndarray = field(init=False)  # (F, 3)
    _v_body: np.ndarray = field(init=False)  # (F, 3)
    _center_facet: int = field(init=False)
    _lattice_plane_normal_body: np.ndarray = field(init=False)  # unit; lattice π at design

    def __post_init__(self) -> None:
        n = self.grid_n
        if n < 1 or n % 2 == 0:
            raise ValueError("grid_n must be a positive odd integer (e.g. 5).")
        half = n // 2
        self._center_facet = half * n + half

        d_sun = self.sun.ray_direction(self.design_when_utc)
        m = np.asarray(self.mount_world, dtype=float).reshape(3)
        a = self.absorber_center.astype(float).reshape(3)

        # Single-flat-mirror lattice plane through M (vectors only).
        n_pi = unit_mirror_normal_at_point(d_sun, m, a)
        self._lattice_plane_normal_body = n_pi.copy()
        e_u, e_v = _tangent_basis(n_pi)

        centers_body: list[np.ndarray] = []
        normals: list[np.ndarray] = []
        us: list[np.ndarray] = []
        vs: list[np.ndarray] = []

        for iy in range(n):
            for ix in range(n):
                du = (ix - half) * self.pitch_m
                dv = (iy - half) * self.pitch_m
                c_body = du * e_u + dv * e_v
                p_w = m + c_body

                nf = unit_mirror_normal_at_point(d_sun, p_w, a)
                u0, v0 = _tangent_basis(nf)
                centers_body.append(c_body)
                normals.append(nf)
                us.append(u0)
                vs.append(v0)

        self._c_body = np.stack(centers_body, axis=0)
        self._n_body = np.stack(normals, axis=0)
        self._u_body = np.stack(us, axis=0)
        self._v_body = np.stack(vs, axis=0)

    @property
    def rotation_point(self) -> np.ndarray:
        return np.asarray(self.mount_world, dtype=float).copy()

    @property
    def center(self) -> np.ndarray:
        """Grid centroid in world (for ray bundle placement)."""
        r = grid_mount_rotation_matrix(self.azimuth_deg, self.elevation_deg)
        return self.mount_world + r @ np.zeros(3, dtype=float)

    @property
    def back_to_rotation_offset_m(self) -> float:
        return 0.0

    def incoming_ray_bundle_extents(
        self, world_ray_direction: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        """
        Tight axis-aligned sampling footprint in the plane ⊥ ``world_ray_direction``.

        Projects all tile corners onto the same ``(u, v)`` basis as ``SunModel.sample_parallel_bundle``
        (via ``orthonormal_basis_from_direction``), takes their bounding rectangle, adds a small
        margin, and returns ``(bundle_center_world, half_u_m, half_v_m)`` so parallel rays fill
        that rectangle instead of a loose circumscribing square.
        """
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
        """
        Tilt [deg] of the lattice plane π from vertical toward horizontal: **0** = π vertical
        (π normal horizontal), **90** = π horizontal (π normal vertical, up or down).

        Uses ``|arcsin(n_z)|`` so the reported tilt stays in ``[0, 90]`` when the solved normal
        dips slightly below the horizon (numerics / back-face); magnitude matches "how vertical
        is the lattice plane".
        """
        r = grid_mount_rotation_matrix(self.azimuth_deg, self.elevation_deg)
        n_w = r @ self._lattice_plane_normal_body.reshape(3)
        nz = float(np.clip(n_w[2], -1.0, 1.0))
        return float(np.rad2deg(abs(np.arcsin(nz))))

    def physical_mount_azimuth_deg(self) -> float:
        """Azimuth [deg] of the lattice-plane normal projected onto the horizontal (x east, y north)."""
        r = grid_mount_rotation_matrix(self.azimuth_deg, self.elevation_deg)
        n_w = r @ self._lattice_plane_normal_body.reshape(3)
        return float(np.rad2deg(np.arctan2(float(n_w[0]), float(n_w[1]))) % 360.0)

    def _facet_absorber_uv_residuals(
        self,
        azimuth_deg: float,
        elevation_deg: float,
        d_sun: np.ndarray,
        absorber: SolarAbsorber,
    ) -> np.ndarray:
        """
        For each facet, take the sun ray through the facet center, reflect off the facet plane,
        intersect the **absorber plane**, and return ``(u, v)`` in the absorber frame
        (``horizontal_axis``, ``vertical_axis`` through ``absorber.center``).

        Stacked as length ``2 * n_facets``: ``[u0, v0, u1, v1, ...]`` — target ``0`` for all
        (nonlinear least squares in ``solve_mount_angles``).
        """
        c_w, n_w, _, _ = self._world_facets_from_angles(azimuth_deg, elevation_deg)
        d = np.asarray(d_sun, dtype=float).reshape(3)
        d = d / max(float(np.linalg.norm(d)), 1e-15)
        na = absorber.normal.reshape(3)
        ca = absorber.center.reshape(3)
        u_ax = absorber.horizontal_axis.reshape(3)
        v_ax = absorber.vertical_axis.reshape(3)

        dn = np.sum(n_w * d.reshape(1, 3), axis=1)
        bad = np.abs(dn) < 1e-12
        r_dir = d.reshape(1, 3) - 2.0 * dn[:, None] * n_w
        denom = r_dir @ na
        bad |= np.abs(denom) < 1e-12
        t = np.sum((ca.reshape(1, 3) - c_w) * na.reshape(1, 3), axis=1) / np.where(
            np.abs(denom) < 1e-15, np.nan, denom
        )
        bad |= ~np.isfinite(t) | (t <= 1e-9)

        t_use = np.where(bad, 0.0, t)
        pt = c_w + t_use[:, None] * r_dir
        rel = pt - ca.reshape(1, 3)
        u = np.sum(rel * u_ax.reshape(1, 3), axis=1)
        v = np.sum(rel * v_ax.reshape(1, 3), axis=1)
        pen = 1e3
        u = np.where(bad, pen, u)
        v = np.where(bad, pen, v)
        return np.stack([u, v], axis=1).reshape(-1)

    def solve_mount_angles(
        self,
        when_utc: datetime,
        absorber_center: np.ndarray,
        absorber: SolarAbsorber,
    ) -> tuple[float, float]:
        """
        Find ``(azimuth_deg, elevation_deg)`` by minimizing facet reflected-ray hits in the
        absorber ``(u, v)`` frame (one sun ray per facet center, ``2 * n_facets`` residuals).

        Uses ``scipy.optimize.least_squares`` with a seed from ``coarse_mount_angles_align_lattice_normal``.
        ``absorber_center`` is kept for API compatibility; the target frame is ``absorber``.
        """
        d_sun = self.sun.ray_direction(when_utc)
        m = np.asarray(self.mount_world, dtype=float).reshape(3)
        a = np.asarray(absorber_center, dtype=float).reshape(3)
        n_flat = unit_mirror_normal_at_point(d_sun, m, a)
        az_seed, el_seed = coarse_mount_angles_align_lattice_normal(self._lattice_plane_normal_body, n_flat)
        x0 = np.array([az_seed, el_seed], dtype=float)

        def fun(x: np.ndarray) -> np.ndarray:
            az, el = _normalize_mount_az_el(float(x[0]), float(x[1]))
            return self._facet_absorber_uv_residuals(az, el, d_sun, absorber)

        res = least_squares(
            fun,
            x0,
            bounds=(np.array([0.0, -90.0]), np.array([360.0, 90.0])),
            ftol=1e-12,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=400,
        )
        best_x = res.x
        best_cost = float(np.dot(res.fun, res.fun))

        for alt in (np.array([0.0, 0.0], dtype=float), np.array([(float(x0[0]) + 180.0) % 360.0, float(x0[1])], dtype=float)):
            alt_res = least_squares(
                fun,
                alt,
                bounds=(np.array([0.0, -90.0]), np.array([360.0, 90.0])),
                ftol=1e-12,
                xtol=1e-10,
                gtol=1e-10,
                max_nfev=400,
            )
            c = float(np.dot(alt_res.fun, alt_res.fun))
            if c < best_cost:
                best_cost = c
                best_x = alt_res.x

        self.azimuth_deg, self.elevation_deg = _normalize_mount_az_el(float(best_x[0]), float(best_x[1]))
        return self.azimuth_deg, self.elevation_deg

    def tile_surface_grids(self, nu: int = 7, nv: int = 7) -> list[np.ndarray]:
        """Each tile: (nu, nv, 3) world corners for Plotly Surface."""
        c_w, n_w, u_w, v_w = self._world_facets()
        h = self.tile_half_m
        su = np.linspace(-h, h, nu)
        sv = np.linspace(-h, h, nv)
        uu, vv = np.meshgrid(su, sv, indexing="xy")
        out: list[np.ndarray] = []
        for f in range(c_w.shape[0]):
            pts = (
                c_w[f]
                + uu[..., None] * u_w[f]
                + vv[..., None] * v_w[f]
            )
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
        """Returns (t_hit, hit_mask) for one facet; t nan where miss."""
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
        )
        reflected.powers_w[~hit_mask] = 0.0
        return hit_mask, points, reflected
