"""
Body-frame facet layout for a rigid alt–az mirror assembly (design time only).

``FacetGridBody`` stores facet centers and normals in the mount body frame at
``(azimuth_deg, elevation_deg) = (0, 0)``. World placement and joint solving live elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from src.geometry import normalize
from src.sun import SunModel


def unit_mirror_normal_at_point(
    d_incoming: np.ndarray,
    point: np.ndarray,
    absorber_center: np.ndarray,
) -> np.ndarray:
    """
    Unit normal of a flat mirror at ``point`` that specularly reflects propagation direction
    ``d_incoming`` (unit) toward ``absorber_center`` (sign so ``dot(d_incoming, n) < 0``).
    """
    d = normalize(np.asarray(d_incoming, dtype=float).reshape(1, 3))[0]
    u = normalize((np.asarray(absorber_center, dtype=float) - np.asarray(point, dtype=float)).reshape(1, 3))[0]
    n = normalize((d - u).reshape(1, 3))[0]
    if float(np.dot(d, n)) > 0.0:
        n = -n
    return n.astype(float)


def unit_facet_normal_toward_point(
    facet_center: np.ndarray,
    sphere_center_world: np.ndarray,
    incoming_toward_facet: np.ndarray,
) -> np.ndarray:
    """Outward air-side normal ``∝ facet - sphere_center``, then incidence sign."""
    v = np.asarray(facet_center, dtype=float).reshape(3) - np.asarray(sphere_center_world, dtype=float).reshape(3)
    ln = float(np.linalg.norm(v))
    if ln < 1e-12:
        d = normalize(np.asarray(incoming_toward_facet, dtype=float).reshape(1, 3))[0]
        return (-d).astype(float)
    n = (v / ln).astype(float)
    d = normalize(np.asarray(incoming_toward_facet, dtype=float).reshape(1, 3))[0]
    if float(np.dot(d, n)) > 0.0:
        n = -n
    return n.astype(float)


def tangent_basis_from_normal(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal u, v spanning the plane perpendicular to unit n (right-handed with n)."""
    ref = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(n, ref))) > 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=float)
    u = normalize(np.cross(n, ref).reshape(1, 3))[0]
    v = normalize(np.cross(n, u).reshape(1, 3))[0]
    return u, v


@dataclass(slots=True)
class FacetGridBody:
    """Rigid facet lattice in body frame (pivot at origin of ``centers_body`` offsets)."""

    centers_body: np.ndarray  # (F, 3)
    normals_body: np.ndarray  # (F, 3)
    u_body: np.ndarray
    v_body: np.ndarray
    lattice_plane_normal_body: np.ndarray
    center_facet_index: int


def _validate_odd_counts(grid_nx: int, grid_ny: int) -> tuple[int, int, int, int, int]:
    nx = int(grid_nx)
    ny = int(grid_ny)
    if nx < 1 or ny < 1:
        raise ValueError("grid_nx and grid_ny must be positive.")
    if nx % 2 == 0 or ny % 2 == 0:
        raise ValueError("grid_nx and grid_ny must be odd so the mount pivot aligns with a facet center.")
    half_x = nx // 2
    half_y = ny // 2
    center_facet = half_y * nx + half_x
    return nx, ny, half_x, half_y, center_facet


def design_optimized_facet_grid(
    mount_world: np.ndarray,
    design_when_utc: datetime,
    absorber_center: np.ndarray,
    grid_nx: int,
    grid_ny: int,
    pitch_m: float,
    sun: SunModel,
) -> FacetGridBody:
    """Facet normals specular toward the absorber at ``design_when_utc`` (per facet center)."""
    nx, ny, half_x, half_y, center_facet = _validate_odd_counts(grid_nx, grid_ny)
    d_sun = sun.ray_direction(design_when_utc)
    m = np.asarray(mount_world, dtype=float).reshape(3)
    a = np.asarray(absorber_center, dtype=float).reshape(3)
    n_pi = unit_mirror_normal_at_point(d_sun, m, a)
    e_u, e_v = tangent_basis_from_normal(n_pi)
    centers_body: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    us: list[np.ndarray] = []
    vs: list[np.ndarray] = []
    for iy in range(ny):
        for ix in range(nx):
            du = (ix - half_x) * pitch_m
            dv = (iy - half_y) * pitch_m
            c_body = du * e_u + dv * e_v
            p_w = m + c_body
            nf = unit_mirror_normal_at_point(d_sun, p_w, a)
            u0, v0 = tangent_basis_from_normal(nf)
            centers_body.append(c_body)
            normals.append(nf)
            us.append(u0)
            vs.append(v0)
    return FacetGridBody(
        centers_body=np.stack(centers_body, axis=0),
        normals_body=np.stack(normals, axis=0),
        u_body=np.stack(us, axis=0),
        v_body=np.stack(vs, axis=0),
        lattice_plane_normal_body=n_pi.copy(),
        center_facet_index=center_facet,
    )


def design_spherical_facet_grid(
    mount_world: np.ndarray,
    design_when_utc: datetime,
    grid_nx: int,
    grid_ny: int,
    pitch_m: float,
    sun: SunModel,
    sphere_center_world: np.ndarray,
) -> FacetGridBody:
    """Facet normals radial to ``sphere_center_world`` (outward cap) at design time."""
    nx, ny, half_x, half_y, center_facet = _validate_odd_counts(grid_nx, grid_ny)
    d_sun = sun.ray_direction(design_when_utc)
    m = np.asarray(mount_world, dtype=float).reshape(3)
    o = np.asarray(sphere_center_world, dtype=float).reshape(3)
    n_pi = unit_facet_normal_toward_point(m, o, d_sun)
    e_u, e_v = tangent_basis_from_normal(n_pi)
    centers_body: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    us: list[np.ndarray] = []
    vs: list[np.ndarray] = []
    for iy in range(ny):
        for ix in range(nx):
            du = (ix - half_x) * pitch_m
            dv = (iy - half_y) * pitch_m
            c_body = du * e_u + dv * e_v
            p_w = m + c_body
            nf = unit_facet_normal_toward_point(p_w, o, d_sun)
            u0, v0 = tangent_basis_from_normal(nf)
            centers_body.append(c_body)
            normals.append(nf)
            us.append(u0)
            vs.append(v0)
    return FacetGridBody(
        centers_body=np.stack(centers_body, axis=0),
        normals_body=np.stack(normals, axis=0),
        u_body=np.stack(us, axis=0),
        v_body=np.stack(vs, axis=0),
        lattice_plane_normal_body=n_pi.copy(),
        center_facet_index=center_facet,
    )
