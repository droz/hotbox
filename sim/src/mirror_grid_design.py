"""
Facet layout in **mirror assembly frame** (mount body at azimuth=0°, elevation=0°).

- Facet centers: flat grid in the **xy** plane at **z = 0** (pitch along x and y).
- Sphere center: **(0, 0, sphere_center_offset_m)** in the same frame.
- Each facet unit normal: **normalize(sphere_center − facet_center)** (toward the sphere center).

World placement: ``p_W = M + R_mount(az,el) @ p_assembly`` (``local_to_mount_body_rotation`` is identity).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry import normalize


def tangent_basis_from_normal(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal u, v spanning the plane perpendicular to unit n (right-handed with n)."""
    ref = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(n, ref))) > 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=float)
    u = normalize(np.cross(n, ref).reshape(1, 3))[0]
    v = normalize(np.cross(n, u).reshape(1, 3))[0]
    return u, v


@dataclass(slots=True, frozen=True)
class FacetGridInLocalFrame:
    """Facet geometry in assembly frame; ``local_to_mount_body_rotation`` maps assembly → body (here: I)."""

    centers_local: np.ndarray  # (F, 3), z = 0
    normals_local: np.ndarray  # (F, 3) unit, toward sphere center
    facet_u_local: np.ndarray
    facet_v_local: np.ndarray
    local_to_mount_body_rotation: np.ndarray  # (3, 3)
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


def design_spherical_facet_grid(
    grid_nx: int,
    grid_ny: int,
    pitch_m: float,
    *,
    sphere_center_offset_m: float,
) -> FacetGridInLocalFrame:
    """Flat facet grid in assembly xy; normals point from each center toward (0, 0, sphere_center_offset_m)."""
    nx, ny, half_x, half_y, center_facet = _validate_odd_counts(grid_nx, grid_ny)
    o = np.array([0.0, 0.0, float(sphere_center_offset_m)], dtype=float)

    centers_l: list[np.ndarray] = []
    normals_l: list[np.ndarray] = []
    us_l: list[np.ndarray] = []
    vs_l: list[np.ndarray] = []
    for iy in range(ny):
        for ix in range(nx):
            lx = float(ix - half_x) * float(pitch_m)
            ly = float(iy - half_y) * float(pitch_m)
            p = np.array([lx, ly, 0.0], dtype=float)
            to_sphere = o - p
            ln = float(np.linalg.norm(to_sphere))
            if ln < 1e-15:
                n = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                n = (to_sphere / ln).astype(float)
            u, v = tangent_basis_from_normal(n)
            centers_l.append(p)
            normals_l.append(n)
            us_l.append(u)
            vs_l.append(v)

    centers_local = np.stack(centers_l, axis=0)
    normals_local = np.stack(normals_l, axis=0)
    facet_u_local = np.stack(us_l, axis=0)
    facet_v_local = np.stack(vs_l, axis=0)
    r_identity = np.eye(3, dtype=float)

    return FacetGridInLocalFrame(
        centers_local=centers_local,
        normals_local=normals_local,
        facet_u_local=facet_u_local,
        facet_v_local=facet_v_local,
        local_to_mount_body_rotation=r_identity,
        center_facet_index=center_facet,
    )
