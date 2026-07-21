"""Tests for shared mirror pointing (bisector tracking)."""

from __future__ import annotations

import numpy as np

from hotbox_shared import (
    MirrorGridSpec,
    bisector_normal,
    bisector_normal_at_mount,
    mount_rotation_matrix,
    normalize,
    pivot_facet_normal_body,
    solve_bisector_tracking,
    solve_bisector_tracking_for_grid,
)


def test_bisector_normal_reflects_incoming_toward_outgoing() -> None:
    incoming = normalize(np.array([0.3, -0.2, -0.9], dtype=float))
    outgoing = normalize(np.array([0.1, 0.4, 0.5], dtype=float))
    n = bisector_normal(incoming, outgoing)
    reflected = incoming - 2.0 * float(np.dot(incoming, n)) * n
    np.testing.assert_allclose(reflected, outgoing, atol=1e-9)


def test_bisector_normal_at_mount_matches_manual() -> None:
    d_sun = normalize(np.array([0.3, -0.2, -0.9], dtype=float))
    mount = np.array([1.0, 0.5, 0.3], dtype=float)
    target = np.array([0.0, 0.0, 1.2], dtype=float)
    n = bisector_normal_at_mount(d_sun, mount, target)
    outgoing = normalize(target - mount)
    reflected = d_sun - 2.0 * float(np.dot(d_sun, n)) * n
    np.testing.assert_allclose(reflected, outgoing, atol=1e-9)


def test_solve_bisector_tracking_aligns_pivot_facet() -> None:
    sun_toward_scene = normalize(np.array([0.0, -np.sqrt(0.5), -np.sqrt(0.5)], dtype=float))
    mount = np.array([0.0, 2.5, 1.0], dtype=float)
    target = np.array([0.0, 0.0, 1.0], dtype=float)
    pivot = pivot_facet_normal_body(grid_nx=3, grid_ny=5, pitch_m=0.26035, radius_of_curvature_m=5.5)

    angles = solve_bisector_tracking(
        sun_direction_toward_scene=sun_toward_scene,
        mount_world=mount,
        target_world=target,
        pivot_facet_normal_body=pivot,
    )
    incoming = sun_toward_scene
    outgoing = normalize(target - mount)
    bisector = bisector_normal(incoming, outgoing)
    got = normalize(mount_rotation_matrix(angles.azimuth_deg, angles.elevation_deg) @ pivot)
    np.testing.assert_allclose(got, bisector, atol=1e-6)


def test_solve_bisector_tracking_for_grid_matches_explicit_pivot() -> None:
    sun_toward_scene = normalize(np.array([0.2, -0.1, -0.95], dtype=float))
    mount = np.array([1.0, 2.0, 1.0], dtype=float)
    target = np.array([0.0, 0.0, 1.0], dtype=float)
    grid = MirrorGridSpec(grid_nx=3, grid_ny=5, pitch_m=0.26035, radius_of_curvature_m=5.5)

    explicit = solve_bisector_tracking(
        sun_direction_toward_scene=sun_toward_scene,
        mount_world=mount,
        target_world=target,
        pivot_facet_normal_body=grid.pivot_normal_body(),
    )
    from_grid = solve_bisector_tracking_for_grid(
        sun_direction_toward_scene=sun_toward_scene,
        mount_world=mount,
        target_world=target,
        grid=grid,
    )
    assert explicit.azimuth_deg == from_grid.azimuth_deg
    assert explicit.elevation_deg == from_grid.elevation_deg
