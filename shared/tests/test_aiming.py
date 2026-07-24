"""Tests for shared mirror pointing (bisector + mount-offset refine)."""

from __future__ import annotations

import numpy as np

from hotbox_shared import (
    MirrorGridSpec,
    bisector_normal,
    bisector_normal_at_mount,
    evaluate_center_ray,
    load_system_constants,
    mount_rotation_matrix,
    normalize,
    pivot_facet_normal_body,
    solve_bisector_tracking,
    solve_bisector_tracking_for_grid,
    solve_tracking,
    solve_tracking_for_grid,
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


def test_evaluate_center_ray_offset_moves_facet() -> None:
    sun = normalize(np.array([0.0, -0.5, -np.sqrt(0.75)], dtype=float))
    mount = np.array([0.0, 2.5, 1.0], dtype=float)
    pivot = np.array([0.0, 0.0, 1.0], dtype=float)
    d = 0.2
    az, el = 30.0, 20.0
    ray = evaluate_center_ray(
        sun_direction_toward_scene=sun,
        mount_world=mount,
        azimuth_deg=az,
        elevation_deg=el,
        mount_offset_d_m=d,
        pivot_facet_normal_body=pivot,
    )
    r = mount_rotation_matrix(az, el)
    want_facet = mount + r @ np.array([0.0, 0.0, d], dtype=float)
    np.testing.assert_allclose(ray.facet_center_world, want_facet, atol=1e-12)
    np.testing.assert_allclose(ray.normal_world, normalize(r @ pivot), atol=1e-12)


def test_solve_tracking_zero_offset_matches_bisector() -> None:
    sun = normalize(np.array([0.1, -0.6, -0.8], dtype=float))
    mount = np.array([0.0, 2.5, 1.0], dtype=float)
    target = np.array([0.0, 0.0, 1.0], dtype=float)
    pivot = pivot_facet_normal_body(grid_nx=3, grid_ny=5, pitch_m=0.26035, radius_of_curvature_m=5.5)
    seed = solve_bisector_tracking(
        sun_direction_toward_scene=sun,
        mount_world=mount,
        target_world=target,
        pivot_facet_normal_body=pivot,
    )
    refined = solve_tracking(
        sun_direction_toward_scene=sun,
        mount_world=mount,
        target_world=target,
        pivot_facet_normal_body=pivot,
        mount_offset_d_m=0.0,
        solve_for_mount_offset=True,
    )
    assert refined.azimuth_deg == seed.azimuth_deg
    assert refined.elevation_deg == seed.elevation_deg


def test_solve_tracking_refine_reduces_center_ray_miss() -> None:
    sun = normalize(np.array([0.15, -0.55, -0.82], dtype=float))
    mount = np.array([0.0, 2.5, 1.0], dtype=float)
    target = np.array([0.0, 0.0, 1.0], dtype=float)
    pivot = pivot_facet_normal_body(grid_nx=3, grid_ny=5, pitch_m=0.26035, radius_of_curvature_m=5.5)
    d = 0.2

    seed = solve_tracking(
        sun_direction_toward_scene=sun,
        mount_world=mount,
        target_world=target,
        pivot_facet_normal_body=pivot,
        mount_offset_d_m=d,
        solve_for_mount_offset=False,
    )
    refined = solve_tracking(
        sun_direction_toward_scene=sun,
        mount_world=mount,
        target_world=target,
        pivot_facet_normal_body=pivot,
        mount_offset_d_m=d,
        solve_for_mount_offset=True,
    )

    def miss(angles) -> float:
        return evaluate_center_ray(
            sun_direction_toward_scene=sun,
            mount_world=mount,
            azimuth_deg=angles.azimuth_deg,
            elevation_deg=angles.elevation_deg,
            mount_offset_d_m=d,
            pivot_facet_normal_body=pivot,
        ).miss_m(target)

    assert miss(refined) < miss(seed)
    assert miss(refined) < 1e-6


def test_solve_tracking_for_grid_respects_refine_flag() -> None:
    sun = normalize(np.array([0.2, -0.4, -0.9], dtype=float))
    mount = np.array([1.0, 2.2, 1.0], dtype=float)
    target = np.array([0.0, 0.0, 1.0], dtype=float)
    grid = MirrorGridSpec(
        grid_nx=3,
        grid_ny=5,
        pitch_m=0.26035,
        radius_of_curvature_m=5.5,
        mount_offset_d_m=0.15,
    )
    seed = solve_tracking_for_grid(
        sun_direction_toward_scene=sun,
        mount_world=mount,
        target_world=target,
        grid=grid,
        solve_for_mount_offset=False,
    )
    refined = solve_tracking_for_grid(
        sun_direction_toward_scene=sun,
        mount_world=mount,
        target_world=target,
        grid=grid,
        solve_for_mount_offset=True,
    )
    assert (seed.azimuth_deg, seed.elevation_deg) != (refined.azimuth_deg, refined.elevation_deg)


def test_system_yaml_loads_solve_for_mount_offset() -> None:
    system = load_system_constants()
    assert system.control.solve_for_mount_offset is True
