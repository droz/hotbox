"""Tests for shared mirror pointing (bisector + mount-offset refine)."""

from __future__ import annotations

import numpy as np

from hotbox_shared import (
    MirrorGridSpec,
    bisector_normal,
    bisector_normal_at_mount,
    evaluate_center_ray,
    horizontal_stow_angles,
    load_system_constants,
    mount_rotation_matrix,
    normalize,
    pivot_facet_normal_body,
    solve_bisector_tracking,
    solve_bisector_tracking_for_grid,
    solve_tracking,
    solve_tracking_for_grid,
    sun_elevation_deg,
    sun_is_above_horizon,
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


def test_solve_tracking_below_horizon_returns_horizontal_stow() -> None:
    # Sun below horizon: world vector toward sun has negative Z → incoming has positive Z.
    sun_toward_scene = normalize(np.array([0.2, -0.3, 0.9], dtype=float))
    assert not sun_is_above_horizon(sun_toward_scene)
    mount = np.array([0.0, 2.5, 1.0], dtype=float)
    target = np.array([0.0, 0.0, 1.0], dtype=float)
    pivot = pivot_facet_normal_body(grid_nx=3, grid_ny=5, pitch_m=0.26035, radius_of_curvature_m=5.5)

    angles = solve_tracking(
        sun_direction_toward_scene=sun_toward_scene,
        mount_world=mount,
        target_world=target,
        pivot_facet_normal_body=pivot,
        mount_offset_d_m=0.1,
        solve_for_mount_offset=True,
    )
    assert angles.night_stow is True
    assert angles.azimuth_deg == 0.0
    assert angles.elevation_deg == 0.0
    got = normalize(mount_rotation_matrix(angles.azimuth_deg, angles.elevation_deg) @ pivot)
    np.testing.assert_allclose(got, np.array([0.0, 0.0, 1.0]), atol=1e-6)


def test_horizontal_stow_angles_aligns_pivot_to_zenith() -> None:
    pivot = np.array([0.0, 0.0, 1.0], dtype=float)
    mount = np.array([0.0, 2.5, 1.0], dtype=float)
    target = np.array([0.0, 0.0, 1.0], dtype=float)
    angles = horizontal_stow_angles(pivot, mount_world=mount, target_world=target)
    assert angles.night_stow is True
    assert angles.azimuth_deg == 0.0
    assert angles.elevation_deg == 0.0
    got = normalize(mount_rotation_matrix(angles.azimuth_deg, angles.elevation_deg) @ pivot)
    np.testing.assert_allclose(got, np.array([0.0, 0.0, 1.0]), atol=1e-12)


def test_sun_elevation_from_incoming() -> None:
    up = normalize(np.array([0.0, -np.sqrt(0.5), -np.sqrt(0.5)], dtype=float))
    assert abs(sun_elevation_deg(up) - 45.0) < 1e-9
    assert sun_is_above_horizon(up)
    down = normalize(np.array([0.0, -0.5, 0.5], dtype=float))
    assert sun_elevation_deg(down) < 0.0
    assert not sun_is_above_horizon(down)


def test_apply_mount_joint_limits_prefers_in_range_dual() -> None:
    from hotbox_shared import (
        MountJointLimits,
        apply_mount_joint_limits,
        dual_mount_angles,
        oven_facing_azimuth_deg,
        relative_azimuth_deg,
        within_mount_joint_limits,
    )

    mount = np.array([0.0, 2.5, 1.0], dtype=float)
    absorber = np.array([0.0, 0.0, 1.0], dtype=float)
    limits = MountJointLimits()
    oven_az = oven_facing_azimuth_deg(mount, absorber)
    # Primary has negative elevation (outside 0..90); dual should be chosen.
    az, el = 30.0, -40.0
    dual_az, dual_el = dual_mount_angles(az, el)
    assert dual_el > 0.0
    got_az, got_el = apply_mount_joint_limits(
        az, el, mount_world=mount, absorber_world=absorber, limits=limits
    )
    assert within_mount_joint_limits(
        got_az, got_el, oven_facing_azimuth_deg=oven_az, limits=limits
    )
    assert abs(got_el - dual_el) < 1e-9
    assert abs(relative_azimuth_deg(got_az, oven_az)) <= limits.azimuth_max_deg + 1e-6


def test_solve_tracking_respects_joint_limits() -> None:
    """Sequential sun steps stay inside el∈[0,90] and |rel az|≤150 without continuity memory."""
    from hotbox_shared import (
        MountJointLimits,
        oven_facing_azimuth_deg,
        relative_azimuth_deg,
        within_mount_joint_limits,
    )

    mount = np.array([0.0, 2.5, 1.0], dtype=float)
    target = np.array([0.0, 0.0, 1.0], dtype=float)
    pivot = pivot_facet_normal_body(grid_nx=3, grid_ny=5, pitch_m=0.26035, radius_of_curvature_m=5.5)
    limits = MountJointLimits()
    oven_az = oven_facing_azimuth_deg(mount, target)
    elev = 45.0
    prev: tuple[float, float] | None = None
    for az_sun in np.linspace(160.0, 200.0, 21):
        el = np.deg2rad(elev)
        az = np.deg2rad(float(az_sun))
        toward_sun = np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)], dtype=float)
        incoming = -toward_sun
        angles = solve_tracking(
            sun_direction_toward_scene=incoming,
            mount_world=mount,
            target_world=target,
            pivot_facet_normal_body=pivot,
            mount_offset_d_m=0.1,
            solve_for_mount_offset=True,
            joint_limits=limits,
        )
        assert within_mount_joint_limits(
            angles.azimuth_deg,
            angles.elevation_deg,
            oven_facing_azimuth_deg=oven_az,
            limits=limits,
        )
        assert 0.0 <= angles.elevation_deg <= 90.0
        assert abs(relative_azimuth_deg(angles.azimuth_deg, oven_az)) <= 150.0 + 1e-6
        if prev is not None:
            daz = abs(((angles.azimuth_deg - prev[0] + 180.0) % 360.0) - 180.0)
            # Limits (not continuity memory) prevent ~180° dual flips.
            assert daz < 90.0, f"az jumped {daz}° from {prev} to {(angles.azimuth_deg, angles.elevation_deg)}"
        prev = (angles.azimuth_deg, angles.elevation_deg)


def test_system_yaml_loads_joint_limits() -> None:
    system = load_system_constants()
    lim = system.control.mount_joint_limits()
    assert lim.elevation_min_deg == 0.0
    assert lim.elevation_max_deg == 90.0
    assert lim.azimuth_min_deg == -150.0
    assert lim.azimuth_max_deg == 150.0
    assert system.control.safe_park_azimuth_deg == 0.0
    assert system.control.safe_park_elevation_deg == 0.0
    assert system.control.idle_aim_height_above_absorber_m == 2.0
