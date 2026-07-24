from __future__ import annotations

import numpy as np

from hotbox_controller.geometry import (
    mirror_normal_for_reflection,
    mount_rotation_matrix,
    normalize,
    pivot_facet_normal_body,
)
from hotbox_controller.scene import build_mirror_scene_entry, reflect_ray
from hotbox_controller.sun import SunVector
from hotbox_controller.tracking import track_absorber


def test_track_absorber_aligns_pivot_facet_normal_to_bisector() -> None:
    sun = SunVector(
        azimuth_deg=180.0,
        elevation_deg=45.0,
        world_vector=np.array([0.0, -np.sqrt(0.5), np.sqrt(0.5)], dtype=float),
    )
    mount = np.array([0.0, 2.5, 1.0], dtype=float)
    absorber = np.array([0.0, 0.0, 1.0], dtype=float)
    target = track_absorber(sun, mount, absorber)

    pivot = pivot_facet_normal_body(grid_nx=3, grid_ny=5, pitch_m=0.26035, radius_of_curvature_m=5.5)
    incoming = -normalize(sun.world_vector)
    outgoing = normalize(absorber - mount)
    bisector = mirror_normal_for_reflection(incoming, outgoing)
    got = normalize(mount_rotation_matrix(target.azimuth_deg, target.elevation_deg) @ pivot)
    np.testing.assert_allclose(got, bisector, atol=1e-6)
    assert target.mode == "tracking"


def test_track_absorber_below_horizon_parks_horizontal() -> None:
    sun = SunVector(
        azimuth_deg=180.0,
        elevation_deg=-20.0,
        world_vector=normalize(
            np.array([0.0, -np.cos(np.deg2rad(20.0)), -np.sin(np.deg2rad(20.0))], dtype=float)
        ),
    )
    # world_vector points toward sun; below horizon ⇒ negative Z
    assert sun.world_vector[2] < 0.0
    target = track_absorber(sun, np.array([0.0, 2.5, 1.0]), np.array([0.0, 0.0, 1.0]))
    assert target.mode == "parked"
    pivot = pivot_facet_normal_body(grid_nx=3, grid_ny=5, pitch_m=0.26035, radius_of_curvature_m=5.5)
    got = normalize(mount_rotation_matrix(target.azimuth_deg, target.elevation_deg) @ pivot)
    np.testing.assert_allclose(got, np.array([0.0, 0.0, 1.0]), atol=1e-6)


def test_build_mirror_scene_entry_incoming_ray_points_toward_facet() -> None:
    sun = SunVector(
        azimuth_deg=180.0,
        elevation_deg=45.0,
        world_vector=np.array([0.0, -np.sqrt(0.5), np.sqrt(0.5)], dtype=float),
    )
    mount = np.array([0.0, 2.5, 1.0], dtype=float)
    target = track_absorber(sun, mount, np.array([0.0, 0.0, 1.0]))
    mirror = build_mirror_scene_entry(
        node_id=0,
        mount_world=mount,
        azimuth_deg=target.azimuth_deg,
        elevation_deg=target.elevation_deg,
        mirror_offset_d_m=0.2,
        sun=sun,
        absorber_world=np.array([0.0, 0.0, 1.0]),
    )
    start = np.asarray(mirror["incoming"]["start"], dtype=float)
    end = np.asarray(mirror["incoming"]["end"], dtype=float)
    incoming_dir = normalize(end - start)
    sun_toward_scene = -normalize(sun.world_vector)
    np.testing.assert_allclose(incoming_dir, sun_toward_scene, atol=1e-6)

    normal = np.asarray(mirror["normal"], dtype=float)
    reflected = reflect_ray(sun_toward_scene, normal)
    reflected_seg = normalize(
        np.asarray(mirror["reflected"]["end"], dtype=float) - np.asarray(mirror["reflected"]["start"], dtype=float)
    )
    np.testing.assert_allclose(reflected_seg, reflected, atol=1e-6)
