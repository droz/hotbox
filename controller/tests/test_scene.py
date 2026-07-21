from __future__ import annotations

import numpy as np

from hotbox_controller.protocol import MirrorStatus
from hotbox_controller.scene import build_target_scene
from hotbox_controller.sun import SunVector
from hotbox_controller.tracking import TrackingTarget, track_absorber


def test_build_target_scene_uses_commanded_targets_not_live_status() -> None:
    sun = SunVector(
        azimuth_deg=180.0,
        elevation_deg=45.0,
        world_vector=np.array([0.0, -np.sqrt(0.5), np.sqrt(0.5)], dtype=float),
    )
    mount = np.array([0.0, 2.5, 1.0], dtype=float)
    absorber = np.array([0.0, 0.0, 1.0], dtype=float)
    target = track_absorber(sun, mount, absorber)

    scene_a = build_target_scene(
        sun=sun,
        absorber_world=absorber,
        targets={0: target},
        calibrations={},
        absorber_height_m=1.0,
    )
    scene_b = build_target_scene(
        sun=sun,
        absorber_world=absorber,
        targets={0: target},
        calibrations={},
        absorber_height_m=1.0,
    )
    assert scene_a["label"] == "target"
    assert scene_a["mirrors"][0]["azimuth_deg"] == target.azimuth_deg
    assert scene_a["mirrors"][0]["elevation_deg"] == target.elevation_deg
    assert scene_a["mirrors"][0]["azimuth_deg"] == scene_b["mirrors"][0]["azimuth_deg"]

    # Wildly different live status must not move the target overlay.
    jitter_status = MirrorStatus(node_id=0, homed=True, azimuth_deg=17.0, elevation_deg=83.0, mode="tracking")
    _ = jitter_status
    scene_jitter = build_target_scene(
        sun=sun,
        absorber_world=absorber,
        targets={0: target},
        calibrations={},
        absorber_height_m=1.0,
    )
    assert scene_jitter["mirrors"][0]["azimuth_deg"] == target.azimuth_deg


def test_build_target_scene_includes_rays() -> None:
    sun = SunVector(
        azimuth_deg=180.0,
        elevation_deg=45.0,
        world_vector=np.array([0.0, -np.sqrt(0.5), np.sqrt(0.5)], dtype=float),
    )
    target = TrackingTarget(azimuth_deg=30.0, elevation_deg=20.0, mode="tracking")
    scene = build_target_scene(
        sun=sun,
        absorber_world=np.array([0.0, 0.0, 1.0]),
        targets={0: target},
        calibrations={},
        absorber_height_m=1.0,
        default_oa_distance_m=2.5,
        default_mirror_offset_d_m=0.2,
    )
    assert scene["mirrors"][0]["node_id"] == 0
    assert len(scene["mirrors"][0]["incoming"]["start"]) == 3
    assert len(scene["mirrors"][0]["reflected"]["end"]) == 3
    assert len(scene["mirrors"][0]["facets"]) == 15
    assert scene["sun"]["display_position"][2] > 0.5
    assert scene["oven"]["body_center"] is not None
