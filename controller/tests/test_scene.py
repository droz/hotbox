from __future__ import annotations

import numpy as np

from hotbox_controller.protocol import MirrorStatus
from hotbox_controller.scene import build_estimated_scene
from hotbox_controller.sun import SunVector


def test_build_estimated_scene_includes_rays() -> None:
    sun = SunVector(
        azimuth_deg=180.0,
        elevation_deg=45.0,
        world_vector=np.array([0.0, -np.sqrt(0.5), np.sqrt(0.5)], dtype=float),
    )
    statuses = {0: MirrorStatus(node_id=0, homed=True, azimuth_deg=30.0, elevation_deg=20.0, mode="tracking")}
    scene = build_estimated_scene(
        sun=sun,
        absorber_world=np.array([0.0, 0.0, 1.0]),
        statuses=statuses,
        calibrations={},
        absorber_height_m=1.0,
    )
    assert scene["mirrors"][0]["node_id"] == 0
    assert len(scene["mirrors"][0]["incoming"]["start"]) == 3
    assert len(scene["mirrors"][0]["reflected"]["end"]) == 3
