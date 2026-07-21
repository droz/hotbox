"""Tests that full raytrace plant layout follows config/system.yaml."""

from __future__ import annotations

import numpy as np

from hotbox_shared import load_system_constants

from src.main import build_default_simulation


def test_build_default_simulation_uses_shared_fleet_mounts() -> None:
    system = load_system_constants()
    sim = build_default_simulation(system=system)

    assert len(sim.mirrors) == system.fleet.assembly_count
    assert sim.sun.latitude_deg == system.default_site.latitude_deg
    assert sim.sun.longitude_deg == system.default_site.longitude_deg
    assert sim.absorber.width_m == system.absorber.width_m
    assert sim.absorber.height_m == system.absorber.height_m

    for mirror, mount in zip(sim.mirrors, system.fleet.mounts, strict=True):
        np.testing.assert_allclose(mirror.mount_world, system.mount_world(mount.node_id), atol=1e-12)
        assert mirror.grid_nx == system.mirror.grid_nx
        assert mirror.grid_ny == system.mirror.grid_ny
        assert mirror.pitch_m == system.mirror.pitch_m
        assert mirror.sphere_center_offset_m == system.mirror.radius_of_curvature_m
        assert mirror.mount_offset_d_m == system.mirror.mount_offset_d_m
