from __future__ import annotations

import math

from hotbox_shared import load_system_constants
from hotbox_shared.firmware_header import render_firmware_header


def test_load_system_constants() -> None:
    system = load_system_constants()
    assert system.absorber.center_height_m == 1.0
    assert system.mirror.grid_nx == 3
    assert system.mirror.grid_ny == 5
    assert system.fleet.assembly_count == 3
    assert len(system.fleet.mounts) == 3
    # Node 0 at bearing 0 sits on the absorber normal (default 90° → +Y).
    mount0 = system.mount_world(0)
    assert abs(mount0[0] - 0.0) < 1e-9
    assert abs(mount0[1] - 2.5) < 1e-9
    assert abs(mount0[2] - 1.0) < 1e-9
    assert system.fleet.mount_by_id(1).bearing_deg == -30.0
    assert system.fleet.mount_by_id(2).bearing_deg == 30.0


def test_mount_bearing_is_relative_to_absorber_normal() -> None:
    system = load_system_constants()
    mount = system.fleet.mount_by_id(0)
    # Rotating the oven normal by +20° rotates a bearing-0 mount by the same amount.
    a = mount.mount_world(normal_angle_from_x_deg=90.0)
    b = mount.mount_world(normal_angle_from_x_deg=110.0)
    assert abs(a[0] - 0.0) < 1e-9
    assert abs(a[1] - mount.oa_distance_m) < 1e-9
    assert abs(b[0] - mount.oa_distance_m * math.cos(math.radians(110.0))) < 1e-9
    assert abs(b[1] - mount.oa_distance_m * math.sin(math.radians(110.0))) < 1e-9


def test_firmware_header_contains_key_defines() -> None:
    system = load_system_constants()
    header = render_firmware_header(system)
    assert "HOTBOX_ABSORBER_CENTER_HEIGHT_M" in header
    assert "HOTBOX_MIRROR_GRID_NX" in header
    assert "HOTBOX_MIRROR_OFFSET_D_M" in header


def test_firmware_header_contains_key_defines() -> None:
    system = load_system_constants()
    header = render_firmware_header(system)
    assert "HOTBOX_ABSORBER_CENTER_HEIGHT_M" in header
    assert "HOTBOX_MIRROR_GRID_NX" in header
    assert "HOTBOX_MIRROR_OFFSET_D_M" in header
