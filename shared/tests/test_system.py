from __future__ import annotations

from hotbox_shared import load_system_constants
from hotbox_shared.firmware_header import render_firmware_header


def test_load_system_constants() -> None:
    system = load_system_constants()
    assert system.absorber.center_height_m == 1.0
    assert system.mirror.grid_nx == 3
    assert system.mirror.grid_ny == 5
    assert system.fleet.assembly_count == 3
    assert len(system.fleet.mounts) == 3
    mount0 = system.fleet.mount_by_id(0).mount_world()
    assert abs(mount0[0] - 2.5) < 1e-9
    assert abs(mount0[1] - 0.0) < 1e-9
    assert abs(mount0[2] - 1.0) < 1e-9


def test_firmware_header_contains_key_defines() -> None:
    system = load_system_constants()
    header = render_firmware_header(system)
    assert "HOTBOX_ABSORBER_CENTER_HEIGHT_M" in header
    assert "HOTBOX_MIRROR_GRID_NX" in header
    assert "HOTBOX_MIRROR_OFFSET_D_M" in header
