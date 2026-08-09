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
    assert "HOTBOX_PIN_ELEVATION_ENC_A" in header
    assert system.pins.elevation_enc_a in header
    assert f"#define HOTBOX_PIN_ELEVATION_ENC_A {system.pins.elevation_enc_a}" in header
    assert f"#define HOTBOX_MOTOR_PWM_HZ ({system.pins.motor_pwm_hz})" in header


def test_actuator_two_stage_gear_ratio() -> None:
    system = load_system_constants()
    assert system.actuator.motor_gear_ratio == 70.0
    assert system.actuator.worm_gear_ratio == 120.0
    assert abs(system.actuator.gear_ratio - 70.0 * 120.0) < 1e-9
    expected_tpd = (
        system.actuator.encoder_ppr * 4.0 * system.actuator.motor_gear_ratio * system.actuator.worm_gear_ratio / 360.0
    )
    assert abs(system.actuator.ticks_per_degree - expected_tpd) < 1e-6
    assert system.actuator.pwm_deadband == 0.05
    header = render_firmware_header(system)
    assert "HOTBOX_MOTOR_GEAR_RATIO (70.000000f)" in header
    assert "HOTBOX_WORM_GEAR_RATIO (120.000000f)" in header
    assert "HOTBOX_PWM_DEADBAND (0.050000f)" in header


def test_pins_reject_invalid_label() -> None:
    from hotbox_shared.system import PinConstants
    import pytest

    with pytest.raises(ValueError, match="elevation_enc_a"):
        PinConstants(elevation_enc_a="GPIO8")
