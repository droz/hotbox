from __future__ import annotations

from hotbox_controller.protocol import CommandName, MirrorCommand, MirrorStatus


def test_set_target_can_roundtrip_with_hold_current_flag() -> None:
    command = MirrorCommand(
        node_id=0,
        command=CommandName.SET_TARGET,
        payload={"hold_current": True},
    )
    frame = command.to_can_frame()
    restored = MirrorCommand.from_can_frame(0, frame)
    assert restored.command == CommandName.SET_TARGET
    assert restored.payload["hold_current"] is True


def test_set_velocity_can_roundtrip() -> None:
    command = MirrorCommand(
        node_id=0,
        command=CommandName.SET_VELOCITY,
        payload={"azimuth_deg_s": -3.5, "elevation_deg_s": 1.25},
    )
    frame = command.to_can_frame()
    restored = MirrorCommand.from_can_frame(0, frame)
    assert restored.command == CommandName.SET_VELOCITY
    assert abs(restored.payload["azimuth_deg_s"] - (-3.5)) < 0.02
    assert abs(restored.payload["elevation_deg_s"] - 1.25) < 0.02


def test_mirror_status_wire_roundtrip() -> None:
    status = MirrorStatus(
        node_id=1,
        azimuth_home="homed",
        elevation_home="homed",
        azimuth_deg=10.5,
        elevation_deg=20.25,
        target_azimuth_deg=12.0,
        target_elevation_deg=25.5,
        azimuth_integral=0.1,
        elevation_integral=-0.2,
        pid_kp=1.2,
        pid_ki=0.5,
        pid_kd=0.01,
        pid_velocity_kp=1.0,
        pid_velocity_ki=0.0,
        pid_velocity_kd=0.0,
        mode="position",
    )
    restored = MirrorStatus.from_wire(status.to_wire().strip())
    assert restored.node_id == 1
    assert restored.azimuth_home == "homed"
    assert restored.elevation_home == "homed"
    assert restored.homed is True
    assert abs(restored.azimuth_deg - 10.5) < 1e-9
    assert abs(restored.target_azimuth_deg - 12.0) < 1e-9
    assert abs(restored.target_elevation_deg - 25.5) < 1e-9
    assert abs(restored.azimuth_integral - 0.1) < 1e-9
    assert abs(restored.elevation_integral - (-0.2)) < 1e-9
    assert restored.pid_kp == 1.2
    assert restored.pid_velocity_kp == 1.0
    assert restored.mode == "position"


def test_mirror_status_wire_without_targets_is_backward_compatible() -> None:
    wire = (
        b'{"node_id":0,"type":"status","azimuth_home":"homed","elevation_home":"homed",'
        b'"azimuth_deg":1.0,"elevation_deg":2.0,"mode":"idle"}'
    )
    status = MirrorStatus.from_wire(wire)
    assert status.target_azimuth_deg is None
    assert status.target_elevation_deg is None
    assert "target_azimuth_deg" not in status.as_dict()


def test_mirror_status_hall_width_roundtrip() -> None:
    status = MirrorStatus(
        node_id=0,
        azimuth_home="homed",
        elevation_home="homed",
        az_hall_width_deg=1.25,
        el_hall_width_deg=2.5,
        mode="position",
    )
    restored = MirrorStatus.from_wire(status.to_wire().strip())
    assert restored.az_hall_width_deg == 1.25
    assert restored.el_hall_width_deg == 2.5
    assert restored.as_dict()["az_hall_width_deg"] == 1.25


def test_mirror_status_legacy_homed_bool() -> None:
    wire = b'{"node_id":0,"type":"status","homed":true,"mode":"idle"}\n'
    status = MirrorStatus.from_wire(wire)
    assert status.azimuth_home == "homed"
    assert status.elevation_home == "homed"
    assert status.homed is True


def test_mirror_status_partial_home() -> None:
    status = MirrorStatus(node_id=0, azimuth_home="homed", elevation_home="homing", mode="homing")
    assert status.homed is False
    assert status.as_dict()["azimuth_home"] == "homed"
    assert status.as_dict()["elevation_home"] == "homing"


def test_set_pid_pos_can_roundtrip() -> None:
    command = MirrorCommand(
        node_id=0,
        command=CommandName.SET_PID_POS,
        payload={"kp": 1.25, "ki": 0.5, "kd": 0.01},
    )
    frame = command.to_can_frame()
    restored = MirrorCommand.from_can_frame(0, frame)
    assert restored.command == CommandName.SET_PID_POS
    assert abs(restored.payload["kp"] - 1.25) < 0.002
    assert abs(restored.payload["ki"] - 0.5) < 0.002


def test_set_pid_vel_can_roundtrip() -> None:
    command = MirrorCommand(
        node_id=0,
        command=CommandName.SET_PID_VEL,
        payload={"kp": 1.0, "ki": 0.0, "kd": 0.0},
    )
    frame = command.to_can_frame()
    restored = MirrorCommand.from_can_frame(0, frame)
    assert restored.command == CommandName.SET_PID_VEL
    assert abs(restored.payload["kp"] - 1.0) < 0.002
    assert abs(restored.payload["ki"] - 0.0) < 0.002


def test_mirror_command_usb_wire_is_compact() -> None:
    """Firmware string-matches commands; spaces after ':' previously caused timeouts."""
    wire = MirrorCommand(node_id=0, command=CommandName.GET_STATUS).to_wire().decode()
    assert '"command":"get_status"' in wire
    assert '"command": "get_status"' not in wire
