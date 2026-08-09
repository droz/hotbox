from __future__ import annotations

from hotbox_controller.protocol import CommandName, MirrorCommand, MirrorStatus


def test_mirror_command_can_roundtrip() -> None:
    command = MirrorCommand(
        node_id=2,
        command=CommandName.SET_TARGET,
        payload={"azimuth_deg": 45.25, "elevation_deg": 12.75},
    )
    frame = command.to_can_frame()
    restored = MirrorCommand.from_can_frame(2, frame)
    assert restored.node_id == 2
    assert restored.command == CommandName.SET_TARGET
    assert abs(restored.payload["azimuth_deg"] - 45.25) < 0.02
    assert abs(restored.payload["elevation_deg"] - 12.75) < 0.02


def test_mirror_status_wire_roundtrip() -> None:
    status = MirrorStatus(
        node_id=1,
        homed=True,
        azimuth_deg=10.5,
        elevation_deg=20.25,
        azimuth_integral=0.1,
        elevation_integral=-0.2,
        pid_kp=1.2,
        pid_ki=0.5,
        pid_kd=0.01,
        mode="position",
    )
    restored = MirrorStatus.from_wire(status.to_wire().strip())
    assert restored.node_id == 1
    assert restored.homed is True
    assert abs(restored.azimuth_deg - 10.5) < 1e-9
    assert abs(restored.azimuth_integral - 0.1) < 1e-9
    assert abs(restored.elevation_integral - (-0.2)) < 1e-9
    assert restored.pid_kp == 1.2
    assert restored.mode == "position"


def test_set_pid_can_roundtrip() -> None:
    command = MirrorCommand(
        node_id=0,
        command=CommandName.SET_PID,
        payload={"kp": 1.25, "ki": 0.5, "kd": 0.01},
    )
    frame = command.to_can_frame()
    restored = MirrorCommand.from_can_frame(0, frame)
    assert restored.command == CommandName.SET_PID
    assert abs(restored.payload["kp"] - 1.25) < 0.002
    assert abs(restored.payload["ki"] - 0.5) < 0.002


def test_mirror_command_usb_wire_is_compact() -> None:
    """Firmware string-matches commands; spaces after ':' previously caused timeouts."""
    wire = MirrorCommand(node_id=0, command=CommandName.GET_STATUS).to_wire().decode()
    assert '"command":"get_status"' in wire
    assert '"command": "get_status"' not in wire
