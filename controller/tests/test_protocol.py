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
    status = MirrorStatus(node_id=1, homed=True, azimuth_deg=10.5, elevation_deg=20.25, mode="tracking")
    restored = MirrorStatus.from_wire(status.to_wire().strip())
    assert restored.node_id == 1
    assert restored.homed is True
    assert abs(restored.azimuth_deg - 10.5) < 1e-9
    assert restored.mode == "tracking"
