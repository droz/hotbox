from __future__ import annotations

from hotbox_sitl.harness import SitlHarness
from hotbox_sitl.mirror_node import SimulatedMirrorNode


def test_sim_transport_home_and_status() -> None:
    from hotbox_controller.protocol import CommandName, MirrorCommand
    from hotbox_controller.transport import SimTransport

    nodes = {0: SimulatedMirrorNode(node_id=0)}
    transport = SimTransport(nodes)

    discovered = list(transport.discover())
    assert len(discovered) == 1

    transport.send(MirrorCommand(node_id=0, command=CommandName.HOME))
    for _ in range(300):
        nodes[0].step(0.05)

    status = transport.poll_status(0)
    assert status.azimuth_home == "homed"
    assert status.elevation_home == "homed"
    assert status.homed is True


def test_sim_transport_home_single_axis() -> None:
    from hotbox_controller.protocol import CommandName, MirrorCommand
    from hotbox_controller.transport import SimTransport

    nodes = {0: SimulatedMirrorNode(node_id=0)}
    transport = SimTransport(nodes)

    transport.send(MirrorCommand(node_id=0, command=CommandName.HOME, payload={"axis": "az"}))
    for _ in range(300):
        nodes[0].step(0.05)

    status = transport.poll_status(0)
    assert status.azimuth_home == "homed"
    assert status.elevation_home == "unhomed"
    assert status.homed is False


def test_firmware_cil_home_and_status() -> None:
    from hotbox_controller.protocol import CommandName, MirrorCommand
    from hotbox_controller.transport import SimTransport
    from hotbox_shared import load_system_constants
    from hotbox_sitl.firmware_axis import FirmwareMirrorNode

    system = load_system_constants()
    node = FirmwareMirrorNode.from_constants(
        0,
        system.actuator,
        oven_facing_azimuth_deg=180.0,
    )
    transport = SimTransport({0: node})
    transport.send(MirrorCommand(node_id=0, command=CommandName.HOME))
    for _ in range(800):
        node.step(0.02)
        status = transport.poll_status(0)
        if status.homed and status.fault is None:
            break

    status = transport.poll_status(0)
    assert status.fault is None, status.fault
    assert status.azimuth_home == "homed"
    assert status.elevation_home == "homed"
    assert status.homed is True
    assert abs(status.azimuth_deg - 180.0) < 5.0
    assert abs(status.elevation_deg - 90.0) < 5.0


def test_sitl_harness_runs() -> None:
    harness = SitlHarness(node_ids=(0,))
    harness.startup()
    snapshot = harness.step(0.05)
    assert "0" in snapshot["mirrors"] or 0 in snapshot["mirrors"]
    assert snapshot["geometry"]["target"] is not None
    assert snapshot["geometry"]["estimated"] is not None
    assert snapshot["geometry"]["true"] is not None
    assert "0" in snapshot["true_miss_m"]
