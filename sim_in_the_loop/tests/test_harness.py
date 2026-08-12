from __future__ import annotations

from hotbox_controller.protocol import CommandName, MirrorCommand
from hotbox_controller.transport import SimTransport
from hotbox_shared import load_system_constants
from hotbox_sitl.harness import SitlHarness
from hotbox_sitl.mirror_node import MirrorNode


def test_cil_home_and_status() -> None:
    system = load_system_constants()
    node = MirrorNode.from_constants(0, system.actuator)
    transport = SimTransport({0: node})

    discovered = list(transport.discover())
    assert len(discovered) == 1

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
    # Wire/firmware joint frame: home_azimuth_deg / home_elevation_deg.
    assert abs(status.azimuth_deg - system.actuator.home_azimuth_deg) < 5.0
    assert abs(status.elevation_deg - system.actuator.home_elevation_deg) < 5.0
    assert status.az_hall_width_deg is not None
    assert status.el_hall_width_deg is not None
    assert abs(float(status.az_hall_width_deg) - 8.0) < 1.5
    assert abs(float(status.el_hall_width_deg) - 8.0) < 1.5


def test_cil_home_single_axis() -> None:
    system = load_system_constants()
    node = MirrorNode.from_constants(0, system.actuator)
    transport = SimTransport({0: node})

    transport.send(MirrorCommand(node_id=0, command=CommandName.HOME, payload={"axis": "az"}))
    for _ in range(800):
        node.step(0.02)
        status = transport.poll_status(0)
        if status.azimuth_home == "homed":
            break

    status = transport.poll_status(0)
    assert status.fault is None, status.fault
    assert status.azimuth_home == "homed"
    assert status.elevation_home == "unhomed"
    assert status.homed is False


def test_cil_multi_node_independent() -> None:
    """Each node loads its own CIL library (separate HAL)."""
    system = load_system_constants()
    nodes = {
        0: MirrorNode.from_constants(0, system.actuator),
        1: MirrorNode.from_constants(1, system.actuator),
    }
    transport = SimTransport(nodes)
    transport.send(MirrorCommand(node_id=0, command=CommandName.HOME, payload={"axis": "az"}))
    for _ in range(800):
        nodes[0].step(0.02)
        nodes[1].step(0.02)
        if transport.poll_status(0).azimuth_home == "homed":
            break

    assert transport.poll_status(0).azimuth_home == "homed"
    assert transport.poll_status(1).azimuth_home == "unhomed"


def test_sitl_harness_runs() -> None:
    harness = SitlHarness(node_ids=(0,))
    harness.startup()
    snapshot = harness.step(0.05)
    assert "0" in snapshot["mirrors"] or 0 in snapshot["mirrors"]
    assert snapshot["geometry"]["target"] is not None
    assert snapshot["geometry"]["estimated"] is not None
    assert snapshot["geometry"]["true"] is not None
    assert "0" in snapshot["true_miss_m"]
