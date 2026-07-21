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
    for _ in range(100):
        nodes[0].step(0.05)

    status = transport.poll_status(0)
    assert status.homed is True


def test_sitl_harness_runs() -> None:
    harness = SitlHarness(node_ids=(0,))
    harness.startup()
    snapshot = harness.step(0.05)
    assert "0" in snapshot["mirrors"] or 0 in snapshot["mirrors"]
    assert snapshot["geometry"]["target"] is not None
    assert snapshot["geometry"]["estimated"] is not None
    assert snapshot["geometry"]["true"] is not None
    assert "0" in snapshot["true_miss_m"]
