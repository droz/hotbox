from __future__ import annotations

from hotbox_controller.app import ControllerApplication
from hotbox_controller.protocol import CommandName, MirrorCommand, MirrorStatus
from hotbox_controller.transport import DiscoveredNode, MirrorTransport


class FakeTransport(MirrorTransport):
    def __init__(self, node_ids: list[int] | None = None) -> None:
        self.node_ids = node_ids or [0]
        self.sent: list[MirrorCommand] = []

    def discover(self):
        for node_id in self.node_ids:
            yield DiscoveredNode(node_id=node_id, transport_name="fake", endpoint=f"fake://{node_id}")

    def send(self, command: MirrorCommand) -> None:
        self.sent.append(command)

    def poll_status(self, node_id: int) -> MirrorStatus:
        return MirrorStatus(node_id=node_id, homed=True, azimuth_deg=10.0, elevation_deg=20.0, mode="idle")


def _app() -> tuple[ControllerApplication, FakeTransport]:
    transport = FakeTransport([0, 1])
    app = ControllerApplication(transport=transport)
    app.fleet.discover()
    return app, transport


def test_supervisor_modes_track_park_manual_round_trip() -> None:
    app, transport = _app()
    assert app.mode == "track"

    app.set_mode("manual")
    assert app.mode == "manual"
    before = len(transport.sent)
    app.control_tick()
    assert len(transport.sent) == before

    app.set_mode("park")
    assert app.mode == "park"
    assert any(c.command == CommandName.SET_TARGET for c in transport.sent)

    app.set_mode("auto")
    assert app.mode == "track"


def test_set_mode_rejects_unknown() -> None:
    app, _transport = _app()
    try:
        app.set_mode("fly")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "unsupported" in str(exc)
