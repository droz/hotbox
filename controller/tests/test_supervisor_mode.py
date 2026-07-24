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
    app.startup()
    return app, transport


def test_fleet_and_mirror_modes() -> None:
    app, transport = _app()
    assert app.mode == "track"
    assert app.node_mode(0) == "track"
    assert app.heat_demand is True

    app.set_mode("jog")
    assert app.mode == "jog"
    assert app.node_mode(1) == "jog"
    before = len(transport.sent)
    app.control_tick()
    # Jog mirrors receive no closed-loop SET_TARGET.
    assert not any(c.command == CommandName.SET_TARGET for c in transport.sent[before:])

    app.set_mirror_mode(0, "park")
    assert app.mode == "mixed"
    assert app.node_mode(0) == "park"
    assert app.node_mode(1) == "jog"

    app.set_mode("track")
    assert app.mode == "track"


def test_heat_demand_diverts_above_absorber() -> None:
    app, transport = _app()
    app.set_mode("track")
    assert app.config.oven.idle_aim_height_above_absorber_m == 2.0

    app.set_heat_demand(False)
    transport.sent.clear()
    app.control_tick()
    diverted = [c for c in transport.sent if c.command == CommandName.SET_TARGET]
    assert diverted
    divert_pose = (diverted[0].payload["azimuth_deg"], diverted[0].payload["elevation_deg"])
    divert_mode = diverted[0].payload.get("mode")

    transport.sent.clear()
    app.set_heat_demand(True)
    app.control_tick()
    on_absorber = [c for c in transport.sent if c.command == CommandName.SET_TARGET]
    assert on_absorber
    absorber_pose = (on_absorber[0].payload["azimuth_deg"], on_absorber[0].payload["elevation_deg"])
    absorber_mode = on_absorber[0].payload.get("mode")

    if absorber_mode == "parked":
        # Sun below horizon: both paths face-up stow.
        assert divert_mode == "parked"
        assert divert_pose == (0.0, 0.0)
        assert absorber_pose == (0.0, 0.0)
    else:
        assert divert_mode == "tracking"
        assert absorber_mode == "tracking"
        assert divert_pose != absorber_pose


def test_park_is_face_up_identity() -> None:
    app, transport = _app()
    transport.sent.clear()
    app.set_mode("park")
    parks = [c for c in transport.sent if c.command == CommandName.SET_TARGET]
    assert parks
    assert all(c.payload.get("azimuth_deg") == 0.0 for c in parks)
    assert all(c.payload.get("elevation_deg") == 0.0 for c in parks)
    assert all(c.payload.get("mode") == "parked" for c in parks)


def test_zero_rate_jog_does_not_override_track_mode() -> None:
    """UI stick-release posts jog@0; that must not clobber a Track/Park switch."""
    from hotbox_controller.app import JogRequest

    app, _transport = _app()
    app.set_mode("jog")
    assert app.mode == "jog"

    app.set_mode("track")
    assert app.mode == "track"
    app.jog(JogRequest(node_id=0, azimuth_rate_deg_s=0.0, elevation_rate_deg_s=0.0))
    assert app.node_mode(0) == "track"
    assert app.mode == "track"

    app.jog(JogRequest(node_id=0, azimuth_rate_deg_s=2.0, elevation_rate_deg_s=0.0))
    assert app.node_mode(0) == "jog"


def test_set_mode_rejects_unknown() -> None:
    app, _transport = _app()
    try:
        app.set_mode("fly")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "unsupported" in str(exc)
