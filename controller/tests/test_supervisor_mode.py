from __future__ import annotations

from hotbox_controller.app import ControllerApplication
from hotbox_controller.protocol import CommandName, MirrorCommand, MirrorStatus
from hotbox_controller.transport import DiscoveredNode, MirrorTransport


class FakeTransport(MirrorTransport):
    def __init__(self, node_ids: list[int] | None = None) -> None:
        self.node_ids = node_ids or [0]
        self.sent: list[MirrorCommand] = []
        self.polls: list[int] = []

    def discover(self):
        for node_id in self.node_ids:
            yield DiscoveredNode(node_id=node_id, transport_name="fake", endpoint=f"fake://{node_id}")

    def send(self, command: MirrorCommand) -> None:
        self.sent.append(command)

    def poll_status(self, node_id: int) -> MirrorStatus:
        self.polls.append(int(node_id))
        return MirrorStatus(
            node_id=node_id,
            azimuth_home="homed",
            elevation_home="homed",
            azimuth_deg=10.0,
            elevation_deg=20.0,
            mode="idle",
        )


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
    # Entering jog seeds a hold pose via set_target (no wire jog command).
    assert any(c.command == CommandName.SET_TARGET for c in transport.sent)
    before = len(transport.sent)
    app.control_tick()
    # Jog with zero rates: no additional closed-loop SET_TARGET from track/park.
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

    transport.sent.clear()
    app.set_heat_demand(True)
    app.control_tick()
    on_absorber = [c for c in transport.sent if c.command == CommandName.SET_TARGET]
    assert on_absorber
    absorber_pose = (on_absorber[0].payload["azimuth_deg"], on_absorber[0].payload["elevation_deg"])

    app.set_heat_demand(False)
    divert_target = app.current_snapshot()["targets"]["0"]
    app.set_heat_demand(True)
    absorber_target = app.current_snapshot()["targets"]["0"]
    if absorber_target["mode"] == "parked":
        assert divert_target["mode"] == "parked"
        assert divert_pose == (0.0, 0.0)
        assert absorber_pose == (0.0, 0.0)
    else:
        assert divert_target["mode"] == "tracking"
        assert absorber_target["mode"] == "tracking"
        assert divert_pose != absorber_pose


def test_park_is_face_up_at_oven_facing() -> None:
    from hotbox_shared import oven_facing_azimuth_deg

    app, transport = _app()
    transport.sent.clear()
    app.set_mode("park")
    parks = [c for c in transport.sent if c.command == CommandName.SET_TARGET]
    assert parks
    assert all(c.payload.get("elevation_deg") == 90.0 for c in parks)
    for c in parks:
        facing = oven_facing_azimuth_deg(
            app._mirror_world_for_node(int(c.node_id)),
            app.absorber_world,
        )
        assert abs(((float(c.payload["azimuth_deg"]) - facing + 180.0) % 360.0) - 180.0) < 1e-6
    assert "mode" not in parks[0].payload
    assert all(t["mode"] == "parked" for t in app.current_snapshot()["targets"].values())


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


def test_jog_integrates_into_set_target() -> None:
    from hotbox_controller.app import JogRequest
    import time

    app, transport = _app()
    transport.sent.clear()
    app.jog(JogRequest(node_id=0, azimuth_rate_deg_s=10.0, elevation_rate_deg_s=0.0))
    assert app.node_mode(0) == "jog"
    # Seed hold + possible first integrate (dt may be ~0).
    assert any(c.command == CommandName.SET_TARGET for c in transport.sent)
    time.sleep(0.05)
    transport.sent.clear()
    app.control_tick()
    moved = [c for c in transport.sent if c.command == CommandName.SET_TARGET and c.node_id == 0]
    assert moved
    assert moved[-1].payload["azimuth_deg"] != 10.0  # left the seed pose (status was 10°)


def test_set_mode_rejects_unknown() -> None:
    app, _transport = _app()
    try:
        app.set_mode("fly")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "unsupported" in str(exc)


def test_send_protocol_command_raw_wire() -> None:
    from hotbox_controller.app import ProtocolCommandRequest

    app, transport = _app()
    before = len(transport.sent)

    home = app.send_protocol_command(ProtocolCommandRequest(command="home", node_id=0))
    assert home["command"] == "home"
    assert transport.sent[before].command == CommandName.HOME
    assert transport.sent[before].node_id == 0

    target = app.send_protocol_command(
        ProtocolCommandRequest(
            command="set_target",
            node_id=1,
            azimuth_deg=200.0,
            elevation_deg=34.0,
        )
    )
    assert target["payload"]["azimuth_deg"] == 200.0
    assert target["payload"]["elevation_deg"] == 34.0
    assert "mode" not in target["payload"]
    assert transport.sent[-1].command == CommandName.SET_TARGET

    status = app.send_protocol_command(ProtocolCommandRequest(command="get_status", node_id=0))
    assert status["mirror_status"]["node_id"] == 0
    assert status["mirror_status"]["homed"] is True

    cleared = app.send_protocol_command(ProtocolCommandRequest(command="clear_error", node_id=0))
    assert cleared["command"] == "clear_error"
    assert transport.sent[-1].command == CommandName.CLEAR_ERROR

    discovered = app.send_protocol_command(ProtocolCommandRequest(command="discover"))
    assert {n["node_id"] for n in discovered["nodes"]} == {0, 1}


def test_send_protocol_command_rejects_unknown() -> None:
    from hotbox_controller.app import ProtocolCommandRequest

    app, _transport = _app()
    try:
        app.send_protocol_command(ProtocolCommandRequest(command="jog", node_id=0))
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "unknown protocol command" in str(exc)


def test_protocol_traffic_is_recorded() -> None:
    from hotbox_controller.app import ProtocolCommandRequest

    app, _transport = _app()
    app.send_protocol_command(ProtocolCommandRequest(command="home", node_id=0))
    app.send_protocol_command(ProtocolCommandRequest(command="get_status", node_id=0))
    traffic = app.current_snapshot()["protocol_traffic"]
    kinds = [e["kind"] for e in traffic if e.get("node_id") == 0]
    assert "home" in kinds
    assert "status" in kinds
    assert any(e["direction"] == "tx" for e in traffic)
    assert any(e["direction"] == "rx" for e in traffic)


def test_protocol_set_target_is_passthrough() -> None:
    """Protocol set_target must not rewrite/clamp az/el (firmware defends itself)."""
    from hotbox_controller.app import ProtocolCommandRequest

    app, transport = _app()
    app.send_protocol_command(
        ProtocolCommandRequest(
            command="set_target",
            node_id=0,
            azimuth_deg=350.0,
            elevation_deg=97.0,
        )
    )
    cmd = transport.sent[-1]
    assert cmd.command == CommandName.SET_TARGET
    assert float(cmd.payload["azimuth_deg"]) == 350.0
    assert float(cmd.payload["elevation_deg"]) == 97.0


def test_jog_stops_at_joint_limits() -> None:
    from hotbox_controller.app import JogRequest
    import time

    app, transport = _app()
    # Seed near the elevation floor and jog further down.
    app._jog_pose[0] = (app._oven_facing_deg(0), 0.5)
    app._node_modes[0] = "jog"
    app._jog_rates[0] = (0.0, -20.0)
    app._jog_last_mono[0] = time.monotonic() - 0.1
    transport.sent.clear()
    app.control_tick()
    moved = [c for c in transport.sent if c.command == CommandName.SET_TARGET and c.node_id == 0]
    assert moved
    assert float(moved[-1].payload["elevation_deg"]) >= app._joint_limits().elevation_min_deg - 1e-6
    assert app._jog_pose[0][1] >= app._joint_limits().elevation_min_deg - 1e-6
    from hotbox_controller.app import ProtocolCommandRequest

    app, transport = _app()
    app.set_mode("track")
    app.set_mirror_mode(0, "raw")
    assert app.node_mode(0) == "raw"
    assert app.node_mode(1) == "track"
    assert app.mode == "mixed"
    assert app.current_snapshot()["mirror_modes"]["0"] == "raw"

    transport.sent.clear()
    transport.polls.clear()
    app.control_tick()
    # Raw node is not polled or commanded; track node still is.
    assert 0 not in transport.polls
    assert 1 in transport.polls
    assert not any(c.node_id == 0 for c in transport.sent)
    assert any(c.command == CommandName.SET_TARGET and c.node_id == 1 for c in transport.sent)

    transport.polls.clear()
    app.current_snapshot()
    assert 0 not in transport.polls
    assert 1 in transport.polls

    app.send_protocol_command(
        ProtocolCommandRequest(
            command="set_target",
            node_id=0,
            azimuth_deg=1.0,
            elevation_deg=2.0,
        )
    )
    assert transport.sent[-1].command == CommandName.SET_TARGET
    assert transport.sent[-1].node_id == 0

    app.set_mode("raw")
    assert app.mode == "raw"
    transport.sent.clear()
    transport.polls.clear()
    app.control_tick()
    assert transport.sent == []
    assert transport.polls == []

    app.set_mode("track")
    transport.sent.clear()
    transport.polls.clear()
    app.control_tick()
    assert any(c.command == CommandName.SET_TARGET for c in transport.sent)
    assert transport.polls
