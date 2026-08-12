from __future__ import annotations

from hotbox_controller.app import ControllerApplication
from hotbox_controller.protocol import CommandName, MirrorCommand, MirrorStatus
from hotbox_controller.transport import DiscoveredNode, MirrorTransport


class FakeTransport(MirrorTransport):
    def __init__(self, node_ids: list[int] | None = None) -> None:
        self.node_ids = node_ids or [0]
        self.sent: list[MirrorCommand] = []
        self.polls: list[int] = []
        self._pose: dict[int, tuple[float, float]] = {
            int(n): (10.0, 20.0) for n in (node_ids or [0])
        }
        self._target: dict[int, tuple[float, float]] = dict(self._pose)
        self._mode: dict[int, str] = {int(n): "idle" for n in (node_ids or [0])}

    def discover(self):
        for node_id in self.node_ids:
            yield DiscoveredNode(node_id=node_id, transport_name="fake", endpoint=f"fake://{node_id}")

    def send(self, command: MirrorCommand) -> None:
        self.sent.append(command)
        nid = int(command.node_id)
        if command.command == CommandName.SET_TARGET:
            if bool(command.payload.get("hold_current")):
                self._target[nid] = self._pose[nid]
            else:
                az = float(command.payload.get("azimuth_deg", self._target[nid][0]))
                el = float(command.payload.get("elevation_deg", self._target[nid][1]))
                self._target[nid] = (az, el)
        elif command.command == CommandName.START:
            self._mode[nid] = "position"
        elif command.command == CommandName.STOP:
            self._mode[nid] = "idle"
        elif command.command == CommandName.SET_VELOCITY:
            self._mode[nid] = "velocity"

    def poll_status(self, node_id: int) -> MirrorStatus:
        self.polls.append(int(node_id))
        az, el = self._pose[int(node_id)]
        taz, tel = self._target[int(node_id)]
        return MirrorStatus(
            node_id=node_id,
            azimuth_home="homed",
            elevation_home="homed",
            azimuth_deg=az,
            elevation_deg=el,
            target_azimuth_deg=taz,
            target_elevation_deg=tel,
            mode=self._mode[int(node_id)],
        )


def _app() -> tuple[ControllerApplication, FakeTransport]:
    transport = FakeTransport([0, 1])
    app = ControllerApplication(transport=transport)
    app.startup()
    return app, transport


def test_set_mode_track_applies_targets_immediately() -> None:
    """Track must send set_target without waiting for a background loop."""
    app, transport = _app()
    app.set_mode("park")
    # Park leaves firmware in position mode; Track only needs a new setpoint.
    for nid in transport.node_ids:
        transport._mode[nid] = "position"
    transport.sent.clear()
    app.set_mode("track")
    assert any(c.command == CommandName.SET_TARGET for c in transport.sent)
    # Already position-servoing → no redundant start.
    assert not any(c.command == CommandName.START for c in transport.sent)


def test_control_tick_skips_unchanged_track_targets() -> None:
    """Settled Track must not flood USB with identical set_target/start every tick."""
    app, transport = _app()
    app.set_mode("track")
    # Pretend firmware is already position-servoing the last target.
    for nid in transport.node_ids:
        transport._mode[nid] = "position"
        if nid in app._last_sent_wire_targets:
            transport._target[nid] = app._last_sent_wire_targets[nid]
    transport.sent.clear()
    app.control_tick()
    assert not any(c.command == CommandName.SET_TARGET for c in transport.sent)
    assert not any(c.command == CommandName.START for c in transport.sent)


def test_track_sends_start_when_idle() -> None:
    app, transport = _app()
    for nid in transport.node_ids:
        transport._mode[nid] = "idle"
    app._last_sent_wire_targets.clear()
    transport.sent.clear()
    app.set_mode("track")
    assert any(c.command == CommandName.SET_TARGET for c in transport.sent)
    assert any(c.command == CommandName.START for c in transport.sent)


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
        # Face-up stow: el=90°, az=oven-facing (node 0 is south → ~180°).
        assert divert_pose[1] == 90.0
        assert absorber_pose[1] == 90.0
        assert abs(divert_pose[0] - 180.0) < 1.0
        assert abs(absorber_pose[0] - 180.0) < 1.0
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


def test_jog_streams_set_velocity() -> None:
    from hotbox_controller.app import JogRequest

    app, transport = _app()
    transport.sent.clear()
    app.jog(JogRequest(node_id=0, azimuth_rate_deg_s=10.0, elevation_rate_deg_s=-2.0))
    assert app.node_mode(0) == "jog"
    # Seed hold uses set_target, then motion uses set_velocity.
    assert any(c.command == CommandName.SET_TARGET for c in transport.sent)
    vels = [c for c in transport.sent if c.command == CommandName.SET_VELOCITY and c.node_id == 0]
    assert vels
    assert vels[-1].payload["azimuth_deg_s"] == 10.0
    assert vels[-1].payload["elevation_deg_s"] == -2.0
    transport.sent.clear()
    app.control_tick()
    assert any(c.command == CommandName.SET_VELOCITY for c in transport.sent)


def test_set_mode_rejects_unknown() -> None:
    app, _transport = _app()
    try:
        app.set_mode("fly")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "unsupported" in str(exc)


def test_geometry_target_uses_hardware_setpoint_not_supervisor() -> None:
    """Yellow overlay follows firmware target_* from status, not track/park solve."""
    from hotbox_controller.app import ProtocolCommandRequest

    app, transport = _app()
    # Seed status cache while still auto-polling, then switch to raw.
    app.current_snapshot()
    app.set_mirror_mode(0, "raw")
    app.send_protocol_command(
        ProtocolCommandRequest(
            command="set_target",
            node_id=0,
            azimuth_deg=55.0,
            elevation_deg=66.0,
        )
    )

    snap = app.current_snapshot()
    mirror = next(m for m in snap["geometry"]["target"]["mirrors"] if m["node_id"] == 0)
    assert abs(mirror["azimuth_deg"] - 55.0) < 1e-6
    assert abs(mirror["elevation_deg"] - 66.0) < 1e-6
    live = next(m for m in snap["geometry"]["live"]["mirrors"] if m["node_id"] == 0)
    assert abs(live["azimuth_deg"] - 10.0) < 1e-6
    assert abs(live["elevation_deg"] - 20.0) < 1e-6
    assert abs(snap["mirrors"]["0"]["target_azimuth_deg"] - 55.0) < 1e-6
    assert abs(snap["mirrors"]["0"]["target_elevation_deg"] - 66.0) < 1e-6


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


def test_jog_release_holds_current_and_starts() -> None:
    from hotbox_controller.app import JogRequest

    app, transport = _app()
    app.jog(JogRequest(node_id=0, azimuth_rate_deg_s=5.0, elevation_rate_deg_s=0.0))
    transport.sent.clear()
    app.jog(JogRequest(node_id=0, azimuth_rate_deg_s=0.0, elevation_rate_deg_s=0.0))
    assert not any(c.command == CommandName.STOP for c in transport.sent)
    holds = [c for c in transport.sent if c.command == CommandName.SET_TARGET]
    assert holds and holds[-1].payload.get("hold_current") is True
    assert any(c.command == CommandName.START for c in transport.sent)


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


def test_jog_streams_velocity_not_position_slew() -> None:
    """Jog with nonzero rates commands set_velocity (limits defended by firmware later)."""
    import time

    app, transport = _app()
    app._jog_pose[0] = (app._oven_facing_deg(0), 0.5)
    app._node_modes[0] = "jog"
    app._jog_rates[0] = (0.0, -20.0)
    app._jog_last_mono[0] = time.monotonic() - 0.1
    transport.sent.clear()
    app.control_tick()
    vels = [c for c in transport.sent if c.command == CommandName.SET_VELOCITY and c.node_id == 0]
    assert vels
    assert float(vels[-1].payload["elevation_deg_s"]) == -20.0


def test_raw_mode_skips_control_loop() -> None:
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
    # Force a new wire command for the track node (otherwise skip-cache suppresses it).
    app._last_sent_wire_targets.pop(1, None)
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
    app._last_sent_wire_targets.clear()
    app.control_tick()
    assert any(c.command == CommandName.SET_TARGET for c in transport.sent)
    assert transport.polls
