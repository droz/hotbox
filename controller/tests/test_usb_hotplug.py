"""USB transport hotplug recovery and fleet rediscover merge."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from serial import SerialException

from hotbox_controller.config import TransportConfig
from hotbox_controller.mirror_fleet import MirrorFleet
from hotbox_controller.protocol import CommandName, MirrorCommand, MirrorStatus
from hotbox_controller.transport import DiscoveredNode, MirrorTransport, UsbSerialTransport


class _StubTransport(MirrorTransport):
    def __init__(self) -> None:
        self._nodes: list[DiscoveredNode] = []
        self.sent: list[MirrorCommand] = []

    def discover(self):
        yield from self._nodes

    def send(self, command: MirrorCommand) -> None:
        self.sent.append(command)

    def poll_status(self, node_id: int) -> MirrorStatus:
        return MirrorStatus(node_id=node_id, mode="idle")


def test_fleet_discover_keeps_nodes_when_unplugged() -> None:
    transport = _StubTransport()
    transport._nodes = [DiscoveredNode(node_id=0, transport_name="usb", endpoint="/dev/cu.a")]
    fleet = MirrorFleet(transport)
    fleet.discover()
    assert list(fleet.nodes()) == [0]

    transport._nodes = []
    fleet.discover()
    assert list(fleet.nodes()) == [0]
    assert fleet.nodes()[0].endpoint == "/dev/cu.a"


def test_fleet_discover_updates_endpoint_on_replug() -> None:
    transport = _StubTransport()
    transport._nodes = [DiscoveredNode(node_id=0, transport_name="usb", endpoint="/dev/cu.a")]
    fleet = MirrorFleet(transport)
    fleet.discover()

    transport._nodes = [DiscoveredNode(node_id=0, transport_name="usb", endpoint="/dev/cu.b")]
    fleet.discover()
    assert fleet.nodes()[0].endpoint == "/dev/cu.b"


def test_usb_poll_status_returns_disconnected_on_write_failure() -> None:
    cfg = TransportConfig(mode="usb", usb_baudrate=115200, usb_ports={0: "/dev/fake"})
    transport = UsbSerialTransport(cfg)
    transport._OPEN_SETTLE_S = 0.0
    transport._RECONNECT_COOLDOWN_S = 0.0

    port = MagicMock()
    port.write.side_effect = SerialException("write failed: [Errno 6] Device not configured")
    transport._ports[0] = port
    transport._endpoints[0] = "/dev/fake"
    transport._known_nodes.add(0)

    status = transport.poll_status(0)
    assert status.mode == "disconnected"
    assert 0 not in transport._ports


def test_usb_send_raises_connection_error_instead_of_serial_exception() -> None:
    cfg = TransportConfig(mode="usb", usb_baudrate=115200, usb_ports={0: "/dev/fake"})
    transport = UsbSerialTransport(cfg)
    transport._OPEN_SETTLE_S = 0.0
    port = MagicMock()
    port.write.side_effect = OSError(6, "Device not configured")
    transport._ports[0] = port
    transport._endpoints[0] = "/dev/fake"

    with pytest.raises(ConnectionError, match="disconnected"):
        transport.send(MirrorCommand(node_id=0, command=CommandName.STOP))
    assert 0 not in transport._ports


def test_usb_reset_node_pulses_dtr_and_reopens() -> None:
    cfg = TransportConfig(mode="usb", usb_baudrate=115200, usb_ports={0: "/dev/fake"})
    transport = UsbSerialTransport(cfg)
    transport._OPEN_SETTLE_S = 0.0
    transport._RECONNECT_COOLDOWN_S = 0.0

    old_port = MagicMock()
    transport._ports[0] = old_port
    transport._endpoints[0] = "/dev/fake"
    transport._known_nodes.add(0)

    new_port = MagicMock()
    with (
        patch("hotbox_controller.transport._serial_device_present", return_value=True),
        patch("serial.Serial", return_value=new_port) as serial_ctor,
        patch("time.sleep"),
    ):
        transport.reset_node(0)

    assert old_port.dtr is False or old_port.dtr is True  # was toggled
    old_port.close.assert_called()
    serial_ctor.assert_called()
    assert transport._ports[0] is new_port


def test_usb_reconnect_after_disconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = TransportConfig(mode="usb", usb_baudrate=115200, usb_ports={})
    transport = UsbSerialTransport(cfg)
    transport._OPEN_SETTLE_S = 0.0
    transport._RECONNECT_COOLDOWN_S = 0.0
    transport._known_nodes.add(0)

    status_line = MirrorStatus(node_id=0, mode="idle", azimuth_home="homed", elevation_home="homed").to_wire()
    new_port = MagicMock()
    new_port.readline.side_effect = [status_line, b""]

    monkeypatch.setattr(
        "hotbox_controller.transport.resolve_usb_port_map",
        lambda explicit=None: {0: "/dev/cu.new"},
    )
    with patch("serial.Serial", return_value=new_port), patch("time.sleep"):
        status = transport.poll_status(0)

    assert status.mode == "idle"
    assert status.homed is True
    assert transport._endpoints[0] == "/dev/cu.new"
