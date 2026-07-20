from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import time
from typing import Any, Iterable

from .config import TransportConfig
from .protocol import CAN_CMD_BASE_ID, CAN_RSP_BASE_ID, CommandName, MirrorCommand, MirrorStatus

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DiscoveredNode:
    node_id: int
    transport_name: str
    endpoint: str


class MirrorTransport(ABC):
    @abstractmethod
    def discover(self) -> Iterable[DiscoveredNode]:
        raise NotImplementedError

    @abstractmethod
    def send(self, command: MirrorCommand) -> None:
        raise NotImplementedError

    @abstractmethod
    def poll_status(self, node_id: int) -> MirrorStatus:
        raise NotImplementedError

    def close(self) -> None:
        return None


class UsbSerialTransport(MirrorTransport):
    def __init__(self, config: TransportConfig) -> None:
        self._config = config
        self._ports: dict[int, Any] = {}
        self._status_cache: dict[int, MirrorStatus] = {}

    def _open_port(self, node_id: int, endpoint: str):
        import serial

        port = serial.Serial(endpoint, baudrate=self._config.usb_baudrate, timeout=0.05)
        port.reset_input_buffer()
        return port

    def discover(self) -> Iterable[DiscoveredNode]:
        for node_id, endpoint in sorted(self._config.usb_ports.items()):
            if node_id not in self._ports:
                try:
                    self._ports[node_id] = self._open_port(node_id, endpoint)
                except Exception as exc:
                    logger.warning("failed to open USB port %s for node %s: %s", endpoint, node_id, exc)
                    continue
            yield DiscoveredNode(node_id=node_id, transport_name="usb", endpoint=endpoint)

    def send(self, command: MirrorCommand) -> None:
        port = self._ports.get(command.node_id)
        if port is None:
            raise KeyError(f"USB node {command.node_id} is not connected")
        port.write(command.to_wire())
        port.flush()
        if command.command == CommandName.GET_STATUS:
            self._status_cache[command.node_id] = self._read_status(command.node_id, port)

    def poll_status(self, node_id: int) -> MirrorStatus:
        if node_id in self._status_cache:
            return self._status_cache.pop(node_id)
        port = self._ports.get(node_id)
        if port is None:
            return MirrorStatus(node_id=node_id, mode="disconnected")
        self.send(MirrorCommand(node_id=node_id, command=CommandName.GET_STATUS))
        return self._status_cache.pop(node_id, MirrorStatus(node_id=node_id, mode="timeout"))

    def _read_status(self, node_id: int, port) -> MirrorStatus:
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            raw = port.readline()
            if not raw:
                continue
            try:
                return MirrorStatus.from_wire(raw.strip())
            except (ValueError, KeyError, TypeError):
                continue
        return MirrorStatus(node_id=node_id, mode="timeout")

    def close(self) -> None:
        for port in self._ports.values():
            port.close()
        self._ports.clear()


class CanTransport(MirrorTransport):
    def __init__(self, config: TransportConfig) -> None:
        self._config = config
        self._bus = None
        self._known_nodes: set[int] = set()

    def _ensure_bus(self):
        if self._bus is not None:
            return self._bus
        import can

        self._bus = can.interface.Bus(
            channel=self._config.can_channel,
            interface="socketcan",
            bitrate=self._config.can_bitrate,
        )
        return self._bus

    def discover(self) -> Iterable[DiscoveredNode]:
        self._ensure_bus()
        for node_id in sorted(self._known_nodes):
            yield DiscoveredNode(node_id=node_id, transport_name="can", endpoint=self._config.can_channel)
        for node_id in sorted(self._config.usb_ports.keys()):
            self._known_nodes.add(node_id)
            yield DiscoveredNode(node_id=node_id, transport_name="can", endpoint=self._config.can_channel)

    def send(self, command: MirrorCommand) -> None:
        import can

        bus = self._ensure_bus()
        arbitration_id = CAN_CMD_BASE_ID + command.node_id
        msg = can.Message(arbitration_id=arbitration_id, data=command.to_can_frame(), is_extended_id=False)
        bus.send(msg)
        self._known_nodes.add(command.node_id)

    def poll_status(self, node_id: int) -> MirrorStatus:
        bus = self._ensure_bus()
        self.send(MirrorCommand(node_id=node_id, command=CommandName.GET_STATUS))
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            msg = bus.recv(timeout=0.05)
            if msg is None:
                continue
            if msg.arbitration_id != CAN_RSP_BASE_ID + node_id:
                continue
            try:
                return MirrorStatus.from_can_frame(node_id, bytes(msg.data))
            except ValueError:
                continue
        return MirrorStatus(node_id=node_id, mode="timeout")

    def close(self) -> None:
        if self._bus is not None:
            self._bus.shutdown()
            self._bus = None


class SimTransport(MirrorTransport):
    def __init__(self, nodes: dict[int, Any], lock: Any | None = None) -> None:
        self._nodes = nodes
        self._lock = lock

    def discover(self) -> Iterable[DiscoveredNode]:
        for node_id in sorted(self._nodes):
            yield DiscoveredNode(node_id=node_id, transport_name="sim", endpoint=f"sim://node/{node_id}")

    def send(self, command: MirrorCommand) -> None:
        if self._lock is None:
            self._send_unlocked(command)
            return
        with self._lock:
            self._send_unlocked(command)

    def _send_unlocked(self, command: MirrorCommand) -> None:
        node = self._nodes.get(command.node_id)
        if node is None:
            raise KeyError(f"sim node {command.node_id} is not registered")
        node.handle_command(command)

    def poll_status(self, node_id: int) -> MirrorStatus:
        if self._lock is None:
            return self._poll_unlocked(node_id)
        with self._lock:
            return self._poll_unlocked(node_id)

    def _poll_unlocked(self, node_id: int) -> MirrorStatus:
        node = self._nodes.get(node_id)
        if node is None:
            return MirrorStatus(node_id=node_id, mode="disconnected")
        return node.status()


def build_transport(config: TransportConfig, sim_nodes: dict[int, Any] | None = None) -> MirrorTransport:
    if config.mode == "can":
        return CanTransport(config)
    if config.mode == "usb":
        return UsbSerialTransport(config)
    if config.mode == "sim":
        if sim_nodes is None:
            raise ValueError("sim transport requires sim_nodes")
        return SimTransport(sim_nodes)
    raise ValueError(f"unsupported transport mode: {config.mode}")
