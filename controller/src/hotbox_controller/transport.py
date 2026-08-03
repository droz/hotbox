from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
import logging
import threading
import time
from typing import Any, Iterable

from hotbox_shared import utc_now

from .config import TransportConfig
from .protocol import CAN_CMD_BASE_ID, CAN_RSP_BASE_ID, CommandName, MirrorCommand, MirrorStatus

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DiscoveredNode:
    node_id: int
    transport_name: str
    endpoint: str


@dataclass(slots=True)
class ProtocolTrafficEntry:
    seq: int
    timestamp_utc: str
    direction: str  # "tx" | "rx"
    node_id: int | None
    kind: str
    payload: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "timestamp_utc": self.timestamp_utc,
            "direction": self.direction,
            "node_id": self.node_id,
            "kind": self.kind,
            "payload": self.payload,
        }


class ProtocolTrafficLog:
    """Ring buffer of recent protocol TX/RX messages for the web console."""

    def __init__(self, capacity: int = 400) -> None:
        self._capacity = max(1, int(capacity))
        self._entries: deque[ProtocolTrafficEntry] = deque(maxlen=self._capacity)
        self._seq = 0
        self._lock = threading.Lock()

    def record(
        self,
        *,
        direction: str,
        kind: str,
        node_id: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> ProtocolTrafficEntry:
        with self._lock:
            self._seq += 1
            entry = ProtocolTrafficEntry(
                seq=self._seq,
                timestamp_utc=utc_now().isoformat(),
                direction=direction,
                node_id=None if node_id is None else int(node_id),
                kind=str(kind),
                payload=dict(payload or {}),
            )
            self._entries.append(entry)
            return entry

    def snapshot(self, *, limit: int = 200, node_id: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            entries = list(self._entries)
        if node_id is not None:
            want = int(node_id)
            entries = [e for e in entries if e.node_id == want]
        if limit > 0:
            entries = entries[-limit:]
        return [e.as_dict() for e in entries]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


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


class LoggingMirrorTransport(MirrorTransport):
    """Wraps a transport and records wire traffic for the protocol console."""

    def __init__(self, inner: MirrorTransport, traffic: ProtocolTrafficLog) -> None:
        self._inner = inner
        self._traffic = traffic
        self._last_status_log_mono: dict[int, float] = {}
        self._last_status_payload: dict[int, dict[str, Any]] = {}
        self._last_tx_log_mono: dict[tuple[int, str], float] = {}
        self._last_tx_payload: dict[tuple[int, str], dict[str, Any]] = {}
        self._status_log_interval_s = 0.25
        self._tx_repeat_interval_s = 0.25

    def discover(self) -> Iterable[DiscoveredNode]:
        nodes = list(self._inner.discover())
        self._traffic.record(
            direction="tx",
            kind="discover",
            node_id=None,
            payload={"node_ids": [n.node_id for n in nodes]},
        )
        return nodes

    def send(self, command: MirrorCommand) -> None:
        key = (int(command.node_id), str(command.command))
        payload = dict(command.payload)
        now = time.monotonic()
        last_t = self._last_tx_log_mono.get(key, 0.0)
        last_payload = self._last_tx_payload.get(key)
        # Collapse high-rate identical repeats (e.g. Track set_target each tick).
        if payload != last_payload or (now - last_t) >= self._tx_repeat_interval_s:
            self._traffic.record(
                direction="tx",
                kind=str(command.command),
                node_id=command.node_id,
                payload=payload,
            )
            self._last_tx_log_mono[key] = now
            self._last_tx_payload[key] = payload
        self._inner.send(command)

    def poll_status(self, node_id: int) -> MirrorStatus:
        status = self._inner.poll_status(node_id)
        payload = status.as_dict()
        now = time.monotonic()
        key = (int(node_id), "get_status")
        last_tx_t = self._last_tx_log_mono.get(key, 0.0)
        if (now - last_tx_t) >= self._tx_repeat_interval_s:
            self._traffic.record(direction="tx", kind="get_status", node_id=node_id, payload={})
            self._last_tx_log_mono[key] = now
            self._last_tx_payload[key] = {}
        last_t = self._last_status_log_mono.get(node_id, 0.0)
        last_payload = self._last_status_payload.get(node_id)
        # Status polls are frequent (UI + control loop). Log on change or periodically.
        if payload != last_payload or (now - last_t) >= self._status_log_interval_s:
            self._traffic.record(direction="rx", kind="status", node_id=node_id, payload=payload)
            self._last_status_log_mono[node_id] = now
            self._last_status_payload[node_id] = payload
        return status

    def close(self) -> None:
        self._inner.close()


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
