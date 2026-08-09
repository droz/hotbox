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
from .usb_discover import resolve_usb_port_map

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

    def reset_node(self, node_id: int) -> None:
        raise RuntimeError(f"{type(self).__name__} does not support hardware reset")

    def close(self) -> None:
        return None


def _is_serial_io_error(exc: BaseException) -> bool:
    """True for USB unplug / broken pipe / device-not-configured style failures."""
    try:
        from serial import SerialException
    except Exception:  # pragma: no cover - pyserial always present for USB mode
        SerialException = ()  # type: ignore[misc, assignment]
    if isinstance(exc, OSError):
        return True
    if SerialException and isinstance(exc, SerialException):
        return True
    return False


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
        now = time.monotonic()
        key = (int(node_id), "get_status")
        last_tx_t = self._last_tx_log_mono.get(key, 0.0)
        # Log TX before the blocking USB read so timestamps reflect request time.
        if (now - last_tx_t) >= self._tx_repeat_interval_s:
            self._traffic.record(direction="tx", kind="get_status", node_id=node_id, payload={})
            self._last_tx_log_mono[key] = now
            self._last_tx_payload[key] = {}

        status = self._inner.poll_status(node_id)
        payload = status.as_dict()
        after = time.monotonic()
        last_t = self._last_status_log_mono.get(node_id, 0.0)
        last_payload = self._last_status_payload.get(node_id)
        # Host-side timeouts are not firmware RX — label them clearly in the log.
        kind = "timeout" if status.mode in {"timeout", "disconnected"} else "status"
        if payload != last_payload or (after - last_t) >= self._status_log_interval_s:
            self._traffic.record(direction="rx", kind=kind, node_id=node_id, payload=payload)
            self._last_status_log_mono[node_id] = after
            self._last_status_payload[node_id] = payload
        return status

    def reset_node(self, node_id: int) -> None:
        self._traffic.record(direction="tx", kind="reset", node_id=node_id, payload={"via": "dtr"})
        self._inner.reset_node(node_id)
        self._traffic.record(
            direction="rx",
            kind="reset",
            node_id=node_id,
            payload={"status": "ok"},
        )

    def close(self) -> None:
        self._inner.close()


class UsbSerialTransport(MirrorTransport):
    """USB CDC transport with hotplug recovery and DTR/open reset."""

    _RECONNECT_COOLDOWN_S = 1.5
    _OPEN_SETTLE_S = 2.0

    def __init__(self, config: TransportConfig) -> None:
        self._config = config
        self._ports: dict[int, Any] = {}
        self._endpoints: dict[int, str] = {}
        self._known_nodes: set[int] = set(int(n) for n in (config.usb_ports or {}))
        self._status_cache: dict[int, MirrorStatus] = {}
        self._last_reconnect_mono: dict[int, float] = {}
        self._lock = threading.Lock()

    def _port_map(self) -> dict[int, str]:
        """Resolve node→device, falling back to VID/PID auto when explicit paths are gone."""
        explicit = self._config.usb_ports or {}
        port_map = resolve_usb_port_map(explicit if explicit else None)
        if not explicit:
            return port_map

        # Explicit map: if every configured path is missing (typical after unplug/
        # replug on macOS when the cu.usbmodem* number changes), use auto-discover.
        missing = [path for path in port_map.values() if not _serial_device_present(path)]
        if missing and len(missing) == len(port_map):
            auto = resolve_usb_port_map(None)
            if auto:
                logger.warning(
                    "configured USB ports missing %s; falling back to auto-discover %s",
                    missing,
                    auto,
                )
                return auto
        return port_map

    def _drop_port_unlocked(self, node_id: int) -> None:
        port = self._ports.pop(node_id, None)
        self._status_cache.pop(node_id, None)
        if port is None:
            return
        try:
            port.close()
        except Exception:
            pass

    def _open_port(self, node_id: int, endpoint: str):
        import serial

        # Nano ESP32 resets when the CDC port opens; give firmware time to boot
        # before the first get_status. Do NOT force DTR/RTS low after open —
        # that breaks TinyUSB CDC RX on this board (writes succeed, replies never arrive).
        port = serial.Serial(
            endpoint,
            baudrate=self._config.usb_baudrate,
            timeout=0.2,
            write_timeout=1.0,
            dsrdtr=False,
            rtscts=False,
        )
        time.sleep(self._OPEN_SETTLE_S)
        try:
            port.reset_input_buffer()
        except Exception:
            pass
        logger.info("opened USB port %s for node %s (settled)", endpoint, node_id)
        return port

    def _resolve_endpoint(self, node_id: int) -> str | None:
        port_map = self._port_map()
        endpoint = port_map.get(node_id)
        if endpoint is None and len(port_map) == 1 and node_id in self._known_nodes:
            # Single-board bring-up: accept the only device for a previously known node.
            endpoint = next(iter(port_map.values()))
        return endpoint

    def _try_reconnect(self, node_id: int) -> bool:
        """Attempt to open the serial port for ``node_id``.

        Settling sleeps happen outside the transport lock so /api/state and the
        control loop are not blocked for the full boot wait under the mutex.
        """
        with self._lock:
            now = time.monotonic()
            last = self._last_reconnect_mono.get(node_id, 0.0)
            if (now - last) < self._RECONNECT_COOLDOWN_S:
                return False
            self._last_reconnect_mono[node_id] = now
            endpoint = self._resolve_endpoint(node_id)
            if endpoint is None:
                return False
            self._drop_port_unlocked(node_id)

        try:
            port = self._open_port(node_id, endpoint)
        except Exception as exc:
            logger.warning("USB reconnect failed for node %s on %s: %s", node_id, endpoint, exc)
            return False

        with self._lock:
            # Another thread may have opened meanwhile; prefer the newest handle.
            self._drop_port_unlocked(node_id)
            self._ports[node_id] = port
            self._endpoints[node_id] = endpoint
            self._known_nodes.add(node_id)
            return True

    def discover(self) -> Iterable[DiscoveredNode]:
        port_map = self._port_map()
        to_open: list[tuple[int, str]] = []
        with self._lock:
            for node_id in list(self._ports):
                if node_id not in port_map or self._endpoints.get(node_id) != port_map.get(node_id):
                    self._drop_port_unlocked(node_id)

            for node_id, endpoint in sorted(port_map.items()):
                self._known_nodes.add(node_id)
                self._endpoints[node_id] = endpoint
                if node_id not in self._ports:
                    to_open.append((node_id, endpoint))

            known_snapshot = sorted(self._known_nodes)

        for node_id, endpoint in to_open:
            try:
                port = self._open_port(node_id, endpoint)
            except Exception as exc:
                logger.warning("failed to open USB port %s for node %s: %s", endpoint, node_id, exc)
                continue
            with self._lock:
                self._drop_port_unlocked(node_id)
                self._ports[node_id] = port
                self._endpoints[node_id] = endpoint

        with self._lock:
            # Currently mapped nodes (open or not) plus remembered unplugged ones.
            out: list[DiscoveredNode] = []
            yielded: set[int] = set()
            for node_id, endpoint in sorted(port_map.items()):
                yielded.add(node_id)
                out.append(DiscoveredNode(node_id=node_id, transport_name="usb", endpoint=endpoint))
            for node_id in known_snapshot:
                if node_id in yielded:
                    continue
                out.append(
                    DiscoveredNode(
                        node_id=node_id,
                        transport_name="usb",
                        endpoint=self._endpoints.get(node_id, ""),
                    )
                )
        yield from out

    def send(self, command: MirrorCommand) -> None:
        node_id = int(command.node_id)
        with self._lock:
            port = self._ports.get(node_id)
        if port is None:
            if not self._try_reconnect(node_id):
                raise ConnectionError(f"USB node {node_id} is not connected")
            with self._lock:
                port = self._ports.get(node_id)
            if port is None:
                raise ConnectionError(f"USB node {node_id} is not connected")

        with self._lock:
            # Re-fetch under lock in case of concurrent drop.
            port = self._ports.get(node_id)
            if port is None:
                raise ConnectionError(f"USB node {node_id} is not connected")
            try:
                port.write(command.to_wire())
                port.flush()
                if command.command == CommandName.GET_STATUS:
                    self._status_cache[node_id] = self._read_status(node_id, port)
            except Exception as exc:
                if _is_serial_io_error(exc):
                    logger.warning("USB write failed for node %s (dropping port): %s", node_id, exc)
                    self._drop_port_unlocked(node_id)
                    raise ConnectionError(f"USB node {node_id} disconnected") from exc
                raise

    def poll_status(self, node_id: int) -> MirrorStatus:
        node_id = int(node_id)
        try:
            with self._lock:
                if node_id in self._status_cache:
                    return self._status_cache.pop(node_id)
                need_reconnect = node_id not in self._ports

            if need_reconnect and not self._try_reconnect(node_id):
                return MirrorStatus(node_id=node_id, mode="disconnected")

            with self._lock:
                port = self._ports.get(node_id)
                if port is None:
                    return MirrorStatus(node_id=node_id, mode="disconnected")
                try:
                    port.write(
                        MirrorCommand(node_id=node_id, command=CommandName.GET_STATUS).to_wire()
                    )
                    port.flush()
                    return self._read_status(node_id, port)
                except Exception as exc:
                    if _is_serial_io_error(exc):
                        logger.warning(
                            "USB poll failed for node %s (dropping port): %s", node_id, exc
                        )
                        self._drop_port_unlocked(node_id)
                        return MirrorStatus(node_id=node_id, mode="disconnected")
                    raise
        except Exception as exc:
            # Never let serial faults bubble into FastAPI /api/state.
            if _is_serial_io_error(exc):
                logger.warning("USB poll_status error for node %s: %s", node_id, exc)
                with self._lock:
                    self._drop_port_unlocked(node_id)
                return MirrorStatus(node_id=node_id, mode="disconnected")
            raise

    def reset_node(self, node_id: int) -> None:
        """Pulse DTR then reopen the CDC port so TinyUSB comes back clean."""
        node_id = int(node_id)
        with self._lock:
            endpoint = self._resolve_endpoint(node_id) or self._endpoints.get(node_id)
            if endpoint is None:
                raise ConnectionError(f"USB node {node_id} has no serial endpoint to reset")

            port = self._ports.get(node_id)
            if port is not None:
                try:
                    # Classic Arduino-style DTR pulse. Keep it brief — never leave DTR
                    # stuck low after reopen (that breaks Nano ESP32 TinyUSB CDC RX).
                    port.dtr = False
                    time.sleep(0.1)
                    port.dtr = True
                    time.sleep(0.05)
                except Exception as exc:
                    logger.info("DTR pulse on open port failed (will reopen): %s", exc)
                self._drop_port_unlocked(node_id)
                do_external_pulse = False
            else:
                do_external_pulse = True

        if do_external_pulse:
            try:
                import serial

                pulse = serial.Serial(
                    endpoint,
                    baudrate=self._config.usb_baudrate,
                    timeout=0.2,
                    write_timeout=1.0,
                    dsrdtr=False,
                    rtscts=False,
                )
                try:
                    pulse.dtr = False
                    time.sleep(0.1)
                    pulse.dtr = True
                    time.sleep(0.05)
                finally:
                    pulse.close()
            except Exception as exc:
                logger.info("DTR pulse open failed for %s: %s (will try reopen)", endpoint, exc)

        # CDC reopen itself also resets Nano ESP32; settle for firmware boot.
        time.sleep(0.3)
        try:
            port = self._open_port(node_id, endpoint)
        except Exception as exc:
            with self._lock:
                self._drop_port_unlocked(node_id)
            raise ConnectionError(f"USB node {node_id} reset reopen failed: {exc}") from exc

        with self._lock:
            self._drop_port_unlocked(node_id)
            self._ports[node_id] = port
            self._endpoints[node_id] = endpoint
            self._known_nodes.add(node_id)
            self._last_reconnect_mono[node_id] = time.monotonic()

    def _read_status(self, node_id: int, port) -> MirrorStatus:
        # Must be called with self._lock held.
        deadline = time.monotonic() + 1.5
        discarded = 0
        while time.monotonic() < deadline:
            try:
                raw = port.readline()
            except Exception as exc:
                if _is_serial_io_error(exc):
                    logger.warning("USB read failed for node %s: %s", node_id, exc)
                    self._drop_port_unlocked(node_id)
                    return MirrorStatus(node_id=node_id, mode="disconnected")
                raise
            if not raw:
                continue
            try:
                return MirrorStatus.from_wire(raw.strip())
            except (ValueError, KeyError, TypeError, UnicodeDecodeError):
                discarded += 1
                continue
        if discarded:
            logger.warning(
                "USB node %s: get_status timed out after discarding %d non-status line(s)",
                node_id,
                discarded,
            )
        else:
            logger.warning("USB node %s: get_status timed out (no serial data)", node_id)
        return MirrorStatus(node_id=node_id, mode="timeout")

    def close(self) -> None:
        with self._lock:
            for node_id in list(self._ports):
                self._drop_port_unlocked(node_id)
            self._ports.clear()
            self._status_cache.clear()


def _serial_device_present(path: str) -> bool:
    """Return True if ``path`` exists as a serial device node (or Windows COM name)."""
    import os

    if not path:
        return False
    # Windows COM ports are not filesystem paths.
    if path.upper().startswith("COM") and path[3:].isdigit():
        return True
    return os.path.exists(path)


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

    def reset_node(self, node_id: int) -> None:
        # Simulation has no USB/DTR line; treat as a successful no-op.
        if int(node_id) not in self._nodes:
            raise KeyError(f"sim node {node_id} is not registered")


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
