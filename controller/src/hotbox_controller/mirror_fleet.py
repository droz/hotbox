from __future__ import annotations

from dataclasses import dataclass, field

from .protocol import CommandName, MirrorCommand, MirrorStatus
from .tracking import TrackingTarget
from .transport import DiscoveredNode, MirrorTransport


@dataclass(slots=True)
class MirrorNode:
    node_id: int
    endpoint: str
    transport_name: str
    status: MirrorStatus = field(default_factory=lambda: MirrorStatus(node_id=-1))


class MirrorFleet:
    def __init__(self, transport: MirrorTransport) -> None:
        self._transport = transport
        self._nodes: dict[int, MirrorNode] = {}

    def discover(self) -> dict[int, MirrorNode]:
        """Merge transport discoveries into the fleet.

        Previously-seen nodes are kept when a rediscover finds nothing (USB
        hot-unplug) so the UI keeps cards and reconnect can restore them.
        """
        discovered = list(self._transport.discover())
        for node in discovered:
            existing = self._nodes.get(node.node_id)
            if existing is not None:
                existing.endpoint = node.endpoint
                existing.transport_name = node.transport_name
            else:
                self._nodes[node.node_id] = MirrorNode(
                    node_id=node.node_id,
                    endpoint=node.endpoint,
                    transport_name=node.transport_name,
                    status=MirrorStatus(node_id=node.node_id),
                )
        return self._nodes

    def nodes(self) -> dict[int, MirrorNode]:
        return dict(self._nodes)

    def home_all(self) -> None:
        for node_id in self._nodes:
            self.home(node_id)

    def home(self, node_id: int) -> None:
        self._transport.send(MirrorCommand(node_id=node_id, command=CommandName.HOME))

    def stop(self, node_id: int) -> None:
        self._transport.send(MirrorCommand(node_id=node_id, command=CommandName.STOP))

    def start(self, node_id: int) -> None:
        self._transport.send(MirrorCommand(node_id=node_id, command=CommandName.START))

    def poll(self) -> dict[int, MirrorStatus]:
        out: dict[int, MirrorStatus] = {}
        for node_id in self._nodes:
            status = self._transport.poll_status(node_id)
            self._nodes[node_id].status = status
            out[node_id] = status
        return out

    def apply_targets(self, targets: dict[int, TrackingTarget], *, start: bool = True) -> None:
        for node_id, target in targets.items():
            self._transport.send(
                MirrorCommand(
                    node_id=node_id,
                    command=CommandName.SET_TARGET,
                    payload={
                        "azimuth_deg": target.azimuth_deg,
                        "elevation_deg": target.elevation_deg,
                    },
                )
            )
            if start:
                self._transport.send(MirrorCommand(node_id=node_id, command=CommandName.START))

    def hold_current(self, node_id: int, *, start: bool = False) -> None:
        """set_target with hold_current; optionally engage position PID."""
        node_id = int(node_id)
        self._transport.send(
            MirrorCommand(
                node_id=node_id,
                command=CommandName.SET_TARGET,
                payload={"hold_current": True},
            )
        )
        if start:
            self._transport.send(MirrorCommand(node_id=node_id, command=CommandName.START))

    def apply_velocities(self, rates: dict[int, tuple[float, float]]) -> None:
        for node_id, (az_rate, el_rate) in rates.items():
            self._transport.send(
                MirrorCommand(
                    node_id=int(node_id),
                    command=CommandName.SET_VELOCITY,
                    payload={
                        "azimuth_deg_s": float(az_rate),
                        "elevation_deg_s": float(el_rate),
                    },
                )
            )
