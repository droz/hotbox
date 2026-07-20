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
        discovered = list(self._transport.discover())
        self._nodes = {
            node.node_id: MirrorNode(node_id=node.node_id, endpoint=node.endpoint, transport_name=node.transport_name, status=MirrorStatus(node_id=node.node_id))
            for node in discovered
        }
        return self._nodes

    def nodes(self) -> dict[int, MirrorNode]:
        return dict(self._nodes)

    def home_all(self) -> None:
        for node_id in self._nodes:
            self._transport.send(MirrorCommand(node_id=node_id, command=CommandName.HOME))

    def poll(self) -> dict[int, MirrorStatus]:
        out: dict[int, MirrorStatus] = {}
        for node_id in self._nodes:
            status = self._transport.poll_status(node_id)
            self._nodes[node_id].status = status
            out[node_id] = status
        return out

    def apply_targets(self, targets: dict[int, TrackingTarget]) -> None:
        for node_id, target in targets.items():
            self._transport.send(
                MirrorCommand(
                    node_id=node_id,
                    command=CommandName.SET_TARGET,
                    payload={
                        "azimuth_deg": target.azimuth_deg,
                        "elevation_deg": target.elevation_deg,
                        "mode": target.mode,
                    },
                )
            )
