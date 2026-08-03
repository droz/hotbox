from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
import json
import struct
from typing import Any


class CommandName(StrEnum):
    DISCOVER = "discover"
    HOME = "home"
    STOP = "stop"
    SET_TARGET = "set_target"
    GET_STATUS = "get_status"
    CLEAR_ERROR = "clear_error"


class CommandId(IntEnum):
    HOME = 1
    STOP = 2
    SET_TARGET = 3
    GET_STATUS = 4
    CLEAR_ERROR = 6


_COMMAND_TO_ID = {
    CommandName.HOME: CommandId.HOME,
    CommandName.STOP: CommandId.STOP,
    CommandName.SET_TARGET: CommandId.SET_TARGET,
    CommandName.GET_STATUS: CommandId.GET_STATUS,
    CommandName.CLEAR_ERROR: CommandId.CLEAR_ERROR,
}
_ID_TO_COMMAND = {value: name for name, value in _COMMAND_TO_ID.items()}

CAN_CMD_BASE_ID = 0x100
CAN_RSP_BASE_ID = 0x200

# Firmware axis/status modes (not host supervisor track/park/jog).
FIRMWARE_MODE_IDS = {"idle": 0, "homing": 1, "position": 2, "fault": 3}
FIRMWARE_MODE_NAMES = {value: name for name, value in FIRMWARE_MODE_IDS.items()}


@dataclass(slots=True)
class MirrorCommand:
    node_id: int
    command: CommandName
    payload: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> bytes:
        return (json.dumps({"node_id": self.node_id, "command": self.command, "payload": self.payload}) + "\n").encode("utf-8")

    @classmethod
    def from_wire(cls, data: bytes) -> MirrorCommand:
        raw = json.loads(data.decode("utf-8"))
        return cls(node_id=int(raw["node_id"]), command=CommandName(raw["command"]), payload=dict(raw.get("payload", {})))

    def to_can_frame(self) -> bytes:
        cmd_id = int(_COMMAND_TO_ID[self.command])
        if self.command == CommandName.SET_TARGET:
            az = int(round(float(self.payload.get("azimuth_deg", 0.0)) * 100.0))
            el = int(round(float(self.payload.get("elevation_deg", 0.0)) * 100.0))
            return struct.pack("<Bhh", cmd_id, az, el)
        return struct.pack("<B", cmd_id)

    @classmethod
    def from_can_frame(cls, node_id: int, data: bytes) -> MirrorCommand:
        if not data:
            raise ValueError("empty CAN frame")
        cmd_id = CommandId(data[0])
        command = _ID_TO_COMMAND[cmd_id]
        payload: dict[str, Any] = {}
        if command == CommandName.SET_TARGET and len(data) >= 5:
            az, el = struct.unpack("<hh", data[1:5])
            payload = {"azimuth_deg": az / 100.0, "elevation_deg": el / 100.0}
        return cls(node_id=node_id, command=command, payload=payload)


@dataclass(slots=True)
class MirrorStatus:
    node_id: int
    homed: bool = False
    fault: str | None = None
    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0
    mode: str = "idle"

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "homed": self.homed,
            "fault": self.fault,
            "azimuth_deg": self.azimuth_deg,
            "elevation_deg": self.elevation_deg,
            "mode": self.mode,
        }

    def to_wire(self) -> bytes:
        return (json.dumps({"node_id": self.node_id, "type": "status", **self.as_dict()}) + "\n").encode("utf-8")

    @classmethod
    def from_wire(cls, data: bytes) -> MirrorStatus:
        raw = json.loads(data.decode("utf-8"))
        if raw.get("type") not in (None, "status"):
            raise ValueError(f"unexpected response type: {raw.get('type')}")
        return cls(
            node_id=int(raw["node_id"]),
            homed=bool(raw.get("homed", False)),
            fault=raw.get("fault"),
            azimuth_deg=float(raw.get("azimuth_deg", 0.0)),
            elevation_deg=float(raw.get("elevation_deg", 0.0)),
            mode=str(raw.get("mode", "idle")),
        )

    def to_can_frame(self) -> bytes:
        az = int(round(self.azimuth_deg * 100.0))
        el = int(round(self.elevation_deg * 100.0))
        return struct.pack(
            "<BBhhB",
            int(CommandId.GET_STATUS),
            1 if self.homed else 0,
            az,
            el,
            FIRMWARE_MODE_IDS.get(self.mode, 0),
        )

    @classmethod
    def from_can_frame(cls, node_id: int, data: bytes) -> MirrorStatus:
        if len(data) < 8:
            raise ValueError("status CAN frame too short")
        _, homed, az, el, mode_id = struct.unpack("<BBhhB", data[:8])
        return cls(
            node_id=node_id,
            homed=bool(homed),
            azimuth_deg=az / 100.0,
            elevation_deg=el / 100.0,
            mode=FIRMWARE_MODE_NAMES.get(mode_id, "idle"),
        )
