"""Discover Arduino Nano ESP32 USB serial ports by VID/PID."""

from __future__ import annotations

import logging
import os
import re
from typing import Iterable

logger = logging.getLogger(__name__)

# Arduino Nano ESP32 CDC (running Arduino-core firmware).
# Boards can also briefly appear as Espressif 303A:1001 in boot/JTAG mode —
# we accept that pair only when the description looks like a Nano ESP32.
_ARDUINO_NANO_ESP32 = (0x2341, 0x0070)
_ESPRESSIF_USB_JTAG = (0x303A, 0x1001)

_DESCRIPTION_HINTS = ("nano esp32", "arduino nano esp32")


def parse_usb_ports_spec(raw: str) -> dict[int, str]:
    """Parse ``node_id:port`` entries separated by commas or whitespace."""
    text = raw.strip()
    if not text:
        return {}
    ports: dict[int, str] = {}
    for chunk in re.split(r"[\s,]+", text):
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"invalid USB port mapping {chunk!r}; want node_id:path")
        node_s, path = chunk.split(":", 1)
        if not path:
            raise ValueError(f"invalid USB port mapping {chunk!r}; empty path")
        ports[int(node_s)] = path
    return ports


def usb_ports_from_env(env: dict[str, str] | None = None) -> dict[int, str]:
    """Read ``HOTBOX_USB_PORTS`` (e.g. ``0:/dev/cu.usbmodem1101,1:/dev/ttyACM1``)."""
    source = os.environ if env is None else env
    raw = source.get("HOTBOX_USB_PORTS", "").strip()
    if not raw:
        return {}
    return parse_usb_ports_spec(raw)


def _info_matches_nano_esp32(*, vid: int | None, pid: int | None, description: str) -> bool:
    pair = (int(vid or 0), int(pid or 0))
    if pair == _ARDUINO_NANO_ESP32:
        return True
    desc = description.lower()
    if pair == _ESPRESSIF_USB_JTAG and any(hint in desc for hint in _DESCRIPTION_HINTS):
        return True
    return any(hint in desc for hint in _DESCRIPTION_HINTS)


def list_nano_esp32_ports(comports: Iterable | None = None) -> list[str]:
    """Return device paths for likely Arduino Nano ESP32 CDC ports."""
    if comports is None:
        from serial.tools import list_ports

        infos = list(list_ports.comports())
    else:
        infos = list(comports)

    found: list[str] = []
    for info in infos:
        device = getattr(info, "device", None)
        if not device:
            continue
        if _info_matches_nano_esp32(
            vid=getattr(info, "vid", None),
            pid=getattr(info, "pid", None),
            description=str(getattr(info, "description", "") or ""),
        ):
            found.append(str(device))
    return sorted(set(found))


def resolve_usb_port_map(
    explicit: dict[int, str] | None = None,
    *,
    comports: Iterable | None = None,
) -> dict[int, str]:
    """
    Resolve ``node_id → serial device``.

    Preference order:
    1. Explicit mapping (config / ``HOTBOX_USB_PORTS``)
    2. Exactly one VID/PID match → ``{0: device}`` for single-board bring-up
    3. Otherwise empty (log candidates; require explicit mapping)
    """
    if explicit:
        return {int(node_id): str(path) for node_id, path in explicit.items()}

    candidates = list_nano_esp32_ports(comports)
    if len(candidates) == 1:
        device = candidates[0]
        logger.info("auto-discovered Nano ESP32 on %s → node_id 0", device)
        return {0: device}
    if not candidates:
        logger.warning("no Arduino Nano ESP32 USB serial ports found")
        return {}
    logger.warning(
        "found %d Nano ESP32 USB ports %s; set HOTBOX_USB_PORTS=node_id:path,... "
        "(VID/PID discovery cannot tell which board is which node)",
        len(candidates),
        candidates,
    )
    return {}
