from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import threading
import time
from typing import Any

from hotbox_shared import utc_now

from .config import GpsConfig, SiteConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GpsFix:
    when_utc: datetime
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    valid: bool = True
    source: str = "fallback"
    satellites: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "when_utc": self.when_utc.isoformat(),
            "latitude_deg": self.latitude_deg,
            "longitude_deg": self.longitude_deg,
            "altitude_m": self.altitude_m,
            "valid": self.valid,
            "source": self.source,
            "satellites": self.satellites,
        }


def _nmea_lat_lon(value: str, hemi: str) -> float | None:
    if not value or not hemi:
        return None
    if "." not in value:
        return None
    degrees = int(value.split(".", 1)[0][:-2] or "0")
    minutes = float(value[len(str(degrees)) :])
    decimal = degrees + minutes / 60.0
    if hemi in {"S", "W"}:
        decimal = -decimal
    return decimal


def parse_nmea_sentence(sentence: str) -> dict[str, Any] | None:
    """Parse a GGA or RMC sentence into partial fix fields."""
    sentence = sentence.strip()
    if not sentence.startswith("$") or "*" not in sentence:
        return None
    body, checksum = sentence[1:].split("*", 1)
    expected = 0
    for ch in body:
        expected ^= ord(ch)
    try:
        if int(checksum[:2], 16) != expected:
            return None
    except ValueError:
        return None

    fields = body.split(",")
    talker = fields[0]
    out: dict[str, Any] = {}

    if talker.endswith("RMC") and len(fields) >= 10:
        if fields[2] != "A":
            return {"valid": False}
        lat = _nmea_lat_lon(fields[3], fields[4])
        lon = _nmea_lat_lon(fields[5], fields[6])
        if lat is None or lon is None:
            return None
        hhmmss = fields[1]
        ddmmyy = fields[9]
        when = _parse_nmea_datetime(ddmmyy, hhmmss)
        out.update({"valid": True, "latitude_deg": lat, "longitude_deg": lon, "when_utc": when})
        return out

    if talker.endswith("GGA") and len(fields) >= 10:
        quality = int(fields[6] or "0")
        lat = _nmea_lat_lon(fields[2], fields[3])
        lon = _nmea_lat_lon(fields[4], fields[5])
        if lat is None or lon is None:
            return {"valid": False}
        alt = float(fields[9] or "0")
        sats = int(fields[7] or "0")
        hhmmss = fields[1]
        when = _parse_nmea_time_only(hhmmss)
        out.update(
            {
                "valid": quality > 0,
                "latitude_deg": lat,
                "longitude_deg": lon,
                "altitude_m": alt,
                "satellites": sats,
                "when_utc": when,
            }
        )
        return out

    return None


def _parse_nmea_datetime(ddmmyy: str, hhmmss: str) -> datetime:
    day = int(ddmmyy[0:2])
    month = int(ddmmyy[2:4])
    year = 2000 + int(ddmmyy[4:6])
    hour = int(hhmmss[0:2])
    minute = int(hhmmss[2:4])
    second = int(float(hhmmss[4:]))
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


def _parse_nmea_time_only(hhmmss: str) -> datetime:
    now = utc_now()
    hour = int(hhmmss[0:2])
    minute = int(hhmmss[2:4])
    second = int(float(hhmmss[4:]))
    return now.replace(hour=hour, minute=minute, second=second, microsecond=0)


class GpsService:
    """Reads NEO-6M NMEA over serial, falling back to configured site coordinates."""

    def __init__(self, fallback_site: SiteConfig, gps_config: GpsConfig | None = None) -> None:
        self._fallback_site = fallback_site
        self._config = gps_config or GpsConfig()
        self._lock = threading.Lock()
        self._latest = GpsFix(
            when_utc=utc_now(),
            latitude_deg=fallback_site.latitude_deg,
            longitude_deg=fallback_site.longitude_deg,
            altitude_m=fallback_site.altitude_m,
            valid=False,
            source="fallback",
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if self._config.enabled and self._config.port:
            self._thread = threading.Thread(target=self._reader_loop, name="gps-reader", daemon=True)
            self._thread.start()

    def current_fix(self) -> GpsFix:
        with self._lock:
            fix = self._latest
            if not fix.valid or (utc_now() - fix.when_utc).total_seconds() > self._config.stale_after_s:
                return GpsFix(
                    when_utc=utc_now(),
                    latitude_deg=self._fallback_site.latitude_deg,
                    longitude_deg=self._fallback_site.longitude_deg,
                    altitude_m=self._fallback_site.altitude_m,
                    valid=False,
                    source="fallback",
                )
            return GpsFix(
                when_utc=fix.when_utc,
                latitude_deg=fix.latitude_deg,
                longitude_deg=fix.longitude_deg,
                altitude_m=fix.altitude_m,
                valid=fix.valid,
                source=fix.source,
                satellites=fix.satellites,
            )

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _reader_loop(self) -> None:
        while not self._stop.is_set():
            try:
                import serial

                with serial.Serial(self._config.port, baudrate=self._config.baudrate, timeout=1.0) as port:
                    logger.info("GPS connected on %s", self._config.port)
                    while not self._stop.is_set():
                        raw = port.readline()
                        if not raw:
                            continue
                        try:
                            line = raw.decode("ascii", errors="ignore")
                        except Exception:
                            continue
                        parsed = parse_nmea_sentence(line)
                        if not parsed:
                            continue
                        self._apply_partial(parsed)
            except Exception as exc:
                logger.warning("GPS reader error on %s: %s", self._config.port, exc)
                time.sleep(2.0)

    def _apply_partial(self, parsed: dict[str, Any]) -> None:
        with self._lock:
            current = self._latest
            when = parsed.get("when_utc", current.when_utc)
            if not isinstance(when, datetime):
                when = current.when_utc
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            self._latest = GpsFix(
                when_utc=when,
                latitude_deg=float(parsed.get("latitude_deg", current.latitude_deg)),
                longitude_deg=float(parsed.get("longitude_deg", current.longitude_deg)),
                altitude_m=float(parsed.get("altitude_m", current.altitude_m)),
                valid=bool(parsed.get("valid", False)),
                source="nmea",
                satellites=parsed.get("satellites", current.satellites),
            )
