from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class SiteConfig:
    latitude_deg: float = 40.7864
    longitude_deg: float = -119.2065
    altitude_m: float = 1190.0


@dataclass(slots=True)
class OvenConfig:
    absorber_height_m: float = 1.0
    safe_park_elevation_deg: float = 75.0
    safe_park_azimuth_deg: float = 180.0


@dataclass(slots=True)
class TransportConfig:
    mode: str = "usb"
    can_channel: str = "can0"
    can_bitrate: int = 250000
    usb_ports: dict[int, str] = field(default_factory=dict)
    usb_baudrate: int = 115200
    sim_node_ids: tuple[int, ...] = (0, 1, 2)


@dataclass(slots=True)
class GpsConfig:
    enabled: bool = False
    port: str = "/dev/ttyAMA0"
    baudrate: int = 9600
    stale_after_s: float = 5.0


@dataclass(slots=True)
class AppConfig:
    site: SiteConfig = field(default_factory=SiteConfig)
    oven: OvenConfig = field(default_factory=OvenConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    gps: GpsConfig = field(default_factory=GpsConfig)
    calibration_path: Path = Path("config/calibration.yaml")
    web_host: str = "0.0.0.0"
    web_port: int = 8000
