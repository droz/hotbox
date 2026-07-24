from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from hotbox_shared import SystemConstants, load_system_constants


@dataclass(slots=True)
class SiteConfig:
    latitude_deg: float = 40.7864
    longitude_deg: float = -119.2065
    altitude_m: float = 1190.0
    timezone_id: str = "America/Los_Angeles"


@dataclass(slots=True)
class OvenConfig:
    absorber_height_m: float = 1.0
    absorber_width_m: float = 0.40
    absorber_panel_height_m: float = 0.40
    safe_park_elevation_deg: float = 75.0
    safe_park_azimuth_deg: float = 180.0


@dataclass(slots=True)
class MirrorPlantConfig:
    grid_nx: int = 3
    grid_ny: int = 5
    tile_side_m: float = 0.254
    pitch_m: float = 0.26035
    mount_offset_d_m: float = 0.2
    radius_of_curvature_m: float = 5.5
    default_oa_distance_m: float = 2.5
    default_mount_height_m: float = 1.0


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
    mirror: MirrorPlantConfig = field(default_factory=MirrorPlantConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    gps: GpsConfig = field(default_factory=GpsConfig)
    system: SystemConstants | None = None
    calibration_path: Path = Path("config/calibration.yaml")
    web_host: str = "0.0.0.0"
    web_port: int = 8000


def app_config_from_system(system: SystemConstants | None = None) -> AppConfig:
    system = system or load_system_constants()
    node_ids = tuple(mount.node_id for mount in system.fleet.mounts)
    return AppConfig(
        site=SiteConfig(
            latitude_deg=system.default_site.latitude_deg,
            longitude_deg=system.default_site.longitude_deg,
            altitude_m=system.default_site.altitude_m,
            timezone_id=system.default_site.timezone_id,
        ),
        oven=OvenConfig(
            absorber_height_m=system.absorber.center_height_m,
            absorber_width_m=system.absorber.width_m,
            absorber_panel_height_m=system.absorber.height_m,
            safe_park_elevation_deg=system.control.safe_park_elevation_deg,
            safe_park_azimuth_deg=system.control.safe_park_azimuth_deg,
        ),
        mirror=MirrorPlantConfig(
            grid_nx=system.mirror.grid_nx,
            grid_ny=system.mirror.grid_ny,
            tile_side_m=system.mirror.tile_side_m,
            pitch_m=system.mirror.pitch_m,
            mount_offset_d_m=system.mirror.mount_offset_d_m,
            radius_of_curvature_m=system.mirror.radius_of_curvature_m,
            default_oa_distance_m=system.mirror.default_oa_distance_m,
            default_mount_height_m=system.mirror.default_mount_height_m,
        ),
        transport=TransportConfig(sim_node_ids=node_ids),
        system=system,
    )
