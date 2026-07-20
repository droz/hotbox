from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel

from .calibration import load_calibrations
from .config import AppConfig, SiteConfig, app_config_from_system
from .gps import GpsService
from .mirror_fleet import MirrorFleet
from .protocol import CommandName, MirrorCommand
from .scene import build_estimated_scene, build_mirror_scene_entry, default_mount_world, mount_world_from_calibration
from .sun import SunService, SunVector
from .tracking import TrackingTarget, safe_park, track_absorber
from .transport import MirrorTransport, build_transport


class JogRequest(BaseModel):
    node_id: int
    azimuth_rate_deg_s: float = 0.0
    elevation_rate_deg_s: float = 0.0


class TargetRequest(BaseModel):
    node_id: int
    azimuth_deg: float
    elevation_deg: float
    mode: str = "tracking"


class ControllerApplication:
    def __init__(
        self,
        config: AppConfig | None = None,
        transport: MirrorTransport | None = None,
    ) -> None:
        self.config = config or app_config_from_system()
        self.gps = GpsService(self.config.site, self.config.gps)
        self.sun = SunService(self.config.site)
        self.transport = transport or build_transport(self.config.transport)
        self.fleet = MirrorFleet(self.transport)
        self.calibrations = load_calibrations(self.config.calibration_path)
        if self.config.system is not None:
            self.absorber_world = self.config.system.absorber.center_world.copy()
        else:
            self.absorber_world = np.array([0.0, 0.0, self.config.oven.absorber_height_m], dtype=float)
        self._true_geometry: dict[str, Any] | None = None
        self.mode = "auto"
        self.fastapi = self._build_fastapi()

    def _mirror_world_for_node(self, node_id: int) -> np.ndarray:
        calibration = self.calibrations.get(node_id)
        if calibration is not None:
            return mount_world_from_calibration(calibration)
        if self.config.system is not None:
            try:
                return self.config.system.fleet.mount_by_id(node_id).mount_world()
            except KeyError:
                pass
        return default_mount_world(
            node_id,
            self.config.oven.absorber_height_m,
            self.config.mirror.default_oa_distance_m,
        )

    def startup(self) -> None:
        self.fleet.discover()

    def set_true_geometry(self, geometry: dict[str, Any] | None) -> None:
        self._true_geometry = geometry

    def control_tick(self) -> None:
        if self.mode != "auto":
            return
        fix = self.gps.current_fix()
        if fix.valid:
            self.sun = SunService(
                SiteConfig(
                    latitude_deg=fix.latitude_deg,
                    longitude_deg=fix.longitude_deg,
                    altitude_m=fix.altitude_m,
                )
            )
        sun = self.sun.sun_vector(fix.when_utc)
        statuses = self.fleet.poll()
        for node_id in self.fleet.nodes():
            if not statuses[node_id].homed:
                continue
            mirror_world = self._mirror_world_for_node(node_id)
            target = track_absorber(sun, mirror_world, self.absorber_world)
            self.fleet.apply_targets({node_id: target})

    def current_snapshot(self) -> dict[str, Any]:
        fix = self.gps.current_fix()
        if fix.valid:
            self.sun = SunService(
                SiteConfig(
                    latitude_deg=fix.latitude_deg,
                    longitude_deg=fix.longitude_deg,
                    altitude_m=fix.altitude_m,
                )
            )
        sun = self.sun.sun_vector(fix.when_utc)
        statuses = self.fleet.poll()

        targets: dict[int, dict[str, float | str]] = {}
        for node_id in self.fleet.nodes():
            mirror_world = self._mirror_world_for_node(node_id)
            target = (
                track_absorber(sun, mirror_world, self.absorber_world)
                if statuses[node_id].homed
                else safe_park(self.config.oven)
            )
            targets[node_id] = asdict(target)

        estimated = build_estimated_scene(
            sun=sun,
            absorber_world=self.absorber_world,
            statuses=statuses,
            calibrations=self.calibrations,
            absorber_height_m=self.config.oven.absorber_height_m,
            default_oa_distance_m=self.config.mirror.default_oa_distance_m,
            default_mirror_offset_d_m=self.config.mirror.mount_offset_d_m,
            system=self.config.system,
        )

        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": self.mode,
            "gps": fix.as_dict(),
            "sun": {
                "azimuth_deg": sun.azimuth_deg,
                "elevation_deg": sun.elevation_deg,
                "world_vector": sun.world_vector.tolist(),
            },
            "transport": self.config.transport.mode,
            "mirrors": {str(node_id): status.as_dict() for node_id, status in statuses.items()},
            "targets": {str(node_id): target for node_id, target in targets.items()},
            "calibration_count": len(self.calibrations),
            "geometry": {
                "estimated": estimated,
                "true": self._true_geometry,
            },
        }

    def home_all(self) -> None:
        self.fleet.home_all()

    def park_all(self) -> None:
        self.mode = "auto"
        target = safe_park(self.config.oven)
        self.fleet.apply_targets({node_id: target for node_id in self.fleet.nodes()})

    def set_manual_target(self, request: TargetRequest) -> None:
        self.mode = "manual"
        self.fleet.apply_targets(
            {
                request.node_id: TrackingTarget(
                    azimuth_deg=request.azimuth_deg,
                    elevation_deg=request.elevation_deg,
                    mode=request.mode,
                )
            }
        )

    def jog(self, request: JogRequest) -> None:
        self.mode = "manual"
        self.transport.send(
            MirrorCommand(
                node_id=request.node_id,
                command=CommandName.JOG,
                payload={
                    "azimuth_rate_deg_s": request.azimuth_rate_deg_s,
                    "elevation_rate_deg_s": request.elevation_rate_deg_s,
                },
            )
        )

    def _build_fastapi(self) -> FastAPI:
        static_dir = Path(__file__).parent / "web" / "static"
        app = FastAPI(title="Hot-Box Controller")
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/api/state")
        def state() -> dict[str, Any]:
            # Read-only: the SITL/control loop owns command issuance.
            return self.current_snapshot()

        @app.post("/api/home")
        def home() -> dict[str, str]:
            self.home_all()
            return {"status": "ok"}

        @app.post("/api/park")
        def park() -> dict[str, str]:
            self.park_all()
            return {"status": "ok"}

        @app.post("/api/auto")
        def auto() -> dict[str, str]:
            self.mode = "auto"
            return {"status": "ok"}

        @app.post("/api/jog")
        def jog(request: JogRequest) -> dict[str, str]:
            self.jog(request)
            return {"status": "ok"}

        @app.post("/api/target")
        def target(request: TargetRequest) -> dict[str, str]:
            self.set_manual_target(request)
            return {"status": "ok"}

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(static_dir / "index.html")

        return app


def build_true_geometry_from_layouts(
    *,
    sun: SunVector,
    absorber_world: np.ndarray,
    layouts: dict[int, Any],
    statuses: dict[int, Any],
    mirror_offset_d_m: float = 0.2,
) -> dict[str, Any]:
    mirrors = []
    for node_id, layout in sorted(layouts.items()):
        status = statuses[node_id]
        mirrors.append(
            build_mirror_scene_entry(
                node_id=node_id,
                mount_world=layout.mount_world,
                azimuth_deg=status.azimuth_deg,
                elevation_deg=status.elevation_deg,
                mirror_offset_d_m=mirror_offset_d_m,
                sun=sun,
                absorber_world=absorber_world,
            )
        )
    return {
        "label": "true",
        "absorber": {"center": np.asarray(absorber_world, dtype=float).reshape(3).tolist()},
        "sun": {
            "azimuth_deg": sun.azimuth_deg,
            "elevation_deg": sun.elevation_deg,
            "world_vector": sun.world_vector.tolist(),
            "display_position": (np.asarray(sun.world_vector, dtype=float).reshape(3) * 8.0).tolist(),
        },
        "mirrors": mirrors,
    }
