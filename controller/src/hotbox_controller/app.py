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
from .scene import build_mirror_scene_entry, build_target_scene, default_mount_world, mount_world_from_calibration
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


class NodeRequest(BaseModel):
    node_id: int


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
                return self.config.system.mount_world(node_id)
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

    def _tracking_kwargs(self) -> dict[str, float | int | bool]:
        mirror = self.config.mirror
        solve_for_mount_offset = True
        if self.config.system is not None:
            solve_for_mount_offset = bool(self.config.system.control.solve_for_mount_offset)
        return {
            "grid_nx": mirror.grid_nx,
            "grid_ny": mirror.grid_ny,
            "pitch_m": mirror.pitch_m,
            "radius_of_curvature_m": mirror.radius_of_curvature_m,
            "mount_offset_d_m": mirror.mount_offset_d_m,
            "solve_for_mount_offset": solve_for_mount_offset,
        }

    def _tracking_targets(
        self,
        sun: SunVector,
        statuses: dict[int, Any],
    ) -> dict[int, TrackingTarget]:
        targets: dict[int, TrackingTarget] = {}
        for node_id in self.fleet.nodes():
            if statuses[node_id].homed:
                mirror_world = self._mirror_world_for_node(node_id)
                targets[node_id] = track_absorber(
                    sun, mirror_world, self.absorber_world, **self._tracking_kwargs()
                )
            else:
                targets[node_id] = safe_park(self.config.oven)
        return targets

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
        targets = self._tracking_targets(sun, statuses)
        for node_id, target in targets.items():
            if not statuses[node_id].homed:
                continue
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
        targets = self._tracking_targets(sun, statuses)

        target_scene = build_target_scene(
            sun=sun,
            absorber_world=self.absorber_world,
            targets=targets,
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
            "targets": {str(node_id): asdict(target) for node_id, target in targets.items()},
            "calibration_count": len(self.calibrations),
            "geometry": {
                "target": target_scene,
                "estimated": target_scene,
                "true": self._true_geometry,
            },
        }

    def home_all(self) -> None:
        self.fleet.home_all()

    def home_one(self, node_id: int) -> None:
        self.mode = "manual"
        self.fleet.home(node_id)

    def stop_one(self, node_id: int) -> None:
        self.mode = "manual"
        self.fleet.stop(node_id)

    def park_all(self) -> None:
        self.mode = "auto"
        target = safe_park(self.config.oven)
        self.fleet.apply_targets({node_id: target for node_id in self.fleet.nodes()})

    def park_one(self, node_id: int) -> None:
        self.mode = "manual"
        self.fleet.apply_targets({node_id: safe_park(self.config.oven)})

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

        @app.post("/api/home_one")
        def api_home_one(request: NodeRequest) -> dict[str, str]:
            self.home_one(request.node_id)
            return {"status": "ok"}

        @app.post("/api/stop_one")
        def api_stop_one(request: NodeRequest) -> dict[str, str]:
            self.stop_one(request.node_id)
            return {"status": "ok"}

        @app.post("/api/park")
        def park() -> dict[str, str]:
            self.park_all()
            return {"status": "ok"}

        @app.post("/api/park_one")
        def api_park_one(request: NodeRequest) -> dict[str, str]:
            self.park_one(request.node_id)
            return {"status": "ok"}

        @app.post("/api/auto")
        def auto() -> dict[str, str]:
            self.mode = "auto"
            return {"status": "ok"}

        @app.post("/api/jog")
        def api_jog(request: JogRequest) -> dict[str, str]:
            self.jog(request)
            return {"status": "ok"}

        @app.post("/api/target")
        def api_target(request: TargetRequest) -> dict[str, str]:
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
    system: Any | None = None,
) -> dict[str, Any]:
    params = {
        "grid_nx": int(getattr(getattr(system, "mirror", None), "grid_nx", 3)),
        "grid_ny": int(getattr(getattr(system, "mirror", None), "grid_ny", 5)),
        "pitch_m": float(getattr(getattr(system, "mirror", None), "pitch_m", 0.26035)),
        "tile_side_m": float(getattr(getattr(system, "mirror", None), "tile_side_m", 0.254)),
        "radius_of_curvature_m": float(getattr(getattr(system, "mirror", None), "radius_of_curvature_m", 5.5)),
    }
    mirrors = []
    mounts = []
    for node_id, layout in sorted(layouts.items()):
        status = statuses[node_id]
        mounts.append(layout.mount_world)
        mirrors.append(
            build_mirror_scene_entry(
                node_id=node_id,
                mount_world=layout.mount_world,
                azimuth_deg=status.azimuth_deg,
                elevation_deg=status.elevation_deg,
                mirror_offset_d_m=mirror_offset_d_m,
                sun=sun,
                absorber_world=absorber_world,
                grid_nx=params["grid_nx"],
                grid_ny=params["grid_ny"],
                pitch_m=params["pitch_m"],
                tile_side_m=params["tile_side_m"],
                radius_of_curvature_m=params["radius_of_curvature_m"],
            )
        )
    from .scene import build_oven_scene

    absorber_width = float(getattr(getattr(system, "absorber", None), "width_m", 0.4))
    absorber_height = float(getattr(getattr(system, "absorber", None), "height_m", 0.4))
    normal_deg = float(getattr(getattr(system, "absorber", None), "normal_angle_from_x_deg", 90.0))
    oven = build_oven_scene(
        absorber_center=absorber_world,
        absorber_width_m=absorber_width,
        absorber_height_m=absorber_height,
        normal_angle_from_x_deg=normal_deg,
        fleet_mounts=mounts,
    )
    sun_distance_m = 10.0
    sun_pos = np.asarray(sun.world_vector, dtype=float).reshape(3)
    sun_pos = sun_pos / max(float(np.linalg.norm(sun_pos)), 1e-12) * sun_distance_m
    return {
        "label": "true",
        "frame": {"x": "east", "y": "north", "z": "up"},
        "ground_z": 0.0,
        "absorber": {"center": np.asarray(absorber_world, dtype=float).reshape(3).tolist()},
        "oven": oven,
        "sun": {
            "azimuth_deg": sun.azimuth_deg,
            "elevation_deg": sun.elevation_deg,
            "world_vector": sun.world_vector.tolist(),
            "display_position": sun_pos.tolist(),
            "distance_m": sun_distance_m,
        },
        "mirrors": mirrors,
    }
