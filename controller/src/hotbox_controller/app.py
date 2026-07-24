from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from hotbox_shared import utc_now
from pydantic import BaseModel

from .calibration import load_calibrations
from .config import AppConfig, SiteConfig, app_config_from_system
from .gps import GpsService
from .mirror_fleet import MirrorFleet
from .protocol import CommandName, MirrorCommand
from .scene import build_mirror_scene_entry, build_target_scene, default_mount_world, mount_world_from_calibration
from .sun import SunService, SunVector
from .tracking import TrackingTarget, idle_dump_world, safe_park, track_point
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


class ModeRequest(BaseModel):
    """Supervisor mode: track | park | jog (``manual`` / ``auto`` accepted as aliases)."""

    mode: str


class MirrorModeRequest(BaseModel):
    node_id: int
    mode: str


class HeatDemandRequest(BaseModel):
    """Simulated oven heat-demand relay (GPIO later)."""

    enabled: bool


# Canonical modes. ``auto`` → track, ``manual`` → jog.
SUPERVISOR_MODES = frozenset({"track", "park", "jog"})


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
        # Simulated oven heat-demand relay. When False, Track mirrors park (product behavior).
        self.heat_demand = True
        # Per-mirror supervisor mode (track|park|jog). Fleet switcher sets all of these.
        self._node_modes: dict[int, str] = {}
        self.fastapi = self._build_fastapi()

    def node_mode(self, node_id: int) -> str:
        return self._node_modes.get(int(node_id), "track")

    @property
    def mode(self) -> str:
        """Fleet-wide mode when unanimous; otherwise ``mixed``."""
        modes = {self.node_mode(node_id) for node_id in self.fleet.nodes()}
        if not modes:
            return "track"
        if len(modes) == 1:
            return next(iter(modes))
        return "mixed"

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
        for node_id in self.fleet.nodes():
            self._node_modes.setdefault(int(node_id), "track")

    def set_true_geometry(self, geometry: dict[str, Any] | None) -> None:
        self._true_geometry = geometry

    def set_heat_demand(self, enabled: bool) -> None:
        """Simulate the oven heat-demand relay (True = oven wants power)."""
        self.heat_demand = bool(enabled)

    def _tracking_kwargs(self) -> dict[str, float | int | bool | object]:
        mirror = self.config.mirror
        solve_for_mount_offset = True
        joint_limits = None
        if self.config.system is not None:
            solve_for_mount_offset = bool(self.config.system.control.solve_for_mount_offset)
            joint_limits = self.config.system.control.mount_joint_limits()
        return {
            "grid_nx": mirror.grid_nx,
            "grid_ny": mirror.grid_ny,
            "pitch_m": mirror.pitch_m,
            "radius_of_curvature_m": mirror.radius_of_curvature_m,
            "mount_offset_d_m": mirror.mount_offset_d_m,
            "solve_for_mount_offset": solve_for_mount_offset,
            "joint_limits": joint_limits,
        }

    def _tracking_targets(
        self,
        sun: SunVector,
        statuses: dict[int, Any],
        *,
        target_world: np.ndarray | None = None,
    ) -> dict[int, TrackingTarget]:
        aim = (
            np.asarray(self.absorber_world, dtype=float).reshape(3)
            if target_world is None
            else np.asarray(target_world, dtype=float).reshape(3)
        )
        kwargs = self._tracking_kwargs()
        targets: dict[int, TrackingTarget] = {}
        for node_id in self.fleet.nodes():
            if statuses[node_id].homed:
                mirror_world = self._mirror_world_for_node(node_id)
                targets[node_id] = track_point(sun, mirror_world, aim, **kwargs)
            else:
                targets[node_id] = safe_park(self.config.oven)
        return targets

    @staticmethod
    def normalize_supervisor_mode(mode: str) -> str:
        key = str(mode).strip().lower()
        if key == "auto":
            return "track"
        if key == "manual":
            return "jog"
        if key not in SUPERVISOR_MODES:
            raise ValueError(f"unsupported supervisor mode: {mode!r} (want track|park|jog)")
        return key

    def set_mode(self, mode: str) -> None:
        """Set Track/Park/Jog on every discovered mirror."""
        normalized = self.normalize_supervisor_mode(mode)
        for node_id in self.fleet.nodes():
            self._set_node_mode(int(node_id), normalized, apply_immediate=False)
        if normalized == "park":
            self._apply_park_all()

    def set_mirror_mode(self, node_id: int, mode: str) -> None:
        """Set Track/Park/Jog for one mirror."""
        self._set_node_mode(int(node_id), self.normalize_supervisor_mode(mode), apply_immediate=True)

    def _set_node_mode(self, node_id: int, mode: str, *, apply_immediate: bool) -> None:
        previous = self._node_modes.get(node_id)
        self._node_modes[node_id] = mode
        if not apply_immediate:
            return
        if mode == "park":
            self.fleet.apply_targets({node_id: safe_park(self.config.oven)})
        elif mode == "jog" and previous != "jog":
            self.fleet.stop(node_id)

    def _apply_park_all(self) -> None:
        target = safe_park(self.config.oven)
        self.fleet.apply_targets({node_id: target for node_id in self.fleet.nodes()})

    def _track_aim_point(self) -> np.ndarray:
        """Absorber center when heat is demanded; otherwise the idle dump above it."""
        if self.heat_demand:
            return np.asarray(self.absorber_world, dtype=float).reshape(3)
        return idle_dump_world(
            self.absorber_world,
            self.config.oven.idle_aim_height_above_absorber_m,
        )

    def _desired_target_for_node(
        self,
        node_id: int,
        *,
        sun: SunVector,
        statuses: dict[int, Any],
        tracking: dict[int, TrackingTarget],
    ) -> TrackingTarget | None:
        """
        Closed-loop command for one mirror, or None when Jog (operator owns the axes).

        Track + heat demand → aim at absorber. Track without demand → aim above absorber.
        Park → face-up stow (az/el from config, default 0°/0°).
        """
        mode = self.node_mode(node_id)
        if mode == "jog":
            return None
        if mode == "park":
            return safe_park(self.config.oven)
        return tracking[node_id]

    def _command_targets(
        self,
        sun: SunVector,
        statuses: dict[int, Any],
    ) -> dict[int, TrackingTarget]:
        tracking = self._tracking_targets(sun, statuses, target_world=self._track_aim_point())
        out: dict[int, TrackingTarget] = {}
        for node_id in self.fleet.nodes():
            desired = self._desired_target_for_node(
                node_id, sun=sun, statuses=statuses, tracking=tracking
            )
            if desired is None:
                # Jog: keep last computed tracking pose for scene preview only.
                out[node_id] = tracking[node_id]
            else:
                out[node_id] = desired
        return out

    def control_tick(self) -> None:
        fix = self.gps.current_fix()
        if fix.valid:
            self.sun = SunService(
                SiteConfig(
                    latitude_deg=fix.latitude_deg,
                    longitude_deg=fix.longitude_deg,
                    altitude_m=fix.altitude_m,
                    timezone_id=self.config.site.timezone_id,
                )
            )
        sun = self.sun.sun_vector(fix.when_utc)
        statuses = self.fleet.poll()
        tracking = self._tracking_targets(sun, statuses, target_world=self._track_aim_point())
        for node_id in self.fleet.nodes():
            if not statuses[node_id].homed:
                continue
            desired = self._desired_target_for_node(
                node_id, sun=sun, statuses=statuses, tracking=tracking
            )
            if desired is None:
                continue
            self.fleet.apply_targets({node_id: desired})

    def current_snapshot(self) -> dict[str, Any]:
        fix = self.gps.current_fix()
        if fix.valid:
            self.sun = SunService(
                SiteConfig(
                    latitude_deg=fix.latitude_deg,
                    longitude_deg=fix.longitude_deg,
                    altitude_m=fix.altitude_m,
                    timezone_id=self.config.site.timezone_id,
                )
            )
        sun = self.sun.sun_vector(fix.when_utc)
        statuses = self.fleet.poll()
        targets = self._command_targets(sun, statuses)
        mirror_modes = {str(node_id): self.node_mode(node_id) for node_id in self.fleet.nodes()}

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
            "timestamp_utc": utc_now().isoformat(),
            "mode": self.mode,
            "heat_demand": self.heat_demand,
            "mirror_modes": mirror_modes,
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
        self.set_mode("jog")
        self.fleet.home_all()

    def home_one(self, node_id: int) -> None:
        self.set_mirror_mode(node_id, "jog")
        self.fleet.home(node_id)

    def stop_one(self, node_id: int) -> None:
        self.set_mirror_mode(node_id, "jog")
        self.fleet.stop(node_id)

    def park_all(self) -> None:
        self.set_mode("park")

    def park_one(self, node_id: int) -> None:
        self.set_mirror_mode(node_id, "park")

    def set_manual_target(self, request: TargetRequest) -> None:
        self.set_mirror_mode(request.node_id, "jog")
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
        self._node_modes[int(request.node_id)] = "jog"
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
        def park() -> dict[str, Any]:
            self.park_all()
            return {"status": "ok", "mode": self.mode}

        @app.post("/api/park_one")
        def api_park_one(request: NodeRequest) -> dict[str, Any]:
            self.park_one(request.node_id)
            return {"status": "ok", "mode": self.mode, "mirror_mode": self.node_mode(request.node_id)}

        @app.post("/api/mode")
        def api_mode(request: ModeRequest) -> dict[str, Any]:
            try:
                self.set_mode(request.mode)
            except ValueError as exc:
                from fastapi import HTTPException

                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {"status": "ok", "mode": self.mode}

        @app.post("/api/mirror_mode")
        def api_mirror_mode(request: MirrorModeRequest) -> dict[str, Any]:
            try:
                self.set_mirror_mode(request.node_id, request.mode)
            except ValueError as exc:
                from fastapi import HTTPException

                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {
                "status": "ok",
                "mode": self.mode,
                "node_id": request.node_id,
                "mirror_mode": self.node_mode(request.node_id),
            }

        @app.post("/api/heat_demand")
        def api_heat_demand(request: HeatDemandRequest) -> dict[str, Any]:
            self.set_heat_demand(request.enabled)
            return {"status": "ok", "heat_demand": self.heat_demand}

        @app.post("/api/auto")
        def auto() -> dict[str, Any]:
            """Resume sun tracking (alias for ``POST /api/mode`` with ``track``)."""
            self.set_mode("track")
            return {"status": "ok", "mode": self.mode}

        @app.post("/api/manual")
        def api_manual() -> dict[str, Any]:
            """Enter jog mode on all mirrors (alias for ``POST /api/mode`` with ``jog``)."""
            self.set_mode("jog")
            return {"status": "ok", "mode": self.mode}

        @app.post("/api/jog")
        def api_jog(request: JogRequest) -> dict[str, Any]:
            self.jog(request)
            return {"status": "ok", "mode": self.mode, "mirror_mode": self.node_mode(request.node_id)}

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
