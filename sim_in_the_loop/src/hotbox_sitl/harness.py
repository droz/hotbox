from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import threading
import time
from typing import Any

import numpy as np
import uvicorn

from hotbox_controller.app import ControllerApplication, build_true_geometry_from_layouts
from hotbox_controller.config import TransportConfig, app_config_from_system
from hotbox_controller.geometry import MirrorCalibration
from hotbox_controller.protocol import CommandName, MirrorCommand
from hotbox_controller.sun import SunService
from hotbox_controller.transport import SimTransport
from hotbox_shared import SystemConstants, load_system_constants

from .mirror_node import SimulatedMirrorNode


@dataclass(slots=True)
class TrueMirrorLayout:
    node_id: int
    mount_world: np.ndarray
    facet_offset_world: np.ndarray
    mirror_offset_d_m: float
    focal_length_m: float


def _layouts_from_system(system: SystemConstants) -> dict[int, TrueMirrorLayout]:
    offset = np.array([0.0, 0.0, system.mirror.mount_offset_d_m], dtype=float)
    layouts: dict[int, TrueMirrorLayout] = {}
    for mount in system.fleet.mounts:
        layouts[mount.node_id] = TrueMirrorLayout(
            node_id=mount.node_id,
            mount_world=mount.mount_world(),
            facet_offset_world=offset.copy(),
            mirror_offset_d_m=system.mirror.mount_offset_d_m,
            focal_length_m=system.mirror.focal_length_m,
        )
    return layouts


def _calibration_from_layout(layout: TrueMirrorLayout, bearing_deg: float) -> MirrorCalibration:
    mount = np.asarray(layout.mount_world, dtype=float).reshape(3)
    return MirrorCalibration(
        node_id=layout.node_id,
        oa_bearing_deg=bearing_deg,
        oa_height_delta_m=float(mount[2]),
        home_azimuth_offset_deg=0.0,
        home_elevation_offset_deg=0.0,
        oa_distance_m=float(np.hypot(mount[0], mount[1])),
        mirror_offset_d_m=layout.mirror_offset_d_m,
        focal_length_m=layout.focal_length_m,
    )


class SitlHarness:
    def __init__(
        self,
        node_ids: tuple[int, ...] | None = None,
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        dt_s: float = 0.05,
        system: SystemConstants | None = None,
    ) -> None:
        self.system = system or load_system_constants()
        if node_ids is None:
            node_ids = tuple(mount.node_id for mount in self.system.fleet.mounts)
        self.host = host
        self.port = port
        self.dt_s = dt_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._latest: dict[str, Any] = {}
        self.nodes = {node_id: SimulatedMirrorNode(node_id=node_id) for node_id in node_ids}
        self.true_layouts = {
            node_id: layout
            for node_id, layout in _layouts_from_system(self.system).items()
            if node_id in self.nodes
        }
        config = app_config_from_system(self.system)
        config.transport = TransportConfig(mode="sim", sim_node_ids=node_ids)
        config.web_host = host
        config.web_port = port
        transport = SimTransport(self.nodes, lock=self._lock)
        self.controller = ControllerApplication(config, transport=transport)
        self.controller.calibrations = {
            node_id: _calibration_from_layout(
                layout,
                bearing_deg=self.system.fleet.mount_by_id(node_id).bearing_deg,
            )
            for node_id, layout in self.true_layouts.items()
        }
        self.sun = SunService(config.site)
        self.absorber_world = self.system.absorber.center_world.copy()

    def startup(self) -> None:
        self.controller.startup()
        for node_id in self.nodes:
            self.nodes[node_id].handle_command(MirrorCommand(node_id=node_id, command=CommandName.HOME))

    def step(self, dt_s: float | None = None) -> dict[str, Any]:
        dt = self.dt_s if dt_s is None else dt_s
        with self._lock:
            for node in self.nodes.values():
                node.step(dt)

            when = datetime.now(timezone.utc)
            sun = self.sun.sun_vector(when)
            statuses = {node_id: node.status() for node_id, node in self.nodes.items()}

            true_geometry = build_true_geometry_from_layouts(
                sun=sun,
                absorber_world=self.absorber_world,
                layouts=self.true_layouts,
                statuses=statuses,
                mirror_offset_d_m=self.system.mirror.mount_offset_d_m,
                system=self.system,
            )
            self.controller.set_true_geometry(true_geometry)

            self.controller.control_tick()
            snapshot = self.controller.current_snapshot()
            self._latest = {
                "mirrors": snapshot["mirrors"],
                "geometry": snapshot["geometry"],
                "true_miss_m": {str(m["node_id"]): m["miss_m"] for m in true_geometry["mirrors"]},
                "estimated_miss_m": {
                    str(m["node_id"]): m["miss_m"] for m in snapshot["geometry"]["estimated"]["mirrors"]
                },
            }
            return self._latest

    def _sim_loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self.step(self.dt_s)
            except Exception as exc:
                print(f"[sitl] simulation step failed: {exc}")
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, self.dt_s - elapsed))

    def run_forever(self) -> None:
        """Run continuous physics + controller web UI until interrupted."""
        self.startup()
        self._stop.clear()
        self._thread = threading.Thread(target=self._sim_loop, name="sitl-sim", daemon=True)
        self._thread.start()
        print("Hot-Box sim-in-the-loop running")
        print(f"Open the UI at http://{self.host}:{self.port}/")
        print(f"Plant constants from config/system.yaml ({self.system.fleet.assembly_count} mirrors)")
        print("Estimated geometry (blue) and true simulator geometry (yellow) are overlaid.")
        print("Use Home / Park / Auto / Jog in the UI to interact with the simulated mirrors.")
        try:
            uvicorn.run(
                self.controller.fastapi,
                host=self.host,
                port=self.port,
                log_level="info",
            )
        finally:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=1.0)

    def run(self, seconds: float = 5.0, dt_s: float = 0.05) -> None:
        """Headless batch run for tests / smoke checks."""
        self.dt_s = dt_s
        self.startup()
        steps = int(seconds / dt_s)
        snapshot: dict[str, Any] = {}
        for _ in range(steps):
            snapshot = self.step(dt_s)
        print("sim_in_the_loop complete")
        print(f"mirrors={snapshot['mirrors']}")
        print(f"true_miss_m={snapshot['true_miss_m']}")
        print(f"estimated_miss_m={snapshot['estimated_miss_m']}")
