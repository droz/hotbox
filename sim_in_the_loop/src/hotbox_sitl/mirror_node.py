"""Mirror node for SITL: real firmware via native C-in-the-loop (CIL).

Python owns only the plant (encoders/halls → ticks; PWM → motion). Firmware
(``axis`` + ``protocol``) runs in a per-node shared library so multi-mirror
sims do not share HAL state.

Build (autobuild also runs from here when sources are newer)::

    cd firmware/native && make

Optional: ``HOTBOX_CIL_LIB_DIR`` overrides the directory that contains
``libfirmware_cil_node{N}.*``.
"""

from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path
import subprocess
import sys

from hotbox_controller.protocol import MirrorCommand, MirrorStatus


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[3]  # …/hotbox/sim_in_the_loop/src/hotbox_sitl/ → repo root


def _lib_name(node_id: int) -> str:
    if sys.platform == "darwin":
        return f"libfirmware_cil_node{node_id}.dylib"
    if sys.platform == "win32":
        return f"firmware_cil_node{node_id}.dll"
    return f"libfirmware_cil_node{node_id}.so"


def _cil_dependencies() -> list[Path]:
    repo_root = _repo_root()
    return [
        repo_root / "firmware" / "src" / "axis.cpp",
        repo_root / "firmware" / "src" / "axis.h",
        repo_root / "firmware" / "src" / "protocol.cpp",
        repo_root / "firmware" / "src" / "protocol.h",
        repo_root / "firmware" / "src" / "config.h",
        repo_root / "firmware" / "include" / "hotbox_geometry.h",
        repo_root / "firmware" / "native" / "firmware_cil.cpp",
        repo_root / "firmware" / "native" / "hal.cpp",
        repo_root / "firmware" / "native" / "include" / "Arduino.h",
        repo_root / "firmware" / "native" / "include" / "ESP32Encoder.h",
        repo_root / "firmware" / "native" / "include" / "PID_v1.h",
        repo_root / "firmware" / "native" / "Makefile",
    ]


def _latest_mtime(paths: list[Path]) -> float:
    times = [path.stat().st_mtime for path in paths if path.exists()]
    return max(times) if times else 0.0


def _ensure_cil_library(node_id: int) -> Path:
    """Build/refresh the per-node CIL library when sources are newer."""
    make_dir = _repo_root() / "firmware" / "native"
    env_dir = os.environ.get("HOTBOX_CIL_LIB_DIR")
    if env_dir:
        lib_path = Path(env_dir) / _lib_name(node_id)
        if not lib_path.exists():
            raise FileNotFoundError(
                f"Native CIL library not found: {lib_path}\n"
                f"Build with: cd {make_dir} && make NODE_IDS={node_id}"
            )
        return lib_path

    lib_path = make_dir / _lib_name(node_id)
    deps = _cil_dependencies()
    needs_build = (not lib_path.exists()) or (_latest_mtime(deps) > lib_path.stat().st_mtime)
    if needs_build:
        subprocess.run(["make", f"NODE_IDS={node_id}"], cwd=make_dir, check=True)
    if not lib_path.exists():
        raise FileNotFoundError(
            f"Native CIL library not found: {lib_path}\n"
            f"Build with: cd {make_dir} && make"
        )
    return lib_path


def _bind_cil_api(lib: ctypes.CDLL) -> None:
    lib.hotbox_cil_init.restype = None
    lib.hotbox_cil_reset.restype = None

    lib.hotbox_cil_set_encoder.argtypes = [ctypes.c_int, ctypes.c_long]
    lib.hotbox_cil_set_encoder.restype = None

    lib.hotbox_cil_get_encoder.argtypes = [ctypes.c_int]
    lib.hotbox_cil_get_encoder.restype = ctypes.c_long

    lib.hotbox_cil_set_hall.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.hotbox_cil_set_hall.restype = None

    lib.hotbox_cil_handle_line.argtypes = [ctypes.c_char_p]
    lib.hotbox_cil_handle_line.restype = None

    lib.hotbox_cil_status_json.restype = ctypes.c_char_p

    lib.hotbox_cil_update.argtypes = [ctypes.c_float]
    lib.hotbox_cil_update.restype = None

    lib.hotbox_cil_step.argtypes = [
        ctypes.c_long,
        ctypes.c_long,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.hotbox_cil_step.restype = None

    lib.hotbox_cil_pwm_az.restype = ctypes.c_float
    lib.hotbox_cil_pwm_el.restype = ctypes.c_float


_cil_libs: dict[int, ctypes.CDLL] = {}


def load_cil_library(node_id: int, *, lib_path: Path | str | None = None) -> ctypes.CDLL:
    """Load the CIL shared library for ``node_id`` (cached per node)."""
    if node_id in _cil_libs:
        return _cil_libs[node_id]

    resolved = Path(lib_path) if lib_path else _ensure_cil_library(node_id)
    lib = ctypes.CDLL(str(resolved))
    _bind_cil_api(lib)
    lib.hotbox_cil_init()
    _cil_libs[node_id] = lib
    return lib


class MirrorNode:
    """SITL mirror: firmware CIL + Python plant (encoders, halls, PWM→motion)."""

    def __init__(
        self,
        node_id: int,
        ticks_per_degree: float = 35.56,
        max_velocity_deg_s: float = 30.0,
        velocity_time_constant_s: float = 0.2,
        control_period_s: float = 0.02,
        *,
        home_azimuth_deg: float = 0.0,
        home_elevation_deg: float = 90.0,
        hall_window_width_deg: float = 8.0,
        lib_path: Path | str | None = None,
    ) -> None:
        self.node_id = node_id
        self.ticks_per_degree = ticks_per_degree
        self.max_velocity_deg_s = max_velocity_deg_s
        self.velocity_time_constant_s = velocity_time_constant_s
        self.control_period_s = control_period_s
        self.hall_half_width_deg = max(0.05, float(hall_window_width_deg) * 0.5)

        self._lib = load_cil_library(node_id, lib_path=lib_path)
        self._lib.hotbox_cil_reset()

        # Plant + firmware share joint frame: az relative to oven-facing, el absolute.
        self._az_hall_deg: float = float(home_azimuth_deg)
        self._el_hall_deg: float = float(home_elevation_deg)
        self._az_angle_deg: float = self._az_hall_deg
        self._el_angle_deg: float = self._el_hall_deg
        self._az_vel_deg_s: float = 0.0
        self._el_vel_deg_s: float = 0.0
        self._inject_plant_state()

    @classmethod
    def from_constants(
        cls,
        node_id: int,
        ac: "hotbox_shared.ActuatorConstants",  # type: ignore[name-defined]
        *,
        lib_path: Path | str | None = None,
    ) -> "MirrorNode":
        return cls(
            node_id=node_id,
            ticks_per_degree=ac.ticks_per_degree,
            max_velocity_deg_s=ac.max_velocity_deg_s,
            velocity_time_constant_s=ac.velocity_time_constant_s,
            control_period_s=ac.control_period_s,
            home_azimuth_deg=ac.home_azimuth_deg,
            home_elevation_deg=ac.home_elevation_deg,
            hall_window_width_deg=ac.hall_window_width_deg,
            lib_path=lib_path,
        )

    def _inject_plant_state(self) -> None:
        az_ticks = int(round(self._az_angle_deg * self.ticks_per_degree))
        el_ticks = int(round(self._el_angle_deg * self.ticks_per_degree))
        self._lib.hotbox_cil_set_encoder(0, az_ticks)
        self._lib.hotbox_cil_set_encoder(1, el_ticks)
        az_hall = abs(self._az_angle_deg - self._az_hall_deg) <= self.hall_half_width_deg
        el_hall = abs(self._el_angle_deg - self._el_hall_deg) <= self.hall_half_width_deg
        self._lib.hotbox_cil_set_hall(0, int(az_hall))
        self._lib.hotbox_cil_set_hall(1, int(el_hall))

    def _apply_pwm_to_plant(self, pwm_az: float, pwm_el: float, dt_s: float) -> None:
        alpha = min(1.0, dt_s / self.velocity_time_constant_s)
        target_az = pwm_az * self.max_velocity_deg_s
        target_el = pwm_el * self.max_velocity_deg_s
        self._az_vel_deg_s += (target_az - self._az_vel_deg_s) * alpha
        self._el_vel_deg_s += (target_el - self._el_vel_deg_s) * alpha
        self._az_angle_deg += self._az_vel_deg_s * dt_s
        self._el_angle_deg += self._el_vel_deg_s * dt_s

    def _substep_once(self, dt_s: float) -> None:
        az_ticks = int(round(self._az_angle_deg * self.ticks_per_degree))
        el_ticks = int(round(self._el_angle_deg * self.ticks_per_degree))
        az_hall = int(abs(self._az_angle_deg - self._az_hall_deg) <= self.hall_half_width_deg)
        el_hall = int(abs(self._el_angle_deg - self._el_hall_deg) <= self.hall_half_width_deg)
        pwm_az = ctypes.c_float()
        pwm_el = ctypes.c_float()
        self._lib.hotbox_cil_step(
            az_ticks,
            el_ticks,
            az_hall,
            el_hall,
            dt_s,
            ctypes.byref(pwm_az),
            ctypes.byref(pwm_el),
        )
        # finishHoming remaps encoder counts; keep the plant in that frame.
        az_after = int(self._lib.hotbox_cil_get_encoder(0))
        el_after = int(self._lib.hotbox_cil_get_encoder(1))
        if az_after != az_ticks:
            self._az_angle_deg = az_after / self.ticks_per_degree
            self._az_vel_deg_s = 0.0
        if el_after != el_ticks:
            self._el_angle_deg = el_after / self.ticks_per_degree
            self._el_vel_deg_s = 0.0
        self._apply_pwm_to_plant(pwm_az.value, pwm_el.value, dt_s)

    def handle_command(self, command: MirrorCommand) -> None:
        if command.node_id != self.node_id:
            return
        line = command.to_wire().decode("utf-8").strip()
        self._lib.hotbox_cil_handle_line(line.encode("utf-8"))

    def status(self) -> MirrorStatus:
        raw = self._lib.hotbox_cil_status_json()
        text = (raw.decode("utf-8") if raw else "") + "\n"
        fw = MirrorStatus.from_wire(text.encode("utf-8"))
        return MirrorStatus(
            node_id=self.node_id,
            azimuth_home=fw.azimuth_home,
            elevation_home=fw.elevation_home,
            fault=fw.fault,
            azimuth_deg=fw.azimuth_deg,
            elevation_deg=fw.elevation_deg,
            target_azimuth_deg=fw.target_azimuth_deg,
            target_elevation_deg=fw.target_elevation_deg,
            azimuth_integral=fw.azimuth_integral,
            elevation_integral=fw.elevation_integral,
            pid_kp=fw.pid_kp,
            pid_ki=fw.pid_ki,
            pid_kd=fw.pid_kd,
            pid_velocity_kp=fw.pid_velocity_kp,
            pid_velocity_ki=fw.pid_velocity_ki,
            pid_velocity_kd=fw.pid_velocity_kd,
            az_hall_width_deg=fw.az_hall_width_deg,
            el_hall_width_deg=fw.el_hall_width_deg,
            mode=fw.mode,
        )

    def step(self, dt_s: float) -> None:
        if dt_s <= 0.0:
            return
        substeps = max(1, int(math.ceil(dt_s / self.control_period_s)))
        sub_dt = dt_s / substeps
        for _ in range(substeps):
            self._substep_once(sub_dt)
