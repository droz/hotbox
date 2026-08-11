"""firmware_axis.py — ctypes wrapper around the native CIL shared library.

The CIL library is real firmware (``axis`` + ``protocol``) compiled for the host.
Python owns the plant (encoders/halls → ticks; PWM → motion) and talks to firmware
the same way USB does: JSON command lines in, status JSON out.

Building the library
--------------------
    cd firmware/native && make

Override the library path with ``HOTBOX_CIL_LIB`` if needed.
"""

from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path
import subprocess

from hotbox_controller.protocol import CommandName, MirrorCommand, MirrorStatus


_LIB_NAMES = ("libfirmware_cil.dylib", "libfirmware_cil.so", "firmware_cil.dll")


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[3]  # …/hotbox/sim_in_the_loop/src/hotbox_sitl/ → repo root


def _native_build_dir() -> Path:
    return _repo_root() / "firmware" / "native"


def _candidate_lib_paths() -> list[Path]:
    repo_root = _repo_root()
    search_dirs = [
        repo_root / "firmware" / "native",
        repo_root / "firmware" / ".pio" / "build" / "native_cil",
    ]
    return [search_dir / name for search_dir in search_dirs for name in _LIB_NAMES]


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
    return max(path.stat().st_mtime for path in paths if path.exists())


def _autobuild_native_cil() -> Path:
    """Build or refresh the native CIL library when sources are newer."""
    build_dir = _native_build_dir()
    candidates = [path for path in _candidate_lib_paths() if path.parent == build_dir]
    if not candidates:
        raise FileNotFoundError(f"could not determine native CIL output path in {build_dir}")
    lib_path = candidates[0]
    deps = _cil_dependencies()
    needs_build = (not lib_path.exists()) or (_latest_mtime(deps) > lib_path.stat().st_mtime)
    if needs_build:
        subprocess.run(["make"], cwd=build_dir, check=True)
    return lib_path


def _default_lib_path() -> Path:
    native_lib = _autobuild_native_cil()
    if native_lib.exists():
        return native_lib
    search_dirs = sorted({path.parent for path in _candidate_lib_paths()})
    for candidate in _candidate_lib_paths():
        if candidate.exists():
            return candidate
    dirs_str = "\n  ".join(str(d) for d in search_dirs)
    raise FileNotFoundError(
        f"Native CIL library not found.  Searched:\n  {dirs_str}\n\n"
        "Build with one of:\n"
        "  cd firmware/native && make          (plain GCC, no PlatformIO)\n"
        "  cd firmware && pio run -e native_cil  (PlatformIO)\n"
        "Or set HOTBOX_CIL_LIB=/path/to/libfirmware_cil.so"
    )


def load_cil_library(path: Path | str | None = None) -> ctypes.CDLL:
    """Load and configure the CIL shared library.  Idempotent — caches result."""
    if load_cil_library._lib is not None:  # type: ignore[attr-defined]
        return load_cil_library._lib  # type: ignore[attr-defined]

    env_path = os.environ.get("HOTBOX_CIL_LIB")
    resolved = Path(env_path) if env_path else (Path(path) if path else _default_lib_path())
    lib = ctypes.CDLL(str(resolved))

    lib.hotbox_cil_init.restype = None
    lib.hotbox_cil_reset.restype = None

    lib.hotbox_cil_set_encoder.argtypes = [ctypes.c_int, ctypes.c_long]
    lib.hotbox_cil_set_encoder.restype = None

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

    lib.hotbox_cil_init()
    load_cil_library._lib = lib  # type: ignore[attr-defined]
    return lib


load_cil_library._lib = None  # type: ignore[attr-defined]


class FirmwareMirrorNode:
    """Drop-in replacement for ``SimulatedMirrorNode`` that runs real firmware via CIL.

    Python models plant physics; firmware sees encoder ticks and halls and outputs PWM.
    Commands/status use the USB JSON protocol path (``handle_line`` / ``status_json``).
    """

    def __init__(
        self,
        node_id: int,
        ticks_per_degree: float = 35.56,
        max_velocity_deg_s: float = 30.0,
        velocity_time_constant_s: float = 0.2,
        control_period_s: float = 0.02,
        lib_path: Path | str | None = None,
    ) -> None:
        self.node_id = node_id
        self.ticks_per_degree = ticks_per_degree
        self.max_velocity_deg_s = max_velocity_deg_s
        self.velocity_time_constant_s = velocity_time_constant_s
        self.control_period_s = control_period_s

        self._lib = load_cil_library(lib_path)
        self._lib.hotbox_cil_reset()

        self._az_angle_deg: float = 0.0
        self._el_angle_deg: float = 0.0
        self._az_vel_deg_s: float = 0.0
        self._el_vel_deg_s: float = 0.0
        self._az_hall_deg: float = 0.0
        self._el_hall_deg: float = 0.0

    @classmethod
    def from_constants(
        cls,
        node_id: int,
        ac: "hotbox_shared.ActuatorConstants",  # type: ignore[name-defined]
        lib_path: Path | str | None = None,
    ) -> "FirmwareMirrorNode":
        return cls(
            node_id=node_id,
            ticks_per_degree=ac.ticks_per_degree,
            max_velocity_deg_s=ac.max_velocity_deg_s,
            velocity_time_constant_s=ac.velocity_time_constant_s,
            control_period_s=ac.control_period_s,
            lib_path=lib_path,
        )

    def _inject_plant_state(self) -> None:
        az_ticks = int(round(self._az_angle_deg * self.ticks_per_degree))
        el_ticks = int(round(self._el_angle_deg * self.ticks_per_degree))
        self._lib.hotbox_cil_set_encoder(0, az_ticks)
        self._lib.hotbox_cil_set_encoder(1, el_ticks)
        az_hall = abs(self._az_angle_deg - self._az_hall_deg) <= 1.0
        el_hall = abs(self._el_angle_deg - self._el_hall_deg) <= 1.0
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
        az_hall = int(abs(self._az_angle_deg - self._az_hall_deg) <= 1.0)
        el_hall = int(abs(self._el_angle_deg - self._el_hall_deg) <= 1.0)
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
        self._apply_pwm_to_plant(pwm_az.value, pwm_el.value, dt_s)

    def handle_command(self, command: MirrorCommand) -> None:
        if command.node_id != self.node_id:
            return
        # Harness convenience only: start near the hall so a home finishes in sim time.
        if command.command == CommandName.HOME:
            axis = str(command.payload.get("axis", "both")).strip().lower()
            if axis in {"az", "azimuth", "both"}:
                self._az_angle_deg = self._az_hall_deg + 5.0
                self._az_vel_deg_s = 0.0
            if axis in {"el", "elevation", "both"}:
                self._el_angle_deg = self._el_hall_deg + 5.0
                self._el_vel_deg_s = 0.0
            self._inject_plant_state()
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
            azimuth_deg=self._az_angle_deg,
            elevation_deg=self._el_angle_deg,
            azimuth_integral=fw.azimuth_integral,
            elevation_integral=fw.elevation_integral,
            pid_kp=fw.pid_kp,
            pid_ki=fw.pid_ki,
            pid_kd=fw.pid_kd,
            pid_velocity_kp=fw.pid_velocity_kp,
            pid_velocity_ki=fw.pid_velocity_ki,
            pid_velocity_kd=fw.pid_velocity_kd,
            mode=fw.mode,
        )

    def step(self, dt_s: float) -> None:
        if dt_s <= 0.0:
            return
        substeps = max(1, int(math.ceil(dt_s / self.control_period_s)))
        sub_dt = dt_s / substeps
        for _ in range(substeps):
            self._substep_once(sub_dt)
