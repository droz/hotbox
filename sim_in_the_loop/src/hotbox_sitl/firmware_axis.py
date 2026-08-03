"""firmware_axis.py — ctypes wrapper around the native CIL shared library.

The CIL (C-in-the-loop) library is the real firmware control code compiled for
the host OS.  This module provides two classes that mirror the SITL-internal
interfaces:

* ``FirmwareAxis``    — single-axis view (matches ``ActuatorModel`` interface).
* ``FirmwareMirrorNode`` — full two-axis node (matches ``SimulatedMirrorNode``).

Building the library
--------------------
    cd firmware/native && make

The output lands in ``firmware/.pio/build/native_cil/``.  This module searches
for it automatically; override with the ``HOTBOX_CIL_LIB`` environment variable.

If the library is not found, ``load_cil_library()`` raises ``FileNotFoundError``
so callers can gracefully fall back to the pure-Python SITL.
"""

from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path
import subprocess

from hotbox_controller.protocol import CommandName, MirrorCommand, MirrorStatus


# ── Library discovery ──────────────────────────────────────────────────────────

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

    # ── Declare return/arg types ──────────────────────────────────────────────
    lib.hotbox_cil_init.restype = None
    lib.hotbox_cil_reset.restype = None

    lib.hotbox_cil_set_encoder.argtypes = [ctypes.c_int, ctypes.c_long]
    lib.hotbox_cil_set_encoder.restype = None

    lib.hotbox_cil_set_hall.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.hotbox_cil_set_hall.restype = None

    lib.hotbox_cil_home.restype = None
    lib.hotbox_cil_stop.restype = None
    lib.hotbox_cil_clear_error.restype = None

    lib.hotbox_cil_set_target.argtypes = [ctypes.c_float, ctypes.c_float]
    lib.hotbox_cil_set_target.restype = None

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

    lib.hotbox_cil_azimuth_deg.restype = ctypes.c_float
    lib.hotbox_cil_elevation_deg.restype = ctypes.c_float
    lib.hotbox_cil_is_homed.restype = ctypes.c_int
    lib.hotbox_cil_mode.restype = ctypes.c_char_p
    lib.hotbox_cil_fault.restype = ctypes.c_char_p

    lib.hotbox_cil_init()
    load_cil_library._lib = lib  # type: ignore[attr-defined]
    return lib

load_cil_library._lib = None  # type: ignore[attr-defined]


# ── FirmwareMirrorNode ────────────────────────────────────────────────────────

class FirmwareMirrorNode:
    """Drop-in replacement for ``SimulatedMirrorNode`` that runs the real
    firmware C++ control code via the native CIL shared library.

    The library contains a single global ``MirrorMount`` instance, so only one
    ``FirmwareMirrorNode`` can exist at a time.  Multi-node SITL runs still use
    ``SimulatedMirrorNode`` for all but the node under test (or use separate
    processes — future work).

    Physics (inertia, motor lag) are still modelled in Python via ``ActuatorModel``.
    The firmware only sees encoder ticks and hall states; it outputs PWM [-1, 1]
    which the Python plant converts back to velocity / position.
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

        # Python-side plant state (position & velocity, both axes).
        self._az_angle_deg: float = 0.0
        self._el_angle_deg: float = 0.0
        self._az_vel_deg_s: float = 0.0
        self._el_vel_deg_s: float = 0.0
        # Hall reference (home) angle for each axis.
        self._az_hall_deg: float = 0.0
        self._el_hall_deg: float = 0.0

    @classmethod
    def from_constants(cls, node_id: int, ac: "hotbox_shared.ActuatorConstants", lib_path: Path | str | None = None) -> "FirmwareMirrorNode":  # type: ignore[name-defined]
        return cls(
            node_id=node_id,
            ticks_per_degree=ac.ticks_per_degree,
            max_velocity_deg_s=ac.max_velocity_deg_s,
            velocity_time_constant_s=ac.velocity_time_constant_s,
            control_period_s=ac.control_period_s,
            lib_path=lib_path,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _inject_plant_state(self) -> None:
        """Push current Python plant state into the firmware HAL."""
        az_ticks = int(round(self._az_angle_deg * self.ticks_per_degree))
        el_ticks = int(round(self._el_angle_deg * self.ticks_per_degree))
        self._lib.hotbox_cil_set_encoder(0, az_ticks)
        self._lib.hotbox_cil_set_encoder(1, el_ticks)

        az_hall = abs(self._az_angle_deg - self._az_hall_deg) <= 1.0
        el_hall = abs(self._el_angle_deg - self._el_hall_deg) <= 1.0
        self._lib.hotbox_cil_set_hall(0, int(az_hall))
        self._lib.hotbox_cil_set_hall(1, int(el_hall))

    def _apply_pwm_to_plant(self, pwm_az: float, pwm_el: float, dt_s: float) -> None:
        """First-order lag plant model: PWM → velocity → position."""
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

    # ── SimulatedMirrorNode interface ─────────────────────────────────────────

    def handle_command(self, command: MirrorCommand) -> None:
        if command.node_id != self.node_id:
            return
        cmd = command.command
        if cmd == CommandName.HOME:
            # Position axes just behind hall so homing sweep finds it quickly.
            self._az_angle_deg = self._az_hall_deg - 5.0
            self._el_angle_deg = self._el_hall_deg - 5.0
            self._az_vel_deg_s = 0.0
            self._el_vel_deg_s = 0.0
            self._inject_plant_state()
            self._lib.hotbox_cil_home()
        elif cmd == CommandName.STOP:
            self._lib.hotbox_cil_stop()
        elif cmd == CommandName.SET_TARGET:
            az = float(command.payload.get("azimuth_deg", 0.0))
            el = float(command.payload.get("elevation_deg", 0.0))
            self._lib.hotbox_cil_set_target(az, el)
        elif cmd == CommandName.CLEAR_ERROR:
            self._lib.hotbox_cil_clear_error()

    def status(self) -> MirrorStatus:
        mode_bytes = self._lib.hotbox_cil_mode()
        mode = mode_bytes.decode() if mode_bytes else "idle"
        fault_bytes = self._lib.hotbox_cil_fault()
        fault = fault_bytes.decode() if fault_bytes else None
        return MirrorStatus(
            node_id=self.node_id,
            homed=bool(self._lib.hotbox_cil_is_homed()),
            fault=fault,
            azimuth_deg=self._az_angle_deg,
            elevation_deg=self._el_angle_deg,
            mode=mode,
        )

    def step(self, dt_s: float) -> None:
        """Advance the plant while running firmware at its own control period."""
        if dt_s <= 0.0:
            return
        substeps = max(1, int(math.ceil(dt_s / self.control_period_s)))
        sub_dt = dt_s / substeps
        for _ in range(substeps):
            self._substep_once(sub_dt)
