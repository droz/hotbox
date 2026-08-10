from __future__ import annotations

from pathlib import Path

from .load import default_system_yaml_path, load_system_constants
from .system import SystemConstants


def render_firmware_header(system: SystemConstants) -> str:
    lines = [
        "#pragma once",
        "",
        "// Auto-generated from config/system.yaml — do not edit by hand.",
        "// Regenerate with: uv run hotbox-gen-firmware-geometry",
        "",
        "#include <stdint.h>",
        "",
        f"#define HOTBOX_ABSORBER_CENTER_HEIGHT_M ({system.absorber.center_height_m:.6f}f)",
        f"#define HOTBOX_ABSORBER_WIDTH_M ({system.absorber.width_m:.6f}f)",
        f"#define HOTBOX_ABSORBER_HEIGHT_M ({system.absorber.height_m:.6f}f)",
        f"#define HOTBOX_MIRROR_GRID_NX ({system.mirror.grid_nx})",
        f"#define HOTBOX_MIRROR_GRID_NY ({system.mirror.grid_ny})",
        f"#define HOTBOX_MIRROR_TILE_SIDE_M ({system.mirror.tile_side_m:.6f}f)",
        f"#define HOTBOX_MIRROR_PITCH_M ({system.mirror.pitch_m:.6f}f)",
        f"#define HOTBOX_MIRROR_OFFSET_D_M ({system.mirror.mount_offset_d_m:.6f}f)",
        f"#define HOTBOX_MIRROR_RADIUS_OF_CURVATURE_M ({system.mirror.radius_of_curvature_m:.6f}f)",
        f"#define HOTBOX_DEFAULT_OA_DISTANCE_M ({system.mirror.default_oa_distance_m:.6f}f)",
        f"#define HOTBOX_DEFAULT_MOUNT_HEIGHT_M ({system.mirror.default_mount_height_m:.6f}f)",
        f"#define HOTBOX_FLEET_ASSEMBLY_COUNT ({system.fleet.assembly_count})",
        f"#define HOTBOX_SAFE_PARK_AZIMUTH_DEG ({system.control.safe_park_azimuth_deg:.6f}f)",
        f"#define HOTBOX_SAFE_PARK_ELEVATION_DEG ({system.control.safe_park_elevation_deg:.6f}f)",
        f"#define HOTBOX_IDLE_AIM_HEIGHT_ABOVE_ABSORBER_M ({system.control.idle_aim_height_above_absorber_m:.6f}f)",
        f"#define HOTBOX_ELEVATION_MIN_DEG ({system.control.elevation_min_deg:.6f}f)",
        f"#define HOTBOX_ELEVATION_MAX_DEG ({system.control.elevation_max_deg:.6f}f)",
        f"#define HOTBOX_AZIMUTH_MIN_DEG ({system.control.azimuth_min_deg:.6f}f)",
        f"#define HOTBOX_AZIMUTH_MAX_DEG ({system.control.azimuth_max_deg:.6f}f)",
        "",
        "// Oven-facing absolute azimuth [deg] per node (joint az limits are relative to this).",
    ]
    from .mount import oven_facing_azimuth_deg

    absorber = system.absorber.center_world
    for mount in system.fleet.mounts:
        facing = oven_facing_azimuth_deg(system.mount_world(mount.node_id), absorber)
        lines.append(
            f"#define HOTBOX_OVEN_FACING_AZIMUTH_DEG_NODE_{int(mount.node_id)} ({facing:.6f}f)"
        )
    lines.extend(
        [
        "",
        "// Actuator constants (from config/system.yaml actuator section)",
        f"#define HOTBOX_ENCODER_PPR ({system.actuator.encoder_ppr}u)",
        f"#define HOTBOX_MOTOR_GEAR_RATIO ({system.actuator.motor_gear_ratio:.6f}f)",
        f"#define HOTBOX_WORM_GEAR_RATIO ({system.actuator.worm_gear_ratio:.6f}f)",
        f"#define HOTBOX_GEAR_RATIO ({system.actuator.gear_ratio:.6f}f)",
        f"#define HOTBOX_TICKS_PER_DEGREE ({system.actuator.ticks_per_degree:.6f}f)",
        f"#define HOTBOX_MAX_VELOCITY_DEG_S ({system.actuator.max_velocity_deg_s:.6f}f)",
        f"#define HOTBOX_MAX_ACCEL_DEG_S2 ({system.actuator.max_accel_deg_s2:.6f}f)",
        f"#define HOTBOX_HOMING_SEARCH_VELOCITY_DEG_S ({system.actuator.homing_search_velocity_deg_s:.6f}f)",
        f"#define HOTBOX_HOMING_CREEP_VELOCITY_DEG_S ({system.actuator.homing_creep_velocity_deg_s:.6f}f)",
        f"#define HOTBOX_HOMING_BACKOFF_DEG ({system.actuator.homing_backoff_deg:.6f}f)",
        f"#define HOTBOX_HOMING_SETTLE_TOL_DEG ({system.actuator.homing_settle_tol_deg:.6f}f)",
        f"#define HOTBOX_PID_KP ({system.actuator.pid_kp:.6f}f)",
        f"#define HOTBOX_PID_KI ({system.actuator.pid_ki:.6f}f)",
        f"#define HOTBOX_PID_KD ({system.actuator.pid_kd:.6f}f)",
        f"#define HOTBOX_PID_INTEGRAL_LIMIT ({system.actuator.pid_integral_limit:.6f}f)",
        f"#define HOTBOX_PWM_DEADBAND ({system.actuator.pwm_deadband:.6f}f)",
        f"#define HOTBOX_POSITION_DEADBAND_DEG ({system.actuator.position_deadband_deg:.6f}f)",
        f"#define HOTBOX_STALL_VELOCITY_THRESHOLD_DEG_S ({system.actuator.stall_velocity_threshold_deg_s:.6f}f)",
        f"#define HOTBOX_STALL_TIMEOUT_S ({system.actuator.stall_timeout_s:.6f}f)",
        f"#define HOTBOX_CONTROL_PERIOD_S ({system.actuator.control_period_s:.6f}f)",
        "",
        "// Arduino pin labels from config/system.yaml pins (not raw ESP32 GPIO numbers).",
        f"#define HOTBOX_PIN_CAN_TX {system.pins.can_tx}",
        f"#define HOTBOX_PIN_CAN_RX {system.pins.can_rx}",
        f"#define HOTBOX_PIN_ELEVATION_MOTOR_P {system.pins.elevation_motor_p}",
        f"#define HOTBOX_PIN_ELEVATION_MOTOR_M {system.pins.elevation_motor_m}",
        f"#define HOTBOX_PIN_ELEVATION_ENC_A {system.pins.elevation_enc_a}",
        f"#define HOTBOX_PIN_ELEVATION_ENC_B {system.pins.elevation_enc_b}",
        f"#define HOTBOX_PIN_ELEVATION_HALL {system.pins.elevation_hall}",
        f"#define HOTBOX_PIN_AZIMUTH_MOTOR_P {system.pins.azimuth_motor_p}",
        f"#define HOTBOX_PIN_AZIMUTH_MOTOR_M {system.pins.azimuth_motor_m}",
        f"#define HOTBOX_PIN_AZIMUTH_ENC_A {system.pins.azimuth_enc_a}",
        f"#define HOTBOX_PIN_AZIMUTH_ENC_B {system.pins.azimuth_enc_b}",
        f"#define HOTBOX_PIN_AZIMUTH_HALL {system.pins.azimuth_hall}",
        f"#define HOTBOX_MOTOR_PWM_HZ ({int(system.pins.motor_pwm_hz)})",
        "",
        ]
    )
    return "\n".join(lines)


def default_firmware_header_path() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    return repo_root / "firmware" / "include" / "hotbox_geometry.h"


def write_firmware_header(path: Path | None = None, system_yaml: Path | None = None) -> Path:
    system = load_system_constants(system_yaml)
    out = path or default_firmware_header_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_firmware_header(system))
    return out


def main() -> None:
    out = write_firmware_header()
    print(f"wrote {out}")
    print(f"from {default_system_yaml_path()}")


if __name__ == "__main__":
    main()
