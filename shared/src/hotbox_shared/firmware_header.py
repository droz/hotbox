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
        "",
    ]
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
