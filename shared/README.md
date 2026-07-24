# hotbox-shared

Shared plant constants and mirror pointing geometry for:

- `controller/`
- `sim_in_the_loop/`
- `sim_full_raytrace/`
- firmware (via generated C header)

## Source of truth

Edit the repo-level file:

```text
config/system.yaml
```

Then load it in Python:

```python
from hotbox_shared import load_system_constants

system = load_system_constants()
print(system.default_site.latitude_deg)
print(system.absorber.center_height_m)
print(system.mirror.grid_nx, system.mirror.grid_ny)
print(system.control.solve_for_mount_offset)
```

## Mirror pointing

Both the live controller and the full raytrace simulation use the same aiming solver in
`hotbox_shared.aiming`. This is the single source of truth for deciding where each alt-az
mount should point.

### Quick start

```python
import numpy as np
from hotbox_shared import MirrorGridSpec, evaluate_center_ray, solve_tracking_for_grid

grid = MirrorGridSpec(
    grid_nx=3,
    grid_ny=5,
    pitch_m=0.26035,
    radius_of_curvature_m=5.5,
    mount_offset_d_m=0.1,
)

# Unit vector from the sun toward the plant (ENU world frame).
sun_toward_scene = -sun_vector_from_pvlib
mount = np.array([0.0, 2.5, 1.0])
target = np.array([0.0, 0.0, 1.0])  # absorber center

angles = solve_tracking_for_grid(
    sun_direction_toward_scene=sun_toward_scene,
    mount_world=mount,
    target_world=target,
    grid=grid,
    solve_for_mount_offset=True,  # also controlled by control.solve_for_mount_offset in YAML
)

# Forward model: where does the center ray go?
ray = evaluate_center_ray(
    sun_direction_toward_scene=sun_toward_scene,
    mount_world=mount,
    azimuth_deg=angles.azimuth_deg,
    elevation_deg=angles.elevation_deg,
    mount_offset_d_m=grid.mount_offset_d_m,
    pivot_facet_normal_body=grid.pivot_normal_body(),
)
print(angles.azimuth_deg, angles.elevation_deg)
print("miss_m", ray.miss_m(target))
print("impact on absorber plane", ray.impact_on_plane(target, np.array([0.0, 1.0, 0.0])))
```

### Algorithm

1. **Bisector seed** — flat heliostat at the mount pivot (ignores `mount_offset_d_m`).
2. **Offset refine** (optional) — `scipy.optimize.least_squares` nudges `(az, el)` so the
   **center facet** reflected ray aims at the absorber. Forward model:
   `evaluate_center_ray`.

Toggle refinement with `control.solve_for_mount_offset` in `config/system.yaml`
(set `false` to keep the bisector seed only).

### API reference

| Symbol | Role |
|--------|------|
| `solve_tracking` / `solve_tracking_for_grid` | **Primary** aiming API (seed + optional refine) |
| `evaluate_center_ray` / `CenterRay` | Forward geometry: facet pose, reflected ray, miss / plane impact |
| `solve_bisector_tracking` | Closed-form pivot bisector seed only |
| `MountAngles` | Result: `azimuth_deg`, `elevation_deg` |
| `MirrorGridSpec` | Facet grid + `mount_offset_d_m` |
| `pivot_facet_center_world` | `mount + R @ (0,0,d)` |
| `mount_rotation_matrix` | Body → world rotation `R_z(az) @ R_x(el)` |

### Frames

Right-handed ENU world frame: **+x east**, **+y north**, **+z up**. Mount body frame
matches world at `(azimuth, elevation) = (0, 0)`. Center facet at `(0, 0, mount_offset_d_m)`
in body coordinates.

## Firmware header

Generate a C header from the same YAML:

```bash
cd shared
uv run hotbox-gen-firmware-geometry
```

This writes `firmware/include/hotbox_geometry.h`.
