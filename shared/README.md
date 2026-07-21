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
```

## Mirror pointing

Both the live controller and the full raytrace simulation use the same bisector-tracking
solver in `hotbox_shared.aiming`. This is the single source of truth for deciding where
each alt-az mount should point.

### Quick start

```python
import numpy as np
from hotbox_shared import MirrorGridSpec, solve_bisector_tracking_for_grid

grid = MirrorGridSpec(
    grid_nx=3,
    grid_ny=5,
    pitch_m=0.26035,
    radius_of_curvature_m=5.5,
)

# Unit vector from the sun toward the plant (ENU world frame).
sun_toward_scene = -sun_vector_from_pvlib

angles = solve_bisector_tracking_for_grid(
    sun_direction_toward_scene=sun_toward_scene,
    mount_world=np.array([0.0, 2.5, 1.0]),
    target_world=np.array([0.0, 0.0, 1.0]),  # absorber center
    grid=grid,
)

print(angles.azimuth_deg, angles.elevation_deg)  # send to firmware
```

### Algorithm

1. **Sun ray** — unit vector from the sun toward the mirrors.
2. **Target ray** — unit vector from the mount pivot toward the absorber center.
3. **Bisector normal** — specular mirror normal that reflects (1) toward (2).
4. **Inverse kinematics** — solve mount `(azimuth, elevation)` so the pivot facet body
   normal aligns with the bisector under `R = R_z(az) @ R_x(el)`.

The approximation is a **flat heliostat at the pivot**: outgoing direction uses
`mount → absorber`, not the facet center offset.

### API reference

| Symbol | Role |
|--------|------|
| `solve_bisector_tracking` | Primary solver — takes explicit pivot facet normal |
| `solve_bisector_tracking_for_grid` | Convenience wrapper using `MirrorGridSpec` |
| `MountAngles` | Result: `azimuth_deg`, `elevation_deg` |
| `MirrorGridSpec` | Facet grid parameters from `system.yaml` mirror section |
| `bisector_normal` | Low-level specular normal from incoming/outgoing rays |
| `bisector_normal_at_mount` | Bisector at a mount position toward a target point |
| `mount_rotation_matrix` | Body → world rotation `R_z(az) @ R_x(el)` |
| `mount_az_el_align_body_normal_to_world` | Inverse kinematics for a body normal |
| `pivot_facet_normal_body` | Center facet normal at identity mount |

### Frames

Right-handed ENU world frame: **+x east**, **+y north**, **+z up**. Mount body frame
matches world at `(azimuth, elevation) = (0, 0)`.

## Firmware header

Generate a C header from the same YAML:

```bash
cd shared
uv run hotbox-gen-firmware-geometry
```

This writes `firmware/include/hotbox_geometry.h`.
