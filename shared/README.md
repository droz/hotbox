# hotbox-shared

Shared plant / geometry constants for:

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

## Firmware header

Generate a C header from the same YAML:

```bash
cd shared
uv run hotbox-gen-firmware-geometry
```

This writes `firmware/include/hotbox_geometry.h`.
