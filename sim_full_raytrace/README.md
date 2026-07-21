# hotbox full raytrace sim

Vectorized optical raytrace for the Hot-Box flat-mirror solar oven (rigid facet grid on
alt–az mounts). Plant geometry comes from the shared repo config; sim-only knobs stay in
`src/main.py`.

## Shared plant constants

Edit:

```text
config/system.yaml
```

Loaded via:

```python
from hotbox_shared import load_system_constants
```

This drives site lat/lon/altitude, absorber size, facet grid, sphere radius, mount offset
`d`, and fleet mount bearings relative to the oven/absorber normal (same source as controller
and SITL).

## Run with uv

```bash
uv run hotbox-sim-full-raytrace
```

The script prints power statistics and opens Plotly figures:

- 3D scene (ground, absorber, mirror tiles, incoming and reflected rays)
- Spot pattern on absorber plane
- Delivered power vs local time (sunrise–sunset on the configured day)

Sim-only options (samples, day-curve dates, visualization density) live in `src/main.py`.
Incoming ray power uses **pvlib** clear-sky DNI (Ineichen) at the simulation time for the
configured site.
