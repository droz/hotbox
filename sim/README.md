# hotbox sim

Simple, vectorized optical raytrace for a cylindrical-mirror solar oven concept.

## Run with uv

```bash
uv run hotbox-sim
```

The script prints power statistics and opens Plotly figures:
- 3D scene (ground, absorber, mirror, incoming and reflected rays)
- Spot pattern on absorber plane
- Delivered power vs local time (sunrise–sunset on the configured day, every 10 minutes)

Edit parameters in `src/main.py` to try different geometry and conditions.
