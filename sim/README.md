# hotbox sim

Simple, vectorized optical raytrace for a cylindrical-mirror solar oven concept.

## Run with uv

```bash
uv run hotbox-sim
```

The script prints power statistics and opens two Plotly figures:
- 3D scene (ground, absorber, mirror, incoming and reflected rays)
- Spot pattern on absorber plane

Edit parameters in `src/main.py` to try different geometry and conditions.
