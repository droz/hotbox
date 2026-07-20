# hotbox controller

Main Python runtime for the Raspberry Pi:

- reads GPS time/location
- computes the sun vector with `pvlib`
- supervises mirror nodes over CAN or USB serial
- manages calibration files
- serves a small FastAPI + Three.js UI

Run locally with:

```bash
uv run hotbox-controller
```
