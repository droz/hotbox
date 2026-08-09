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

Then open `http://127.0.0.1:8000/`.

**USB bring-up:** with a single Arduino Nano ESP32 connected, the board is auto-discovered
(VID/PID `2341:0070`) and mapped to node 0. For multiple boards, set
`HOTBOX_USB_PORTS=0:/dev/...,1:/dev/...`. See the root README section
“Flash firmware and test a real board”.

**Simulation:** use `sim_in_the_loop` (`HOTBOX_TRANSPORT=sim`) instead of this package alone.
