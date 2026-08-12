# hotbox sim in the loop

Controller + **firmware C-in-the-loop** (real `axis`/`protocol` C++) + Python plant,
talking the same JSON protocol as USB, with a live web UI.

## Prerequisites

Native CIL libraries (autobuild on first run, or explicitly):

```bash
cd firmware/native && make
```

This produces `libfirmware_cil_node{0,1,2}.dylib` (macOS) or `.so` (Linux) — one
library per mirror so multi-node sims do not share HAL state.

## Live interactive mode (default)

```bash
uv run hotbox-sim-in-the-loop
```

Then open:

```text
http://127.0.0.1:8000/
```

Or from another device on the same network (server binds to all interfaces by default):

```text
http://<your-lan-ip>:8000/
```

The page shows:

- current reported pose (blue)
- firmware position setpoint / target (yellow)
- GPS / sun / mirror status
- Home, Park, Auto, and jog controls

Optional flags:

```bash
uv run hotbox-sim-in-the-loop --host 127.0.0.1 --port 8000
```

Use `--host 127.0.0.1` if you want localhost-only access.

## Headless batch mode

Useful for smoke tests:

```bash
uv run hotbox-sim-in-the-loop --batch-seconds 3
```
