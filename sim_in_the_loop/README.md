# hotbox sim in the loop

Fast development simulator for:

- controller + simulated mirrors talking over the same protocol
- continuous physics with a live web UI
- simplified single-ray-per-facet geometry checks
- estimated vs true geometry overlay

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

- estimated controller geometry (blue)
- true simulator geometry (yellow)
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
