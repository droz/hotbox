# Baked and Happy Hot-Box

Solar oven project with motorized mirrors, a Raspberry Pi supervisor, and two separate simulation stacks.

## System Overview

The oven body keeps its original controller panel, but the heating element relay is repurposed as a demand signal for the solar concentrator. When the oven wants power, a set of motorized mirrors aims reflected sunlight at the absorber on the back of the oven. When the oven does not want power, the mirrors park toward a safe sky target.

Each mirror is a faceted spherical mirror mounted on an alt-azimuth mount:

- azimuth rotates first around the world `Z` axis
- elevation rotates second around the mirror `X` axis
- each mirror has a local mount frame `Fm`
- the oven absorber center `O` is the origin of the world frame `Fw`

The world frame used by the software is:

- `X`: east
- `Y`: north
- `Z`: up

## Hardware Architecture

### Mirror Controller

Each mirror has its own Arduino Nano ESP32 board. The board controls:

- two brushed DC motors through DRV8871 drivers
- one quadrature encoder per motor
- one hall sensor per axis for homing/reference

The mirror controller communicates with the main controller in two ways:

- **CAN bus** for normal operation in the final system
- **USB serial** for debugging, bench bring-up, and testing without the CAN bus

Pin mappings are documented in `electrical/pinouts.txt`.

### Main Controller

The main controller is a Raspberry Pi with:

- an MCP2515 CAN hat on SPI
- a GPIO input that reads the oven's original heat-demand relay state
- a GY-NEO6MV2 NEO-6M GPS module for time and location

The Pi:

- computes the sun location from GPS data using `pvlib`
- converts the sun position into a world-frame vector
- computes where each mirror should point
- sends commands to each mirror node
- serves a small web UI for monitoring, manual control, and calibration

## Software Architecture

## `firmware/`

Arduino-first mirror-node firmware with the same high-level command set over CAN and USB serial.

Responsibilities:

- homing from hall sensors
- encoder counting
- motor PWM and direction control
- closed-loop axis tracking with acceleration limiting
- stall/fault reporting
- mirror status reporting

The current scaffold is intentionally thin and is meant to grow around proven libraries rather than custom low-level drivers.

## `controller/`

Python package for the Raspberry Pi runtime.

Responsibilities:

- GPS NMEA parsing for the NEO-6M (with site fallback when no fix)
- sun vector calculation with `pvlib`
- mirror discovery and supervision
- transport abstraction for `CAN` and `USB serial`
- target generation for active tracking or safe parking
- calibration file loading/saving
- FastAPI server plus a lightweight Three.js web page
- estimated vs true geometry overlay for simulation debugging

Current module split:

- `hotbox_controller/gps.py`
- `hotbox_controller/sun.py`
- `hotbox_controller/scene.py`
- `hotbox_controller/transport.py`
- `hotbox_controller/mirror_fleet.py`
- `hotbox_controller/tracking.py`
- `hotbox_controller/calibration.py`
- `hotbox_controller/app.py`

## `sim_in_the_loop/`

Fast simulation for controller and firmware development.

Purpose:

- compile or shim the mirror firmware logic on a laptop
- simulate motors, encoders, and hall sensors
- run the same command protocol used by the real system
- visualize simplified aiming geometry
- debug calibration and mirror pointing

This simulation intentionally uses:

- one representative ray per facet center
- no occlusion model
- simulator-owned "true" calibration values

Its goal is software integration and geometry debugging, not high-fidelity thermal/power prediction.

## `sim_full_raytrace/`

Renamed version of the original exploratory optical simulator. This remains the place for:

- many-ray optical studies
- occlusion effects
- delivered power estimation
- geometry experiments that need more fidelity than the in-the-loop simulator

## Tracking And Control Flow

1. The Pi boots and starts the controller service.
2. GPS provides current UTC time and site coordinates.
3. The controller loads saved calibration data.
4. Mirror nodes are discovered over the configured transport (`CAN` or `USB serial`).
5. Mirrors are homed if needed.
6. The controller computes the current sun vector from `pvlib`.
7. If the oven requests heat, mirrors track the absorber.
8. If the oven does not request heat, mirrors park to a safe sky direction.

## Calibration

Per mirror, the known geometric values are:

- distance `|OA_n|`
- mirror offset `d = |A_nM_n|`
- mirror focal length

The unknowns to calibrate are:

- bearing of `OA_n` relative to north
- height difference between `O` and `A_n`
- homed azimuth offset
- homed elevation offset

Calibration flow:

1. Home the mirror.
2. Jog until the center facet points directly at `O`.
3. Jog until the sun is focused on `O`.
4. Save the solved calibration values to disk.

The controller package already includes calibration file load/save support. The numerical solve and guided routine still need to be filled out in more detail as hardware data becomes available.

## Library Strategy

Prefer existing libraries where they fit:

- firmware: `ESP32Encoder`, `PID`, `CAN`, PlatformIO-managed dependencies
- controller: `pvlib`, `python-can`, `pyserial`, `FastAPI`, `numpy`, `scipy`, `PyYAML`
- UI: `Three.js`

Custom code should stay focused on:

- project-specific geometry
- transport-neutral mirror protocol
- calibration math
- virtual actuator and sensor models

## Repository Layout

- `config/` — shared plant constants (`system.yaml`) and calibration files
- `shared/` — Python package that loads those shared constants
- `controller/` — Raspberry Pi controller package
- `electrical/` — pinouts and electronics notes
- `firmware/` — mirror-node firmware
- `mechanical/` — CAD and printable parts
- `sim_full_raytrace/` — high-fidelity optical simulator
- `sim_in_the_loop/` — fast integration simulator

## Shared plant constants

Geometry and plant constants live in one place:

```text
config/system.yaml
```

Loaded by:

```python
from hotbox_shared import load_system_constants
system = load_system_constants()
```

This includes:

- site lat/lon/altitude
- absorber size and height
- mirror facet grid (`nx`/`ny`), tile size, pitch
- mount offset `d`, sphere radius / focal length
- fleet mount bearings and `|OA|` design values

Generate the firmware C header from the same file:

```bash
cd shared && uv run hotbox-gen-firmware-geometry
```

That writes `firmware/include/hotbox_geometry.h`.

## Development

Python packages use `uv`.

Examples:

```bash
cd controller && uv run hotbox-controller
cd sim_in_the_loop && uv run hotbox-sim-in-the-loop
# then open http://127.0.0.1:8000/
cd sim_full_raytrace && uv run hotbox-sim-full-raytrace
```

`hotbox-sim-in-the-loop` starts a continuous simulation and the controller web UI.
Use `--batch-seconds N` for a short headless run.

### Transports

The controller supports three mirror transports:

- `usb` — JSON lines over USB serial (`pyserial`)
- `can` — compact binary frames over SocketCAN (`python-can`)
- `sim` — in-process simulated mirror nodes (used by `sim_in_the_loop/`)

Configure transport mode in `controller/src/hotbox_controller/config.py`.

Firmware uses PlatformIO:

```bash
cd firmware && pio run
```
