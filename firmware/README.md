# hotbox firmware

Arduino-first mirror controller for the Arduino Nano ESP32.

Goals:

- same high-level command protocol over CAN and USB serial
- brushed DC motor control through DRV8871
- quadrature encoder + hall homing
- host-shim friendly structure for simulation

Build / flash with `uv` (installs PlatformIO into this package env).
Each mirror has its own build env, and upload always builds that node first:

```bash
cd firmware
uv run hotbox-firmware-build --node-id 0
uv run hotbox-firmware-upload --node-id 1
# optional: uv run hotbox-firmware-upload --node-id 1 --port /dev/cu.usbmodem1101
```

`--node-id` must be `0`, `1`, or `2` (matches `config/system.yaml` mounts).

Or call PlatformIO directly:

```bash
pio run -e nano_esp32_node0
pio run -t upload -e nano_esp32_node1
```

The firmware implements:

- brushed-axis control with `ESP32Encoder` + `PID_v1`
- hall-sensor homing
- position servo (`set_target`); host integrates jog rates into targets
- status modes: `idle` | `homing` | `position` | `fault`
- USB serial JSON protocol and ESP32-S3 TWAI (CAN) binary protocol
- TWAI soft-fails without a transceiver so USB bench bring-up still works
