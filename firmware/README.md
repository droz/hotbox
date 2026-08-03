# hotbox firmware

Arduino-first mirror controller for the Arduino Nano ESP32.

Goals:

- same high-level command protocol over CAN and USB serial
- brushed DC motor control through DRV8871
- quadrature encoder + hall homing
- host-shim friendly structure for simulation

Build with PlatformIO:

```bash
pio run
```

The firmware implements:

- brushed-axis control with `ESP32Encoder` + `PID_v1`
- hall-sensor homing
- position servo (`set_target`); host integrates jog rates into targets
- status modes: `idle` | `homing` | `position` | `fault`
- the same JSON command protocol used over USB serial
