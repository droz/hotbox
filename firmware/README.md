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
- acceleration-limited tracking and jog modes
- the same JSON command protocol used over USB serial
