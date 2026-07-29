// firmware_cil.cpp — C-in-the-loop (CIL) shared library entry point.
//
// Compiles the real firmware axis.cpp + protocol.cpp against stub Arduino/
// ESP32Encoder/PID_v1 headers and exposes a simple C API so Python (via ctypes)
// can drive the exact same control code that runs on hardware.
//
// Build:  see platformio.ini [env:native_cil]  or the Makefile in this folder.
// Python: see sim_in_the_loop/src/hotbox_sitl/firmware_axis.py

#include <cstring>

// Pull in stub HAL globals first, before any firmware header.
#include "Arduino.h"
#include "ESP32Encoder.h"
#include "PID_v1.h"

// HAL state is defined in hal.cpp; extern declarations come from Arduino.h / ESP32Encoder.h.

// ── Include real firmware sources ─────────────────────────────────────────────
// config.h pulls in hotbox_geometry.h which defines all the tunable constants.
#include "../src/config.h"
#include "../src/axis.h"

// ── Module-level MirrorMount instance ────────────────────────────────────────
static hotbox::MirrorMount g_mount;
static bool g_initialised = false;

// ── Helper: map pin number → analog_out index for PWM reads ──────────────────
// Returns the net signed PWM fraction [-1, 1] that driveMotor() produced.
// driveMotor writes to motor_p_ and motor_m_ pins via analogWrite (0-255).
static float _read_signed_pwm(int motor_p, int motor_m) {
    int p = g_hal_analog_out[motor_p];
    int m = g_hal_analog_out[motor_m];
    // At most one of p/m is non-zero at a time (see driveMotor).
    return static_cast<float>(p - m) / 255.0f;
}

// ── Public C API ──────────────────────────────────────────────────────────────
extern "C" {

// Call once before any other function.
void hotbox_cil_init(void) {
    if (g_initialised) return;
    g_mount.begin();
    g_initialised = true;
}

// Reset to power-on state (useful between test runs).
void hotbox_cil_reset(void) {
    g_initialised = false;
    for (int i = 0; i < 32; ++i) { g_hal_analog_out[i] = 0; g_hal_digital_in[i] = 0; g_hal_digital_out[i] = 0; }
    for (int i = 0; i < 2;  ++i) { g_encoder_counts[i] = 0; }
    // Reconstruct in place — MirrorMount has no dynamic alloc so this is safe.
    g_mount.~MirrorMount();
    new (&g_mount) hotbox::MirrorMount();
    hotbox_cil_init();
}

// ── Inject plant state ────────────────────────────────────────────────────────

// Set the quadrature encoder count for an axis.
// axis: 0 = azimuth (horizontal), 1 = elevation (vertical).
void hotbox_cil_set_encoder(int axis, long ticks) {
    if (axis >= 0 && axis < 2) g_encoder_counts[axis] = ticks;
}

// Set the hall sensor digital input for an axis.
void hotbox_cil_set_hall(int axis, int triggered) {
    // D7 = azimuth hall, D4 = elevation hall (see config.h).
    // axis 0 → pin D7 (7), axis 1 → pin D4 (4).
    int pin = (axis == 0) ? 7 : 4;
    g_hal_digital_in[pin] = triggered ? 1 : 0;
}

// ── Commands (mirror the real firmware protocol) ──────────────────────────────

void hotbox_cil_home(void)  { g_mount.home(); }
void hotbox_cil_stop(void)  { g_mount.stop(); }

void hotbox_cil_set_target(float azimuth_deg, float elevation_deg, int parked) {
    g_mount.setTarget(azimuth_deg, elevation_deg, parked ? "parked" : "tracking");
}

void hotbox_cil_jog(float azimuth_rate_deg_s, float elevation_rate_deg_s) {
    g_mount.jog(azimuth_rate_deg_s, elevation_rate_deg_s);
}

void hotbox_cil_clear_error(void) { g_mount.clearError(); }

// ── Advance the control loop by dt_s seconds ─────────────────────────────────
// Call this every simulation step, *after* injecting encoder counts and hall state.
void hotbox_cil_update(float dt_s) {
    g_mount.update(dt_s);
}

// ── Read outputs ──────────────────────────────────────────────────────────────

// PWM fraction [-1.0, 1.0] last written by driveMotor() for each axis.
float hotbox_cil_pwm_az(void) {
    return _read_signed_pwm(hotbox::kHorizMotorP, hotbox::kHorizMotorM);
}

float hotbox_cil_pwm_el(void) {
    return _read_signed_pwm(hotbox::kVertMotorP, hotbox::kVertMotorM);
}

// Position as computed by the firmware (from encoder ticks).
float hotbox_cil_azimuth_deg(void)   { return g_mount.azimuthDeg(); }
float hotbox_cil_elevation_deg(void) { return g_mount.elevationDeg(); }

int   hotbox_cil_is_homed(void) { return g_mount.isHomed() ? 1 : 0; }

// Mode string: "idle", "homing", "tracking", "parked", "jog", "fault".
const char* hotbox_cil_mode(void) { return g_mount.modeText(); }

// Fault string or NULL.
const char* hotbox_cil_fault(void) { return g_mount.faultText(); }

}  // extern "C"
