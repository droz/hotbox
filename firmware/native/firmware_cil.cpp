// firmware_cil.cpp — C-in-the-loop (CIL) shared library entry point.
//
// Compiles real firmware axis.cpp + protocol.cpp against stub Arduino HAL and
// exposes a thin C API so Python can:
//   * inject plant I/O (encoders, halls)
//   * advance the control loop and read PWM
//   * send the same JSON command lines / read the same status JSON as USB
//
// Homing and other command semantics live entirely in firmware (ProtocolHandler
// + MirrorMount). This file is only transport + plant plumbing.
//
// Build:  see platformio.ini [env:native_cil]  or the Makefile in this folder.
// Python: see sim_in_the_loop/src/hotbox_sitl/firmware_axis.py

#include <cstring>

#include "Arduino.h"
#include "ESP32Encoder.h"
#include "PID_v1.h"

#include "../src/config.h"
#include "../src/axis.h"
#include "../src/protocol.h"

static hotbox::MirrorMount g_mount;
static hotbox::ProtocolHandler g_protocol(&g_mount);
static bool g_initialised = false;
static char g_status_buf[768];

static float _read_signed_pwm(int motor_p, int motor_m) {
    int p = g_hal_analog_out[motor_p];
    int m = g_hal_analog_out[motor_m];
    return static_cast<float>(p - m) / 255.0f;
}

extern "C" {

void hotbox_cil_init(void) {
    if (g_initialised) {
        return;
    }
    g_mount.begin();
    g_initialised = true;
}

void hotbox_cil_reset(void) {
    g_initialised = false;
    for (int i = 0; i < 32; ++i) {
        g_hal_analog_out[i] = 0;
        g_hal_digital_in[i] = 0;
        g_hal_digital_out[i] = 0;
    }
    for (int i = 0; i < 2; ++i) {
        g_encoder_counts[i] = 0;
    }
    g_mount.~MirrorMount();
    new (&g_mount) hotbox::MirrorMount();
    // ProtocolHandler holds a mount pointer; remount in place keeps the address.
    hotbox_cil_init();
}

void hotbox_cil_set_encoder(int axis, long ticks) {
    if (axis >= 0 && axis < 2) {
        g_encoder_counts[axis] = ticks;
    }
}

void hotbox_cil_set_hall(int axis, int triggered) {
    const int pin = (axis == 0) ? hotbox::kHorizHall : hotbox::kVertHall;
    // Active-low halls: triggered → pin LOW. Fires native change ISR on edges.
    hotbox_hal_set_digital_in(pin, triggered ? 0 : 1);
}

/** USB-wire JSON command line (with or without trailing newline). */
void hotbox_cil_handle_line(const char* line) {
    if (line == nullptr) {
        return;
    }
    g_protocol.handleLine(String(line));
}

/** Same status object the USB ``get_status`` / emitStatus path produces. */
const char* hotbox_cil_status_json(void) {
    if (g_protocol.formatStatus(g_status_buf, sizeof(g_status_buf)) < 0) {
        g_status_buf[0] = '\0';
    }
    return g_status_buf;
}

void hotbox_cil_update(float dt_s) { g_mount.update(dt_s); }

void hotbox_cil_step(
    long az_ticks,
    long el_ticks,
    int az_hall,
    int el_hall,
    float dt_s,
    float* pwm_az,
    float* pwm_el
) {
    g_encoder_counts[0] = az_ticks;
    g_encoder_counts[1] = el_ticks;
    hotbox_hal_set_digital_in(hotbox::kHorizHall, az_hall ? 0 : 1);
    hotbox_hal_set_digital_in(hotbox::kVertHall, el_hall ? 0 : 1);
    g_mount.update(dt_s);
    if (pwm_az != nullptr) {
        *pwm_az = _read_signed_pwm(hotbox::kHorizMotorP, hotbox::kHorizMotorM);
    }
    if (pwm_el != nullptr) {
        *pwm_el = _read_signed_pwm(hotbox::kVertMotorP, hotbox::kVertMotorM);
    }
}

float hotbox_cil_pwm_az(void) {
    return _read_signed_pwm(hotbox::kHorizMotorP, hotbox::kHorizMotorM);
}

float hotbox_cil_pwm_el(void) {
    return _read_signed_pwm(hotbox::kVertMotorP, hotbox::kVertMotorM);
}

}  // extern "C"
