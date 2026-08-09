// hal.cpp — HAL global state and function definitions for the native CIL build.
// All other TUs (axis.cpp, firmware_cil.cpp) link here for HAL calls so they
// all read/write the same arrays.

#include "include/Arduino.h"
#include "include/ESP32Encoder.h"

volatile int  g_hal_digital_out[32] = {};
volatile int  g_hal_digital_in[32]  = {};
volatile int  g_hal_analog_out[32]  = {};
volatile long g_encoder_counts[2]   = {};
_SerialStub Serial;

void pinMode(int /*pin*/, int /*mode*/) {}

void analogWrite(int pin, int value) {
    if (pin >= 0 && pin < 32) g_hal_analog_out[pin] = value;
}

void analogWriteFrequency(uint32_t /*frequency*/) {}

int digitalRead(int pin) {
    return (pin >= 0 && pin < 32) ? g_hal_digital_in[pin] : 0;
}

void digitalWrite(int pin, int value) {
    if (pin >= 0 && pin < 32) g_hal_digital_out[pin] = value;
}

void delay(unsigned long /*ms*/) {}
unsigned long millis() { return 0; }
