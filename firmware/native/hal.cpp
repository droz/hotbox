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

puType ESP32Encoder::useInternalWeakPullResistors = puType::up;

struct HalPinIsr {
  HalIsrCb cb = nullptr;
  void* arg = nullptr;
  int mode = CHANGE;
  bool active = false;
};

static HalPinIsr g_hal_isr[32] = {};

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
void noInterrupts() {}
void interrupts() {}

void attachInterruptArg(int pin, HalIsrCb cb, void* arg, int mode) {
    if (pin < 0 || pin >= 32 || cb == nullptr) {
        return;
    }
    g_hal_isr[pin].cb = cb;
    g_hal_isr[pin].arg = arg;
    g_hal_isr[pin].mode = mode;
    g_hal_isr[pin].active = true;
}

void hotbox_hal_set_digital_in(int pin, int level) {
    if (pin < 0 || pin >= 32) {
        return;
    }
    const int prev = g_hal_digital_in[pin];
    const int next = level ? HIGH : LOW;
    g_hal_digital_in[pin] = next;
    if (prev == next || !g_hal_isr[pin].active || g_hal_isr[pin].cb == nullptr) {
        return;
    }
    const bool rising = (prev == LOW && next == HIGH);
    const bool falling = (prev == HIGH && next == LOW);
    const int mode = g_hal_isr[pin].mode;
    const bool fire =
        (mode == CHANGE && (rising || falling)) ||
        (mode == RISING && rising) ||
        (mode == FALLING && falling);
    if (fire) {
        g_hal_isr[pin].cb(g_hal_isr[pin].arg);
    }
}
