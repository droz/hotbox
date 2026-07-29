#pragma once
// Minimal Arduino HAL stub for native (host) compilation.
// GPIO state is injectable by the test harness via the functions below.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

// ── Types ────────────────────────────────────────────────────────────────────
using String = std::string;

// ── Pin mode / value constants ────────────────────────────────────────────────
constexpr int INPUT  = 0;
constexpr int OUTPUT = 1;
constexpr int HIGH   = 1;
constexpr int LOW    = 0;

// Pin aliases used by config.h (map to arbitrary ints for native)
// These must not clash with any real index used in hotbox::g_*_encoder selection.
constexpr int A0 = 14, A1 = 15, A2 = 16, A3 = 17;
constexpr int D2 = 2,  D3 = 3,  D4 = 4;
constexpr int D5 = 5,  D6 = 6,  D7 = 7;
constexpr int D9 = 9,  D10 = 10;

// ── HAL state — defined in hal.cpp, declared extern here ─────────────────────
// volatile: prevents the optimizer from caching reads across the hal function
// boundary when firmware_cil.cpp and axis.cpp are compiled as separate TUs.
extern volatile int  g_hal_digital_out[32];
extern volatile int  g_hal_digital_in[32];
extern volatile int  g_hal_analog_out[32];  // 0-255 PWM value

// ── HAL functions — defined in hal.cpp (non-inline so all TUs share one copy) ─
void pinMode(int pin, int mode);
void analogWrite(int pin, int value);
int  digitalRead(int pin);
void digitalWrite(int pin, int value);
void delay(unsigned long ms);
unsigned long millis();

// std::fabs / std::abs are already available via <cmath>; no redefinition needed.

// Serial stub — swallowed in native mode.  Defined in hal.cpp.
struct _SerialStub {
    template<typename T> _SerialStub& print(T) { return *this; }
    template<typename T> _SerialStub& println(T) { return *this; }
    template<typename T, typename U> _SerialStub& print(T, U) { return *this; }
    void begin(int) {}
    bool available() { return false; }
    char read() { return 0; }
};
extern _SerialStub Serial;
