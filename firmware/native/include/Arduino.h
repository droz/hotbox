#pragma once
// Minimal Arduino HAL stub for native (host) compilation.
// GPIO state is injectable by the test harness via the functions below.

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

// ── Arduino-like String (enough for firmware protocol.cpp) ───────────────────
class String {
 public:
  std::string s;

  String() = default;
  String(const char* c) : s(c != nullptr ? c : "") {}
  String(const std::string& o) : s(o) {}
  String(char c) : s(1, c) {}

  int length() const { return static_cast<int>(s.size()); }
  const char* c_str() const { return s.c_str(); }

  int indexOf(const String& needle) const {
    const auto p = s.find(needle.s);
    return p == std::string::npos ? -1 : static_cast<int>(p);
  }
  int indexOf(char ch, unsigned from = 0) const {
    const auto p = s.find(ch, from);
    return p == std::string::npos ? -1 : static_cast<int>(p);
  }

  String substring(unsigned start) const {
    if (start >= s.size()) {
      return String();
    }
    return String(s.substr(start));
  }
  String substring(unsigned start, unsigned end) const {
    if (start >= s.size() || end <= start) {
      return String();
    }
    const unsigned capped = end > s.size() ? static_cast<unsigned>(s.size()) : end;
    return String(s.substr(start, capped - start));
  }

  float toFloat() const {
    char* end = nullptr;
    const float v = std::strtof(s.c_str(), &end);
    return v;
  }

  bool startsWith(const char* prefix) const {
    if (prefix == nullptr) {
      return false;
    }
    return s.rfind(prefix, 0) == 0;
  }
  bool startsWith(const String& prefix) const { return startsWith(prefix.s.c_str()); }

  void toLowerCase() {
    for (char& c : s) {
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
  }

  bool operator==(const char* o) const { return s == (o != nullptr ? o : ""); }
  bool operator==(const String& o) const { return s == o.s; }
  bool operator!=(const char* o) const { return !(*this == o); }
  bool operator!=(const String& o) const { return !(*this == o); }

  String& operator+=(const String& o) {
    s += o.s;
    return *this;
  }
  String& operator+=(const char* o) {
    if (o != nullptr) {
      s += o;
    }
    return *this;
  }
};

inline String operator+(const String& a, const String& b) { return String(a.s + b.s); }
inline String operator+(const char* a, const String& b) {
  return String(std::string(a != nullptr ? a : "") + b.s);
}
inline String operator+(const String& a, const char* b) {
  return String(a.s + (b != nullptr ? b : ""));
}

// ── Pin mode / value constants ────────────────────────────────────────────────
constexpr int INPUT  = 0;
constexpr int OUTPUT = 1;
constexpr int INPUT_PULLUP = 2;
constexpr int HIGH   = 1;
constexpr int LOW    = 0;
constexpr int CHANGE = 1;
constexpr int RISING = 2;
constexpr int FALLING = 3;

// Pin aliases used by config.h (map to arbitrary ints for native)
constexpr int A0 = 14, A1 = 15, A2 = 16, A3 = 17;
constexpr int D2 = 2,  D3 = 3,  D4 = 4;
constexpr int D5 = 5,  D6 = 6,  D7 = 7;
constexpr int D9 = 9,  D10 = 10;

// ── HAL state — defined in hal.cpp, declared extern here ─────────────────────
extern volatile int  g_hal_digital_out[32];
extern volatile int  g_hal_digital_in[32];
extern volatile int  g_hal_analog_out[32];  // 0-255 PWM value

void pinMode(int pin, int mode);
void analogWrite(int pin, int value);
void analogWriteFrequency(uint32_t frequency);
int  digitalRead(int pin);
void digitalWrite(int pin, int value);
void delay(unsigned long ms);
unsigned long millis();
void noInterrupts();
void interrupts();

inline int digitalPinToInterrupt(int pin) { return pin; }
using HalIsrCb = void (*)(void*);
void attachInterruptArg(int pin, HalIsrCb cb, void* arg, int mode);
/** Set input level and fire simulated GPIO ISRs on edges (native CIL). */
void hotbox_hal_set_digital_in(int pin, int level);

// Serial stub — discarded in native mode (acks / debug). Defined inhal.cpp.
struct _SerialStub {
    template <typename T>
    _SerialStub& print(T) {
        return *this;
    }
    template <typename T>
    _SerialStub& println(T) {
        return *this;
    }
    template <typename T, typename U>
    _SerialStub& print(T, U) {
        return *this;
    }
    void println() {}
    void begin(int) {}
    void flush() {}
    bool available() { return false; }
    char read() { return 0; }
};
extern _SerialStub Serial;
