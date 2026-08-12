#include <Arduino.h>

#include "axis.h"
#include "can_bus.h"
#include "config.h"
#include "protocol.h"

#include <cstring>

hotbox::MirrorMount g_mount;
hotbox::ProtocolHandler g_protocol(&g_mount);
hotbox::CanBus g_can(&g_protocol);

namespace {

constexpr size_t kSerialLineMax = 512;
constexpr int kSerialBytesPerLoop = 96;

}  // namespace

void setup() {
  Serial.begin(115200);
  // USB CDC: wait briefly for the host to open the port after reset-on-open.
  const unsigned long wait_start = millis();
  while (!Serial && (millis() - wait_start) < 3000) {
    delay(10);
  }
  delay(200);
  g_mount.begin();
  g_can.begin();  // Soft-fails without a transceiver; USB still works.
  Serial.print("{\"hotbox\":\"mirror_firmware\",\"transport\":\"usb\",\"node_id\":");
  Serial.print(HOTBOX_NODE_ID);
  Serial.print(",\"az_hall\":");
  Serial.print(g_mount.azimuthHallTriggered() ? "true" : "false");
  Serial.print(",\"el_hall\":");
  Serial.print(g_mount.elevationHallTriggered() ? "true" : "false");
  Serial.println("}");
}

void loop() {
  static char line_buf[kSerialLineMax];
  static size_t line_len = 0;
  static unsigned long last_update_ms = 0;
  static bool was_homing = false;
  const unsigned long now_ms = millis();
  const float dt_s = (now_ms - last_update_ms) / 1000.0f;

  // Control first: never let USB RX/TX starve the motor loop (runaway PWM).
  if (last_update_ms == 0 || dt_s >= hotbox::kControlPeriodS) {
    const float step_dt = (last_update_ms == 0) ? hotbox::kControlPeriodS : dt_s;
    g_mount.update(step_dt);
    last_update_ms = now_ms;

    const bool homing = std::strcmp(g_mount.modeText(), "homing") == 0;
    if (was_homing && !homing) {
      // Homing finished (success → idle/homed, or fault). Push status so the
      // host does not have to wait for the next poll.
      g_protocol.emitStatus();
    }
    was_homing = homing;
  }

  int budget = kSerialBytesPerLoop;
  while (budget-- > 0 && Serial.available() > 0) {
    const char ch = static_cast<char>(Serial.read());
    if (ch == '\n' || ch == '\r') {
      if (line_len > 0) {
        line_buf[line_len] = '\0';
        g_protocol.handleLine(String(line_buf));
        line_len = 0;
      }
    } else if (line_len + 1 < kSerialLineMax) {
      line_buf[line_len++] = ch;
    } else {
      // Overflow / garbage: drop the line rather than grow forever.
      line_len = 0;
    }
  }

  g_can.poll();
}
