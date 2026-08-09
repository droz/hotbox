#include <Arduino.h>

#include "axis.h"
#include "can_bus.h"
#include "config.h"
#include "protocol.h"

#include <cstring>

hotbox::MirrorMount g_mount;
hotbox::ProtocolHandler g_protocol(&g_mount);
hotbox::CanBus g_can(&g_protocol);

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
  Serial.flush();
}

void loop() {
  static String line;
  static unsigned long last_update_ms = 0;
  static bool was_homing = false;
  const unsigned long now_ms = millis();
  const float dt_s = (now_ms - last_update_ms) / 1000.0f;

  while (Serial.available() > 0) {
    const char ch = static_cast<char>(Serial.read());
    if (ch == '\n') {
      g_protocol.handleLine(line);
      line = "";
    } else {
      line += ch;
    }
  }

  g_can.poll();

  if (dt_s >= hotbox::kControlPeriodS) {
    g_mount.update(dt_s);
    last_update_ms = now_ms;

    const bool homing = std::strcmp(g_mount.modeText(), "homing") == 0;
    if (was_homing && !homing) {
      // Homing finished (success → idle/homed, or fault). Push status so the
      // host does not have to wait for the next poll.
      g_protocol.emitStatus();
    }
    was_homing = homing;
  }
}
