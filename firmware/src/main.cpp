#include <Arduino.h>

#include "axis.h"
#include "config.h"
#include "protocol.h"

hotbox::MirrorMount g_mount;
hotbox::ProtocolHandler g_protocol(&g_mount);

void setup() {
  Serial.begin(115200);
  delay(200);
  g_mount.begin();
  Serial.print("{\"hotbox\":\"mirror_firmware\",\"transport\":\"usb\",\"node_id\":");
  Serial.print(HOTBOX_NODE_ID);
  Serial.println("}");
}

void loop() {
  static String line;
  static unsigned long last_update_ms = 0;
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

  if (dt_s >= kControlPeriodS) {
    g_mount.update(dt_s);
    last_update_ms = now_ms;
  }
}
