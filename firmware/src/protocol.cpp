#include "protocol.h"

namespace hotbox {

ProtocolHandler::ProtocolHandler(MirrorMount* mount) : mount_(mount) {}

void ProtocolHandler::emitAck(const char* command, bool ok) {
  Serial.print("{\"node_id\":");
  Serial.print(HOTBOX_NODE_ID);
  Serial.print(",\"type\":\"ack\",\"command\":\"");
  Serial.print(command);
  Serial.print("\",\"ok\":");
  Serial.print(ok ? "true" : "false");
  Serial.println("}");
}

void ProtocolHandler::emitStatus() {
  Serial.print("{\"node_id\":");
  Serial.print(HOTBOX_NODE_ID);
  Serial.print(",\"type\":\"status\",\"homed\":");
  Serial.print(mount_->isHomed() ? "true" : "false");
  Serial.print(",\"azimuth_deg\":");
  Serial.print(mount_->azimuthDeg(), 3);
  Serial.print(",\"elevation_deg\":");
  Serial.print(mount_->elevationDeg(), 3);
  Serial.print(",\"mode\":\"");
  Serial.print(mount_->modeText());
  Serial.print("\",\"fault\":");
  if (mount_->faultText() == nullptr) {
    Serial.print("null");
  } else {
    Serial.print("\"");
    Serial.print(mount_->faultText());
    Serial.print("\"");
  }
  Serial.println("}");
}

void ProtocolHandler::handleLine(const String& line) {
  if (line.indexOf("\"command\":\"home\"") >= 0) {
    mount_->home();
    emitAck("home", true);
    return;
  }
  if (line.indexOf("\"command\":\"stop\"") >= 0) {
    mount_->stop();
    emitAck("stop", true);
    return;
  }
  if (line.indexOf("\"command\":\"clear_error\"") >= 0) {
    mount_->clearError();
    emitAck("clear_error", true);
    return;
  }
  if (line.indexOf("\"command\":\"get_status\"") >= 0) {
    emitStatus();
    return;
  }
  if (line.indexOf("\"command\":\"set_target\"") >= 0) {
    int az_index = line.indexOf("\"azimuth_deg\":");
    int el_index = line.indexOf("\"elevation_deg\":");
    float az = 0.0f;
    float el = 0.0f;
    if (az_index >= 0) {
      az = line.substring(az_index + 14).toFloat();
    }
    if (el_index >= 0) {
      el = line.substring(el_index + 16).toFloat();
    }
    mount_->setTarget(az, el);
    emitAck("set_target", true);
    return;
  }
  emitAck("unknown", false);
}

}  // namespace hotbox
