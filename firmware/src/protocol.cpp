#include "protocol.h"

#include <cmath>
#include <cstring>

namespace hotbox {

ProtocolHandler::ProtocolHandler(MirrorMount* mount) : mount_(mount) {}

uint8_t ProtocolHandler::modeId() const {
  const char* mode = mount_->modeText();
  if (mode == nullptr) {
    return kModeIdle;
  }
  if (strcmp(mode, "homing") == 0) {
    return kModeHoming;
  }
  if (strcmp(mode, "position") == 0) {
    return kModePosition;
  }
  if (strcmp(mode, "fault") == 0) {
    return kModeFault;
  }
  return kModeIdle;
}

void ProtocolHandler::emitAck(const char* command, bool ok) {
  Serial.print("{\"node_id\":");
  Serial.print(HOTBOX_NODE_ID);
  Serial.print(",\"type\":\"ack\",\"command\":\"");
  Serial.print(command);
  Serial.print("\",\"ok\":");
  Serial.print(ok ? "true" : "false");
  Serial.println("}");
  Serial.flush();
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
  Serial.print("\",\"az_hall\":");
  Serial.print(mount_->azimuthHallTriggered() ? "true" : "false");
  Serial.print(",\"el_hall\":");
  Serial.print(mount_->elevationHallTriggered() ? "true" : "false");
  Serial.print(",\"fault\":");
  if (mount_->faultText() == nullptr) {
    Serial.print("null");
  } else {
    Serial.print("\"");
    Serial.print(mount_->faultText());
    Serial.print("\"");
  }
  Serial.println("}");
  Serial.flush();
}

void ProtocolHandler::fillStatusCan(uint8_t out[8]) const {
  const int16_t az = static_cast<int16_t>(lroundf(mount_->azimuthDeg() * 100.0f));
  const int16_t el = static_cast<int16_t>(lroundf(mount_->elevationDeg() * 100.0f));
  out[0] = kCanCmdGetStatus;
  out[1] = mount_->isHomed() ? 1 : 0;
  out[2] = static_cast<uint8_t>(az & 0xff);
  out[3] = static_cast<uint8_t>((az >> 8) & 0xff);
  out[4] = static_cast<uint8_t>(el & 0xff);
  out[5] = static_cast<uint8_t>((el >> 8) & 0xff);
  out[6] = modeId();
  out[7] = 0;
}

bool ProtocolHandler::handleBinary(const uint8_t* data, size_t len) {
  if (data == nullptr || len < 1) {
    return false;
  }
  switch (data[0]) {
    case kCanCmdHome:
      mount_->home();
      return false;
    case kCanCmdStop:
      mount_->stop();
      return false;
    case kCanCmdClearError:
      mount_->clearError();
      return false;
    case kCanCmdSetTarget: {
      if (len < 5) {
        return false;
      }
      const int16_t az_c = static_cast<int16_t>(data[1] | (data[2] << 8));
      const int16_t el_c = static_cast<int16_t>(data[3] | (data[4] << 8));
      mount_->setTarget(az_c / 100.0f, el_c / 100.0f);
      return false;
    }
    case kCanCmdGetStatus:
      return true;
    default:
      return false;
  }
}

void ProtocolHandler::handleLine(const String& line) {
  // Accept both compact (`"command":"home"`) and spaced (`"command": "home"`) JSON.
  auto hasCommand = [&](const char* name) -> bool {
    String a = String("\"command\":\"") + name + "\"";
    String b = String("\"command\": \"") + name + "\"";
    return line.indexOf(a) >= 0 || line.indexOf(b) >= 0;
  };

  if (hasCommand("home")) {
    mount_->home();
    // Immediate ack = command accepted / homing started. Completion is
    // visible via mode "homing" → "idle" and homed=true (status push + polls).
    emitAck("home", true);
    return;
  }
  if (hasCommand("stop")) {
    mount_->stop();
    emitAck("stop", true);
    return;
  }
  if (hasCommand("clear_error")) {
    mount_->clearError();
    emitAck("clear_error", true);
    return;
  }
  if (hasCommand("get_status")) {
    emitStatus();
    return;
  }
  if (hasCommand("set_target")) {
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
