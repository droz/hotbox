#include "protocol.h"

#include <cmath>
#include <cstring>

namespace hotbox {
namespace {

bool hasCommand(const String& line, const char* name) {
  String a = String("\"command\":\"") + name + "\"";
  String b = String("\"command\": \"") + name + "\"";
  return line.indexOf(a) >= 0 || line.indexOf(b) >= 0;
}

bool readFloatField(const String& line, const char* key, float* out) {
  String needle = String("\"") + key + "\":";
  int index = line.indexOf(needle);
  if (index < 0) {
    needle = String("\"") + key + "\": ";
    index = line.indexOf(needle);
  }
  if (index < 0) {
    return false;
  }
  *out = line.substring(index + needle.length()).toFloat();
  return true;
}

}  // namespace

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
  Serial.print(",\"azimuth_integral\":");
  Serial.print(mount_->azimuthIntegralTerm(), 4);
  Serial.print(",\"elevation_integral\":");
  Serial.print(mount_->elevationIntegralTerm(), 4);
  Serial.print(",\"pid_kp\":");
  Serial.print(mount_->pidKp(), 4);
  Serial.print(",\"pid_ki\":");
  Serial.print(mount_->pidKi(), 4);
  Serial.print(",\"pid_kd\":");
  Serial.print(mount_->pidKd(), 4);
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
    case kCanCmdReset:
      mount_->reset();
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
    case kCanCmdSetPid: {
      // kp,ki,kd as int16 milli-units (value * 1000).
      if (len < 7) {
        return false;
      }
      const int16_t kp_m = static_cast<int16_t>(data[1] | (data[2] << 8));
      const int16_t ki_m = static_cast<int16_t>(data[3] | (data[4] << 8));
      const int16_t kd_m = static_cast<int16_t>(data[5] | (data[6] << 8));
      mount_->setPid(kp_m / 1000.0f, ki_m / 1000.0f, kd_m / 1000.0f);
      return false;
    }
    case kCanCmdGetStatus:
      return true;
    default:
      return false;
  }
}

void ProtocolHandler::handleLine(const String& line) {
  if (hasCommand(line, "home")) {
    mount_->home();
    // Immediate ack = command accepted / homing started. Completion is
    // visible via mode "homing" → "idle" and homed=true (status push + polls).
    emitAck("home", true);
    return;
  }
  if (hasCommand(line, "stop")) {
    mount_->stop();
    emitAck("stop", true);
    return;
  }
  if (hasCommand(line, "clear_error")) {
    mount_->clearError();
    emitAck("clear_error", true);
    return;
  }
  if (hasCommand(line, "reset")) {
    emitAck("reset", true);
    mount_->reset();  // may reboot; ack is flushed first
    return;
  }
  if (hasCommand(line, "get_status")) {
    emitStatus();
    return;
  }
  if (hasCommand(line, "set_target")) {
    float az = 0.0f;
    float el = 0.0f;
    readFloatField(line, "azimuth_deg", &az);
    readFloatField(line, "elevation_deg", &el);
    mount_->setTarget(az, el);
    emitAck("set_target", true);
    return;
  }
  if (hasCommand(line, "set_pid")) {
    float kp = mount_->pidKp();
    float ki = mount_->pidKi();
    float kd = mount_->pidKd();
    readFloatField(line, "kp", &kp);
    readFloatField(line, "ki", &ki);
    readFloatField(line, "kd", &kd);
    mount_->setPid(kp, ki, kd);
    emitAck("set_pid", true);
    return;
  }
  emitAck("unknown", false);
}

}  // namespace hotbox
