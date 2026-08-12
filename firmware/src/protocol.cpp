#include "protocol.h"

#include <cmath>
#include <cstdio>
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

bool readStringField(const String& line, const char* key, String* out) {
  String needle = String("\"") + key + "\":\"";
  int index = line.indexOf(needle);
  if (index < 0) {
    needle = String("\"") + key + "\": \"";
    index = line.indexOf(needle);
  }
  if (index < 0) {
    return false;
  }
  const int start = index + needle.length();
  const int end = line.indexOf('"', start);
  if (end < 0) {
    return false;
  }
  *out = line.substring(start, end);
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
  if (strcmp(mode, "velocity") == 0) {
    return kModeVelocity;
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

int ProtocolHandler::formatStatus(char* buf, size_t buflen) const {
  if (buf == nullptr || buflen < 32) {
    return -1;
  }
  const char* fault = mount_->faultText();
  char az_width[24];
  char el_width[24];
  if (mount_->azimuthHasHallWidth()) {
    snprintf(az_width, sizeof(az_width), "%.3f",
             static_cast<double>(mount_->azimuthHallWidthDeg()));
  } else {
    snprintf(az_width, sizeof(az_width), "null");
  }
  if (mount_->elevationHasHallWidth()) {
    snprintf(el_width, sizeof(el_width), "%.3f",
             static_cast<double>(mount_->elevationHallWidthDeg()));
  } else {
    snprintf(el_width, sizeof(el_width), "null");
  }
  const int n = snprintf(
      buf,
      buflen,
      "{\"node_id\":%d,\"type\":\"status\",\"azimuth_home\":\"%s\",\"elevation_home\":\"%s\","
      "\"azimuth_deg\":%.3f,\"elevation_deg\":%.3f,"
      "\"azimuth_integral\":%.4f,\"elevation_integral\":%.4f,"
      "\"pid_kp\":%.4f,\"pid_ki\":%.4f,\"pid_kd\":%.4f,"
      "\"pid_velocity_kp\":%.4f,\"pid_velocity_ki\":%.4f,\"pid_velocity_kd\":%.4f,"
      "\"mode\":\"%s\",\"az_hall\":%s,\"el_hall\":%s,"
      "\"az_hall_width_deg\":%s,\"el_hall_width_deg\":%s,"
      "\"fault\":%s%s%s}",
      HOTBOX_NODE_ID,
      mount_->azimuthHomeState(),
      mount_->elevationHomeState(),
      static_cast<double>(mount_->azimuthDeg()),
      static_cast<double>(mount_->elevationDeg()),
      static_cast<double>(mount_->azimuthIntegralTerm()),
      static_cast<double>(mount_->elevationIntegralTerm()),
      static_cast<double>(mount_->pidKp()),
      static_cast<double>(mount_->pidKi()),
      static_cast<double>(mount_->pidKd()),
      static_cast<double>(mount_->pidVelocityKp()),
      static_cast<double>(mount_->pidVelocityKi()),
      static_cast<double>(mount_->pidVelocityKd()),
      mount_->modeText(),
      mount_->azimuthHallTriggered() ? "true" : "false",
      mount_->elevationHallTriggered() ? "true" : "false",
      az_width,
      el_width,
      fault == nullptr ? "null" : "\"",
      fault == nullptr ? "" : fault,
      fault == nullptr ? "" : "\"");
  if (n < 0 || static_cast<size_t>(n) >= buflen) {
    return -1;
  }
  return n;
}

void ProtocolHandler::emitStatus() {
  char buf[768];
  if (formatStatus(buf, sizeof(buf)) < 0) {
    return;
  }
  Serial.println(buf);
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
    case kCanCmdSetVelocity: {
      // Same packing as set_target: int16 centi-units (°/s * 100).
      if (len < 5) {
        return false;
      }
      const int16_t az_c = static_cast<int16_t>(data[1] | (data[2] << 8));
      const int16_t el_c = static_cast<int16_t>(data[3] | (data[4] << 8));
      mount_->setVelocity(az_c / 100.0f, el_c / 100.0f);
      return false;
    }
    case kCanCmdSetPidPos: {
      // Position kp,ki,kd as int16 milli-units (value * 1000).
      if (len < 7) {
        return false;
      }
      const int16_t kp_m = static_cast<int16_t>(data[1] | (data[2] << 8));
      const int16_t ki_m = static_cast<int16_t>(data[3] | (data[4] << 8));
      const int16_t kd_m = static_cast<int16_t>(data[5] | (data[6] << 8));
      mount_->setPid(kp_m / 1000.0f, ki_m / 1000.0f, kd_m / 1000.0f);
      return false;
    }
    case kCanCmdSetPidVel: {
      // Velocity kp,ki,kd as int16 milli-units (value * 1000).
      if (len < 7) {
        return false;
      }
      const int16_t kp_m = static_cast<int16_t>(data[1] | (data[2] << 8));
      const int16_t ki_m = static_cast<int16_t>(data[3] | (data[4] << 8));
      const int16_t kd_m = static_cast<int16_t>(data[5] | (data[6] << 8));
      mount_->setVelocityPid(kp_m / 1000.0f, ki_m / 1000.0f, kd_m / 1000.0f);
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
    String axis = "both";
    readStringField(line, "axis", &axis);
    axis.toLowerCase();
    if (axis == "az" || axis == "azimuth") {
      mount_->homeAzimuth();
    } else if (axis == "el" || axis == "elevation") {
      mount_->homeElevation();
    } else {
      mount_->home();
    }
    // Immediate ack = command accepted / homing started. Completion is
    // visible via azimuth_home / elevation_home in status.
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
    emitAck("set_target", mount_->setTarget(az, el));
    return;
  }
  if (hasCommand(line, "set_velocity")) {
    float az = 0.0f;
    float el = 0.0f;
    readFloatField(line, "azimuth_deg_s", &az);
    readFloatField(line, "elevation_deg_s", &el);
    emitAck("set_velocity", mount_->setVelocity(az, el));
    return;
  }
  if (hasCommand(line, "set_pid_pos")) {
    float kp = mount_->pidKp();
    float ki = mount_->pidKi();
    float kd = mount_->pidKd();
    readFloatField(line, "kp", &kp);
    readFloatField(line, "ki", &ki);
    readFloatField(line, "kd", &kd);
    mount_->setPid(kp, ki, kd);
    emitAck("set_pid_pos", true);
    return;
  }
  if (hasCommand(line, "set_pid_vel")) {
    float kp = mount_->pidVelocityKp();
    float ki = mount_->pidVelocityKi();
    float kd = mount_->pidVelocityKd();
    readFloatField(line, "kp", &kp);
    readFloatField(line, "ki", &ki);
    readFloatField(line, "kd", &kd);
    mount_->setVelocityPid(kp, ki, kd);
    emitAck("set_pid_vel", true);
    return;
  }
  emitAck("unknown", false);
}

}  // namespace hotbox
