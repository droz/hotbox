#pragma once

#include <cstddef>
#include <cstdint>

#include "axis.h"

namespace hotbox {

// Wire command IDs — must match controller `protocol.CommandId`.
enum CanCommandId : uint8_t {
  kCanCmdHome = 1,
  kCanCmdStop = 2,
  kCanCmdSetTarget = 3,
  kCanCmdGetStatus = 4,
  kCanCmdReset = 5,
  kCanCmdClearError = 6,
  kCanCmdSetPid = 7,
  kCanCmdSetVelocity = 8,
};

constexpr uint32_t kCanCmdBaseId = 0x100;
constexpr uint32_t kCanRspBaseId = 0x200;
constexpr uint32_t kCanBitrate = 250000;

// Firmware axis modes — must match controller `FIRMWARE_MODE_IDS`.
enum FirmwareModeId : uint8_t {
  kModeIdle = 0,
  kModeHoming = 1,
  kModePosition = 2,
  kModeFault = 3,
  kModeVelocity = 4,
};

class ProtocolHandler {
 public:
  explicit ProtocolHandler(MirrorMount* mount);

  void handleLine(const String& line);
  void emitStatus();
  /** Write one status JSON object (no trailing newline) into ``buf``. Returns length or -1. */
  int formatStatus(char* buf, size_t buflen) const;

  /** Apply a binary CAN command payload (no transport). Returns true if GET_STATUS. */
  bool handleBinary(const uint8_t* data, size_t len);

  /** Pack status into the 8-byte CAN response format used by the host. */
  void fillStatusCan(uint8_t out[8]) const;

 private:
  void emitAck(const char* command, bool ok);
  uint8_t modeId() const;
  MirrorMount* mount_;
};

}  // namespace hotbox
