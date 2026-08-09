#pragma once

#include "protocol.h"

namespace hotbox {

/**
 * ESP32-S3 TWAI (CAN) transport for the mirror wire protocol.
 *
 * Soft-fails when no transceiver is present so USB serial bring-up still works.
 */
class CanBus {
 public:
  explicit CanBus(ProtocolHandler* protocol);

  /** Install/start TWAI. Returns false if unavailable (USB-only ok). */
  bool begin();

  /** Poll RX queue; reply to get_status on the response arbitration id. */
  void poll();

  bool enabled() const { return enabled_; }

 private:
  bool sendStatus();

  ProtocolHandler* protocol_;
  bool enabled_ = false;
};

}  // namespace hotbox
