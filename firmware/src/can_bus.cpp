#include "can_bus.h"

#include "config.h"

#include "driver/gpio.h"
#include "driver/twai.h"

namespace hotbox {

CanBus::CanBus(ProtocolHandler* protocol) : protocol_(protocol) {}

bool CanBus::begin() {
  const gpio_num_t tx_pin = static_cast<gpio_num_t>(kCanTxPin);
  const gpio_num_t rx_pin = static_cast<gpio_num_t>(kCanRxPin);

  twai_general_config_t g_config =
      TWAI_GENERAL_CONFIG_DEFAULT(tx_pin, rx_pin, TWAI_MODE_NORMAL);
  g_config.rx_queue_len = 16;
  g_config.tx_queue_len = 8;

  // Match controller TransportConfig.can_bitrate (250000).
  twai_timing_config_t t_config = TWAI_TIMING_CONFIG_250KBITS();
  twai_filter_config_t f_config = TWAI_FILTER_CONFIG_ACCEPT_ALL();

  if (twai_driver_install(&g_config, &t_config, &f_config) != ESP_OK) {
    Serial.println("{\"hotbox\":\"can\",\"ok\":false,\"error\":\"install\"}");
    enabled_ = false;
    return false;
  }
  if (twai_start() != ESP_OK) {
    twai_driver_uninstall();
    Serial.println("{\"hotbox\":\"can\",\"ok\":false,\"error\":\"start\"}");
    enabled_ = false;
    return false;
  }

  enabled_ = true;
  Serial.print("{\"hotbox\":\"can\",\"ok\":true,\"bitrate\":");
  Serial.print(kCanBitrate);
  Serial.print(",\"node_id\":");
  Serial.print(HOTBOX_NODE_ID);
  Serial.println("}");
  return true;
}

bool CanBus::sendStatus() {
  if (!enabled_ || protocol_ == nullptr) {
    return false;
  }
  twai_message_t msg = {};
  msg.identifier = kCanRspBaseId + HOTBOX_NODE_ID;
  msg.extd = 0;
  msg.rtr = 0;
  msg.data_length_code = 8;
  protocol_->fillStatusCan(msg.data);
  return twai_transmit(&msg, pdMS_TO_TICKS(20)) == ESP_OK;
}

void CanBus::poll() {
  if (!enabled_ || protocol_ == nullptr) {
    return;
  }

  twai_message_t msg;
  while (twai_receive(&msg, 0) == ESP_OK) {
    if (msg.rtr || msg.extd) {
      continue;
    }
    if (msg.identifier != (kCanCmdBaseId + HOTBOX_NODE_ID)) {
      continue;
    }
    const bool want_status = protocol_->handleBinary(msg.data, msg.data_length_code);
    if (want_status) {
      sendStatus();
    }
  }
}

}  // namespace hotbox
