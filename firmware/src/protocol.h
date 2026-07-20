#pragma once

#include "axis.h"

namespace hotbox {

class ProtocolHandler {
 public:
  explicit ProtocolHandler(MirrorMount* mount);

  void handleLine(const String& line);
  void emitStatus();

 private:
  void emitAck(const char* command, bool ok);
  MirrorMount* mount_;
};

}  // namespace hotbox
