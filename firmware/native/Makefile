# Build native C-in-the-loop (CIL) shared libraries — one per mirror node.
# Each dylib/so has its own HAL + firmware statics so multi-node SITL works.
#
# Usage:
#   cd firmware/native && make
#   cd firmware/native && make NODE_IDS="0 1"   # subset
#   cd firmware/native && make clean
#
# Output: libfirmware_cil_node{N}.dylib (macOS) or .so (Linux)

CXX      ?= g++
NODE_IDS ?= 0 1 2

CXXFLAGS_COMMON = -std=c++17 -O3 -DNDEBUG -fPIC \
            -I include \
            -I ../include \
            -D NATIVE_CIL

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
    LIB_EXT  = .dylib
    LDFLAGS  = -dynamiclib
else
    LIB_EXT  = .so
    LDFLAGS  = -shared
endif

LIBS = $(foreach id,$(NODE_IDS),libfirmware_cil_node$(id)$(LIB_EXT))
SRCS = firmware_cil.cpp hal.cpp ../src/axis.cpp ../src/protocol.cpp

.PHONY: all clean

all: $(LIBS)

libfirmware_cil_node%$(LIB_EXT): $(SRCS)
	$(CXX) $(CXXFLAGS_COMMON) -D HOTBOX_NODE_ID=$* $(LDFLAGS) -o $@ $(SRCS)

clean:
	rm -f libfirmware_cil_node*$(LIB_EXT) libfirmware_cil$(LIB_EXT) *.o
