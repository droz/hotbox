# Build the native CIL shared library without PlatformIO.
# Usage:
#   cd firmware/native && make
#   cd firmware/native && make clean
#
# Output: libfirmware_cil.dylib (macOS) or libfirmware_cil.so (Linux)
# The Python harness looks for it in the same directory as this Makefile,
# OR in firmware/.pio/build/native_cil/ (PlatformIO output).

CXX      ?= g++
CXXFLAGS  = -std=c++17 -O0 -fPIC -g \
            -I include \
            -I ../include \
            -D HOTBOX_NODE_ID=0 \
            -D NATIVE_CIL

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
    LIB      = libfirmware_cil.dylib
    LDFLAGS  = -dynamiclib
else
    LIB      = libfirmware_cil.so
    LDFLAGS  = -shared
endif

.PHONY: all clean

all: $(LIB)

$(LIB): firmware_cil.cpp hal.cpp ../src/axis.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ firmware_cil.cpp hal.cpp ../src/axis.cpp

clean:
	rm -f $(LIB) *.o
