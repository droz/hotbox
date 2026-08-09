from __future__ import annotations

from types import SimpleNamespace

import pytest

from hotbox_controller.usb_discover import (
    list_nano_esp32_ports,
    parse_usb_ports_spec,
    resolve_usb_port_map,
    usb_ports_from_env,
)


def test_parse_usb_ports_spec() -> None:
    assert parse_usb_ports_spec("0:/dev/ttyACM0,1:/dev/ttyACM1") == {
        0: "/dev/ttyACM0",
        1: "/dev/ttyACM1",
    }
    assert parse_usb_ports_spec(" 2:/dev/cu.usbmodem1 ") == {2: "/dev/cu.usbmodem1"}


def test_usb_ports_from_env() -> None:
    assert usb_ports_from_env({}) == {}
    assert usb_ports_from_env({"HOTBOX_USB_PORTS": "0:/dev/ttyACM0"}) == {0: "/dev/ttyACM0"}


def test_list_nano_esp32_ports_filters_vid_pid() -> None:
    ports = list_nano_esp32_ports(
        [
            SimpleNamespace(device="/dev/ttyACM0", vid=0x2341, pid=0x0070, description="Nano ESP32"),
            SimpleNamespace(device="/dev/ttyUSB0", vid=0x0403, pid=0x6001, description="FTDI"),
            SimpleNamespace(
                device="/dev/ttyACM1",
                vid=0x303A,
                pid=0x1001,
                description="Espressif USB JTAG/serial debug unit",
            ),
        ]
    )
    assert ports == ["/dev/ttyACM0"]


def test_resolve_single_auto_maps_to_node_zero() -> None:
    mapping = resolve_usb_port_map(
        None,
        comports=[
            SimpleNamespace(device="/dev/cu.usbmodem1101", vid=0x2341, pid=0x0070, description="Nano ESP32"),
        ],
    )
    assert mapping == {0: "/dev/cu.usbmodem1101"}


def test_resolve_multiple_requires_explicit() -> None:
    mapping = resolve_usb_port_map(
        None,
        comports=[
            SimpleNamespace(device="/dev/ttyACM0", vid=0x2341, pid=0x0070, description="Nano ESP32"),
            SimpleNamespace(device="/dev/ttyACM1", vid=0x2341, pid=0x0070, description="Nano ESP32"),
        ],
    )
    assert mapping == {}


def test_resolve_explicit_wins() -> None:
    mapping = resolve_usb_port_map(
        {1: "/dev/ttyACM9"},
        comports=[
            SimpleNamespace(device="/dev/ttyACM0", vid=0x2341, pid=0x0070, description="Nano ESP32"),
        ],
    )
    assert mapping == {1: "/dev/ttyACM9"}


def test_parse_usb_ports_rejects_bad() -> None:
    with pytest.raises(ValueError):
        parse_usb_ports_spec("not-a-mapping")
