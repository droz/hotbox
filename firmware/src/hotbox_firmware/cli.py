"""CLI entry points for building and uploading Nano ESP32 firmware."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Must match mount node_id values in config/system.yaml.
ALLOWED_NODE_IDS = frozenset({0, 1, 2})


def firmware_root() -> Path:
    """Return the ``firmware/`` directory (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def env_for_node(node_id: int) -> str:
    if node_id not in ALLOWED_NODE_IDS:
        allowed = ", ".join(str(n) for n in sorted(ALLOWED_NODE_IDS))
        raise ValueError(f"node-id must be one of {{{allowed}}}, got {node_id}")
    return f"nano_esp32_node{node_id}"


def run_pio(args: list[str]) -> int:
    """Run PlatformIO from this project's environment."""
    cmd = [sys.executable, "-m", "platformio", *args]
    print("+", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=firmware_root())
    return int(completed.returncode)


def _strip_remainder(extra: list[str]) -> list[str]:
    if extra and extra[0] == "--":
        return extra[1:]
    return list(extra)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--node-id",
        type=int,
        required=True,
        choices=sorted(ALLOWED_NODE_IDS),
        help="Mirror node id baked into firmware (matches config/system.yaml mounts).",
    )
    parser.add_argument(
        "pio_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to PlatformIO after --",
    )


def build_firmware(node_id: int, *, pio_args: list[str] | None = None) -> int:
    """Build firmware for ``node_id`` into its dedicated PlatformIO env."""
    env = env_for_node(node_id)
    return run_pio(["run", "-e", env, *list(pio_args or ())])


def upload_firmware(
    node_id: int,
    *,
    port: str | None = None,
    pio_args: list[str] | None = None,
) -> int:
    """Build firmware for ``node_id``, then upload that exact build."""
    # Hard dependency: never upload without a successful build for this node.
    build_rc = build_firmware(node_id, pio_args=pio_args)
    if build_rc != 0:
        return build_rc

    env = env_for_node(node_id)
    cmd = ["run", "-t", "upload", "-e", env]
    if port:
        cmd.extend(["--upload-port", port])
    cmd.extend(list(pio_args or ()))
    return run_pio(cmd)


def build_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build Hot-Box firmware with PlatformIO.")
    _add_common_args(parser)
    args = parser.parse_args(argv)
    raise SystemExit(build_firmware(args.node_id, pio_args=_strip_remainder(args.pio_args)))


def upload_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build then upload Hot-Box firmware (build is always run first)."
    )
    _add_common_args(parser)
    parser.add_argument(
        "--port",
        "-p",
        default=None,
        help="Serial port (e.g. /dev/cu.usbmodem1101). Auto-detect if omitted.",
    )
    args = parser.parse_args(argv)
    raise SystemExit(
        upload_firmware(
            args.node_id,
            port=args.port,
            pio_args=_strip_remainder(args.pio_args),
        )
    )
