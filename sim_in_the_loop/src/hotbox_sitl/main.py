from __future__ import annotations

import argparse

from hotbox_sitl.harness import SitlHarness


def main() -> None:
    parser = argparse.ArgumentParser(description="Hot-Box in-the-loop simulator with live web UI")
    parser.add_argument("--host", default="0.0.0.0", help="Web UI bind address (0.0.0.0 for LAN access)")
    parser.add_argument("--port", type=int, default=8000, help="Web UI port")
    parser.add_argument("--dt", type=float, default=0.05, help="Simulation timestep seconds")
    parser.add_argument(
        "--batch-seconds",
        type=float,
        default=None,
        help="If set, run a headless batch for this many seconds instead of the live UI",
    )
    args = parser.parse_args()

    harness = SitlHarness(host=args.host, port=args.port, dt_s=args.dt)
    if args.batch_seconds is not None:
        harness.run(seconds=args.batch_seconds, dt_s=args.dt)
        return
    harness.run_forever()


if __name__ == "__main__":
    main()
