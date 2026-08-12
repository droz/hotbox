from __future__ import annotations

import uvicorn

from .app import ControllerApplication


def main() -> None:
    app = ControllerApplication()
    app.startup()
    # Closed-loop Track/Park (and jog rate streaming) need a periodic tick.
    # SITL runs control_tick from its physics thread instead.
    app.start_control_loop()
    try:
        uvicorn.run(app.fastapi, host=app.config.web_host, port=app.config.web_port)
    finally:
        app.stop_control_loop()


if __name__ == "__main__":
    main()
