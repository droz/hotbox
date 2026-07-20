from __future__ import annotations

import uvicorn

from .app import ControllerApplication


def main() -> None:
    app = ControllerApplication()
    app.startup()
    uvicorn.run(app.fastapi, host=app.config.web_host, port=app.config.web_port)


if __name__ == "__main__":
    main()
