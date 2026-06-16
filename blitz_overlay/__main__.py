"""`python -m blitz_overlay` / `blitz-overlay` — start engine + browser host (one command)."""
from __future__ import annotations

import threading
import webbrowser

import uvicorn

from blitz_overlay.config import OverlayConfig
from blitz_overlay.server import create_app


def main() -> None:
    config = OverlayConfig.from_env()
    app = create_app(config)
    url = f"http://{config.host}:{config.port}/"
    print("\n  Blitz Live Consensus Overlay")
    print(f"  → open {url} and allow camera access")
    print("  Raw video never leaves your device; only feature vectors reach the engine.\n")
    if config.open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=config.host, port=config.port, log_level="warning")


if __name__ == "__main__":
    main()
