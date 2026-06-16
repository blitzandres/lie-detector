"""FastAPI app: serves the browser overlay (static) and the /ws feature-frame endpoint.

One process = engine + browser host, so the whole thing starts with one command.
Raw video never reaches here — only feature frames (spec §3).
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from blitz_overlay.config import OverlayConfig
from blitz_overlay.pipeline import OverlaySession

WEB_DIR = Path(__file__).resolve().parents[1] / "apps" / "overlay-web"


def create_app(config: OverlayConfig | None = None) -> FastAPI:
    config = config or OverlayConfig.from_env()
    app = FastAPI(title="Blitz Live Consensus Overlay")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        session = OverlaySession(
            gate_threshold=config.gate,
            baseline_seconds=config.baseline_seconds,
            log_dir=config.log_dir,
        )
        try:
            while True:
                raw = await websocket.receive_json()
                consensus = session.process(raw)
                if session.should_emit(consensus.ts):
                    await websocket.send_json(consensus.to_dict())
        except WebSocketDisconnect:
            return

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app
