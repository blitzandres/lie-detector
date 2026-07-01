"""FastAPI app: serves the browser overlay (static) and the /ws feature-frame endpoint.

One process = engine + browser host, so the whole thing starts with one command.
Raw video never reaches here — only feature frames (spec §3).
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from blitz_overlay.config import OverlayConfig
from blitz_overlay.content.ollama_judge import OllamaContentJudge
from blitz_overlay.pipeline import OverlaySession

WEB_DIR = Path(__file__).resolve().parents[1] / "apps" / "overlay-web"


def _make_content_judge():
    """Real Ollama judge only when BLITZ_OVERLAY_CONTENT=ollama; otherwise None → the session's
    deterministic StubContentJudge (so the test suite never hits a live LLM)."""
    if os.getenv("BLITZ_OVERLAY_CONTENT", "").lower() == "ollama":
        return OllamaContentJudge(model=os.getenv("BLITZ_OVERLAY_OLLAMA_MODEL", "llama3.2:3b"))
    return None


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
            content_judge=_make_content_judge(),
        )
        try:
            while True:
                raw = await websocket.receive_json()
                if raw.get("type") == "turn":
                    result = await asyncio.to_thread(
                        session.judge_turn,
                        raw.get("question", ""), raw.get("answer", ""),
                        int(raw.get("t0", 0)), int(raw.get("t1", 0)),
                    )
                    result["type"] = "turn_result"
                    await websocket.send_json(result)
                    continue
                consensus = session.process(raw)
                if session.should_emit(consensus.ts):
                    await websocket.send_json(consensus.to_dict())
        except WebSocketDisconnect:
            return

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app
