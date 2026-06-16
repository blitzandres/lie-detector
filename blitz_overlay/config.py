"""Typed config loaded from environment / .env (READINESS #16). No secrets needed in Stage 1."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


@dataclass
class OverlayConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    gate: float = 0.65
    baseline_seconds: int = 90
    open_browser: bool = True
    log_dir: str = "logs"

    @classmethod
    def from_env(cls) -> OverlayConfig:
        _load_dotenv()
        return cls(
            host=os.environ.get("BLITZ_OVERLAY_HOST", "127.0.0.1"),
            port=int(os.environ.get("BLITZ_OVERLAY_PORT", "8000")),
            gate=float(os.environ.get("BLITZ_OVERLAY_GATE", "0.65")),
            baseline_seconds=int(os.environ.get("BLITZ_OVERLAY_BASELINE_SECONDS", "90")),
            open_browser=os.environ.get("BLITZ_OVERLAY_OPEN_BROWSER", "1") == "1",
            log_dir=os.environ.get("BLITZ_OVERLAY_LOG_DIR", "logs"),
        )
