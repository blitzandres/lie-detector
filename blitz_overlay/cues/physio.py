"""rPPG heart-rate cue — the physiological family voter (spec §11)."""
from __future__ import annotations

from collections import deque

from blitz_overlay.cues.base import CueDetector
from blitz_overlay.rppg import estimate_bpm
from blitz_overlay.schemas import FeatureFrame

WINDOW_SECONDS = 10
ESTIMATE_EVERY_MS = 1000  # recompute BPM at most once per second


class RppgHeartRate(CueDetector):
    cue_id = "physio.heart_rate"
    direction = 1  # elevated HR is the suspicious direction (autonomic arousal)

    def __init__(self, fps: float = 30.0) -> None:
        super().__init__()
        self.fps = fps
        self._buf: deque[tuple[int, list[float]]] = deque()  # (ts, [r,g,b]) forehead ROI
        self._last_estimate_ts = -ESTIMATE_EVERY_MS
        self._last_bpm: float | None = None
        self._last_appended_ts: int = -1  # deduplicates calls at the same timestamp

    def measure(self, frame: FeatureFrame) -> float | None:
        if not frame.rppg or "forehead_rgb" not in frame.rppg:
            return None
        now = frame.ts
        if now != self._last_appended_ts:
            self._buf.append((now, [float(c) for c in frame.rppg["forehead_rgb"]]))
            self._last_appended_ts = now
        cutoff = now - WINDOW_SECONDS * 1000
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()
        if now - self._last_estimate_ts < ESTIMATE_EVERY_MS:
            return self._last_bpm
        self._last_estimate_ts = now
        samples = [rgb for _, rgb in self._buf]
        self._last_bpm = estimate_bpm(samples, fps=self.fps)
        return self._last_bpm

    def quality(self, frame: FeatureFrame) -> float:
        base = super().quality(frame)
        fill = min(1.0, len(self._buf) / (WINDOW_SECONDS * self.fps))
        # Skin-aware: down-weight when little real skin was sampled in the ROI (hair/glasses/
        # shadow). skin_fraction comes from the browser's per-pixel skin mask; absent -> 1.0.
        skin = 1.0
        if frame.rppg:
            skin = max(0.0, min(1.0, float(frame.rppg.get("skin_fraction", 1.0))))
        return base * fill * skin
