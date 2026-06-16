"""Rolling per-person baseline (calibration mode="rolling") — READINESS #7.

Builds an in-session baseline from a sliding time window of recent feature values
instead of a fixed enrollment clip, then scores incoming values as robust-Z (median/MAD)
deviations. Used by the live overlay where there is no enrollment step. No flags are
permitted until the fill window (default 90s, spec §8) has elapsed (status CALIBRATING).
"""
from __future__ import annotations

from collections import defaultdict, deque

from core.calibration.baseline import compute_robust_z

DEFAULT_BASELINE_SECONDS = 90
DEFAULT_WINDOW_SECONDS = 180


class RollingBaseline:
    mode = "rolling"

    def __init__(
        self,
        baseline_seconds: int = DEFAULT_BASELINE_SECONDS,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
    ):
        self.baseline_seconds = baseline_seconds
        self.window_seconds = window_seconds
        self._values: dict[str, deque] = defaultdict(deque)  # cue_id -> deque[(ts_ms, value)]
        self._first_ts: int | None = None
        self._last_ts: int = 0

    def update(self, features: dict[str, float], ts_ms: int) -> None:
        if self._first_ts is None:
            self._first_ts = ts_ms
        self._last_ts = ts_ms
        cutoff = ts_ms - self.window_seconds * 1000
        for cue_id, value in features.items():
            dq = self._values[cue_id]
            dq.append((ts_ms, float(value)))
            while dq and dq[0][0] < cutoff:
                dq.popleft()

    @property
    def elapsed_seconds(self) -> float:
        if self._first_ts is None:
            return 0.0
        return (self._last_ts - self._first_ts) / 1000.0

    @property
    def is_calibrating(self) -> bool:
        return self.elapsed_seconds < self.baseline_seconds

    @property
    def ready(self) -> bool:
        return not self.is_calibrating

    def observation_count(self, cue_id: str) -> int:
        return len(self._values.get(cue_id, ()))

    def normalize(self, cue_id: str, raw_value: float) -> float:
        """Robust-Z vs the rolling window. Returns 0.0 while calibrating or if underpowered.

        When the baseline has zero variance (MAD == 0), falls back to unit-spread so that
        genuine deviations from a flat baseline still produce a non-zero z-score, consistent
        with PersonalBaseline.normalize behaviour.
        """
        if self.is_calibrating:
            return 0.0
        dq = self._values.get(cue_id)
        if not dq or len(dq) < 5:
            return 0.0
        return compute_robust_z(raw_value, [v for _, v in dq], flat_fallback=True)
