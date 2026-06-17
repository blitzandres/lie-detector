"""BellController — the earned, sustained, synchrony-gated alarm.

Reads (read-only) the SynchronyDetector burst + the fused posterior. Rings only when a
burst AND posterior >= risk_floor hold continuously for hold_ms (debounced). Honest label,
never "lie". The operating point (k / lit_z / risk_floor) is moved by the sensitivity slider
via map_sensitivity — never the science cue weights.
"""
from __future__ import annotations

BELL_LABEL = "strong deception-pattern convergence"


class BellController:
    def __init__(self, hold_ms: int = 1500, risk_floor: float = 0.65):
        self.hold_ms = hold_ms
        self.risk_floor = risk_floor
        self._since_ts: int | None = None  # when the current satisfied streak began
        self._ringing = False

    def set_params(self, *, hold_ms: int | None = None, risk_floor: float | None = None) -> None:
        if hold_ms is not None:
            self.hold_ms = hold_ms
        if risk_floor is not None:
            self.risk_floor = risk_floor

    def update(self, ts: int, convergence: dict, posterior: float) -> dict:
        condition = bool(convergence.get("burst")) and posterior >= self.risk_floor
        just_rang = False
        if condition:
            if self._since_ts is None:
                self._since_ts = ts
            if ts - self._since_ts >= self.hold_ms and not self._ringing:
                self._ringing = True
                just_rang = True
        else:
            self._since_ts = None
            self._ringing = False

        sustained_ms = (ts - self._since_ts) if self._since_ts is not None else 0
        record = None
        if just_rang:
            record = {
                "ts": ts,
                "cue_ids": list(convergence.get("lit_cue_ids", [])),
                "families": list(convergence.get("families_lit", [])),
                "risk": round(posterior, 4),
            }
        return {
            "ringing": self._ringing,
            "just_rang": just_rang,
            "sustained_ms": sustained_ms,
            "label": BELL_LABEL,
            "record": record,
        }


def map_sensitivity(sensitivity: float) -> dict:
    """Map a 0..1 slider to the bell operating point. 0 = conservative, 1 = max (more alarms).

    Never touches science weights or the >=2-families requirement.
    """
    s = max(0.0, min(1.0, sensitivity))
    return {
        "k": int(round(3 - s)),               # 3 -> 2
        "lit_z": round(2.0 - 0.5 * s, 3),     # 2.0 -> 1.5
        "risk_floor": round(0.65 - 0.20 * s, 3),  # 0.65 -> 0.45
    }
