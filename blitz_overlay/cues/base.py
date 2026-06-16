"""Abstract cue-detector interface (the per-cue plugin contract for the live overlay).

Each detector is stateful and lives for one session. It reads a FeatureFrame, derives a
single scalar measurement, normalizes it against the rolling baseline, and emits a
CueEvent when the deviation passes the cue's direction/threshold. Metadata (family,
region, effect size, tier) comes from the science-driven weights config.

The detector NEVER mutates the baseline — the session owns baseline updates once per frame.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from blitz_overlay.schemas import FeatureFrame
from blitz_overlay.weights import weight_for
from core.calibration import RollingBaseline
from core.schemas.cue_event import CueEvent, Modality, Phase

_MODALITY: dict[str, Modality] = {
    "visual": Modality.VISUAL,
    "physio": Modality.PHYSIOLOGICAL,
    "audio": Modality.AUDIO,
    "linguistic": Modality.LINGUISTIC,
}

# A cue fires when its directed robust-Z meets this threshold.
DEFAULT_Z_THRESHOLD = 2.0


class CueDetector(ABC):
    cue_id: str = ""
    z_threshold: float = DEFAULT_Z_THRESHOLD
    direction: int = 1  # +1: high values suspicious; -1: low values suspicious

    def __init__(self) -> None:
        spec = weight_for(self.cue_id)
        self.family: str = spec["family"]
        self.region: str = spec["region"]
        self.effect_size_d: float = spec["effect_size_d"]
        self.reliability_tier: int = spec["reliability_tier"]
        self.modality: Modality = _MODALITY[self.family]

    @abstractmethod
    def measure(self, frame: FeatureFrame) -> float | None:
        """Return this cue's scalar measurement for the frame, or None if unavailable."""

    def quality(self, frame: FeatureFrame) -> float:
        """Extraction confidence — scales with landmark confidence (spec §8 low-light path).

        Public and overridable: e.g. the rPPG detector scales this by buffer fill.
        """
        return max(0.0, min(1.0, frame.confidence))

    def update(self, frame: FeatureFrame, baseline: RollingBaseline,
               value: float | None = None) -> CueEvent | None:
        """Read frame, normalize against baseline (read-only), emit CueEvent or None."""
        if value is None:
            value = self.measure(frame)
        if value is None:
            return None
        z = baseline.normalize(self.cue_id, value)
        directed_z = z * self.direction
        if directed_z < self.z_threshold:
            return None
        return CueEvent(
            cue_id=self.cue_id,
            modality=self.modality,
            timestamp_ms=frame.ts,
            phase=Phase.RESPONSE,
            raw_value=float(value),
            z_score=directed_z,
            llr=0.0,
            quality=self.quality(frame),
            question_id="live",
            region=self.region,
            effect_size_d=self.effect_size_d,
            reliability_tier=self.reliability_tier,
        )
