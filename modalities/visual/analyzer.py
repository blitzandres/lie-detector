"""Offline research visual analyzer (spec 2c) — recorded video only.

Mirrors modalities/audio/analyzer.py: clip-level scalar features → CueEvents,
normalized against the shared PersonalBaseline and fused by the existing core.
Backends are swappable and lazy; tests always use stubs.
"""
from __future__ import annotations

from core.schemas.cue_event import CueEvent, Modality, Phase
from modalities.visual.backends import AUBackend, PyFeatBackend, VisualFrame
from modalities.visual.flow import FlowSource

MIN_FRAMES = 10          # fewer usable face frames than this → honest abstain
SMILE_FLOOR = 0.3        # AU12 below this = no smile to authenticate

CUE_SPECS: dict[str, dict] = {
    "visual.au_stress_brow": {
        "effect_size_d": 0.30, "reliability_tier": 3,
        "citation": "FACS AU1+AU2+AU4 combination — fear/stress brow (catalog cue 9 family).",
    },
    "visual.au_lip_press": {
        "effect_size_d": 0.30, "reliability_tier": 3,
        "citation": "AU24/AU23 lip press/tighten — withholding (catalog cue 3).",
    },
    "visual.au_contempt": {
        "effect_size_d": 0.30, "reliability_tier": 2,
        "citation": "AU14 unilateral contempt (catalog cue 5/12 family).",
    },
    "visual.duchenne_deficit": {
        "effect_size_d": 0.35, "reliability_tier": 3,
        "citation": "Ekman Duchenne marker — AU12 without AU6 = masked smile.",
    },
    "visual.emotion_leakage": {
        "effect_size_d": 0.30, "reliability_tier": 3,
        "citation": "Negative-affect leakage during neutral accounts (Py-Feat emotion head).",
    },
    "visual.head_dynamics": {
        "effect_size_d": 0.25, "reliability_tier": 3,
        "citation": "Head movement dynamics (catalog cue 14 family) — person-relative.",
    },
    "visual.expressivity_rigidity": {
        "effect_size_d": 0.35, "reliability_tier": 3,
        "citation": "Decreased expressivity/illustrators under load (DePaulo 2003 family); "
                    "raw value is NEGATED AU std so higher = more rigid.",
    },
    "visual.au_micro_burst": {
        "effect_size_d": 0.20, "reliability_tier": 4,
        "citation": "Micro-expression onset proxy — HONEST: low base rates, modest effect "
                    "sizes (Porter & ten Brinke 2008); weighted low by design.",
    },
    "visual.flow_agitation": {
        "effect_size_d": 0.20, "reliability_tier": 4,
        "citation": "Dense optical-flow peak — gross motion agitation; weak, exploratory.",
    },
}


class VisualAnalyzer:
    """Recorded-video visual modality plugin (research tier)."""

    def __init__(self, backend: AUBackend | None = None, flow: FlowSource | None = None):
        self.backend = backend or PyFeatBackend()
        self.flow = flow                      # optional second pass (sequential, cheap)
        self.cue_specs = CUE_SPECS

    # ── feature extraction ────────────────────────────────────────────────────

    def extract_features(self, video_path: str) -> dict[str, float]:
        frames = [f for f in self.backend.extract(video_path) if f.face_present]
        if len(frames) < MIN_FRAMES:
            raise ValueError(
                f"input_quality_insufficient: {len(frames)} usable face frames "
                f"(< {MIN_FRAMES}) in {video_path}"
            )
        feats = {
            "visual.au_stress_brow": self._mean(frames, self._stress_brow),
            "visual.au_lip_press": self._mean(frames, self._lip_press),
            "visual.au_contempt": self._mean(frames, lambda f: f.aus.get("AU14", 0.0)),
            "visual.duchenne_deficit": self._duchenne_deficit(frames),
            "visual.emotion_leakage": self._mean(frames, self._negative_affect),
            "visual.head_dynamics": self._head_dynamics(frames),
            "visual.expressivity_rigidity": self._rigidity(frames),
            "visual.au_micro_burst": self._micro_burst(frames),
        }
        if self.flow is not None:
            samples = self.flow.extract(video_path)
            if samples:
                feats["visual.flow_agitation"] = max(s.peak_magnitude for s in samples)
        return feats

    def build_baseline_observations(self, video_paths: list[str]) -> dict[str, list[float]]:
        observations: dict[str, list[float]] = {}
        for path in video_paths:
            for cue_id, value in self.extract_features(path).items():
                observations.setdefault(cue_id, []).append(value)
        return observations

    def analyze(self, video_path: str, question_id: str, baseline,
                timestamp_ms: int = 0) -> list[CueEvent]:
        features = self.extract_features(video_path)
        cues: list[CueEvent] = []
        for cue_id, raw_value in features.items():
            spec = self.cue_specs[cue_id]
            cues.append(CueEvent(
                cue_id=cue_id,
                modality=Modality.VISUAL,
                timestamp_ms=timestamp_ms,
                phase=Phase.RESPONSE,
                raw_value=raw_value,
                z_score=baseline.normalize(cue_id, raw_value),
                llr=0.0,
                quality=0.9,
                question_id=question_id,
                effect_size_d=spec["effect_size_d"],
                reliability_tier=spec["reliability_tier"],
            ))
        return cues

    # ── per-cue math ──────────────────────────────────────────────────────────

    @staticmethod
    def _mean(frames: list[VisualFrame], fn) -> float:
        vals = [fn(f) for f in frames]
        return sum(vals) / len(vals)

    @staticmethod
    def _stress_brow(f: VisualFrame) -> float:
        return min(f.aus.get("AU01", 0.0), f.aus.get("AU02", 0.0), f.aus.get("AU04", 0.0))

    @staticmethod
    def _lip_press(f: VisualFrame) -> float:
        return f.aus.get("AU24", f.aus.get("AU23", 0.0))

    @staticmethod
    def _negative_affect(f: VisualFrame) -> float:
        return max(f.emotions.get(k, 0.0) for k in ("fear", "anger", "disgust", "sadness"))

    @staticmethod
    def _duchenne_deficit(frames: list[VisualFrame]) -> float:
        vals = []
        for f in frames:
            au12 = f.aus.get("AU12", 0.0)
            if au12 >= SMILE_FLOOR:
                vals.append(au12 * max(0.0, au12 - f.aus.get("AU06", 0.0)))
        return sum(vals) / len(vals) if vals else 0.0

    @staticmethod
    def _head_dynamics(frames: list[VisualFrame]) -> float:
        steps = []
        for a, b in zip(frames, frames[1:], strict=False):
            dt = max(1, b.ts_ms - a.ts_ms) / 1000.0
            dist = sum((b.head_pose.get(k, 0.0) - a.head_pose.get(k, 0.0)) ** 2
                       for k in ("yaw", "pitch", "roll")) ** 0.5
            steps.append(dist / dt)
        return sum(steps) / len(steps) if steps else 0.0

    @staticmethod
    def _rigidity(frames: list[VisualFrame]) -> float:
        keys: set[str] = set()
        for f in frames:
            keys.update(f.aus)
        stds = []
        for k in keys:
            vals = [f.aus.get(k, 0.0) for f in frames]
            mean = sum(vals) / len(vals)
            stds.append((sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5)
        # NEGATED so that "more rigid" = higher raw value = positive fusion direction
        return -(sum(stds) / len(stds)) if stds else 0.0

    @staticmethod
    def _micro_burst(frames: list[VisualFrame]) -> float:
        """Peak frame-to-frame mean AU delta, max across early/mid/late thirds."""
        third = max(2, len(frames) // 3)
        peaks = []
        for seg_start in range(0, len(frames), third):
            seg = frames[seg_start:seg_start + third]
            peak = 0.0
            for a, b in zip(seg, seg[1:], strict=False):
                keys = set(a.aus) | set(b.aus)
                if not keys:
                    continue
                delta = sum(abs(b.aus.get(k, 0.0) - a.aus.get(k, 0.0)) for k in keys) / len(keys)
                peak = max(peak, delta)
            peaks.append(peak)
        return max(peaks) if peaks else 0.0
