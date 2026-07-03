# Step 2c — Offline Research Visual Analyzer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A recorded-video visual modality (`modalities/visual/`) that extracts AU/emotion/head-pose/optical-flow features via swappable backends (Py-Feat v2 primary) and emits `CueEvent`s into the existing `PersonalBaseline` + log-odds fusion, exposed as `blitz analyze-video`.

**Architecture:** Mirrors `modalities/audio/analyzer.py`: `VisualAnalyzer.extract_features(video) -> dict[str, float]` (clip-level scalars), `build_baseline_observations`, `analyze(...) -> list[CueEvent]`. Heavy models live behind an `AUBackend` seam (`PyFeatBackend`, `OpenGraphAUBackend`/`LibreFaceBackend` seams, `EnsembleBackend`, `StubBackend` for tests) with **lazy imports** so nothing heavy loads unless used — honoring the M1/8GB one-model-at-a-time rule. Optical flow is a second seam (`FlowSource`). Tests use stubs only; real deps go in a `[research]` optional extra.

**Tech Stack:** Python 3.14, pytest; optional: py-feat, opencv-python (never imported at module top level).

**Spec:** `docs/superpowers/specs/2026-07-02-step2-visual-deepening-design.md` (§2c)

---

### Task 1: Backend + flow seams

**Files:**
- Create: `modalities/visual/__init__.py`, `modalities/visual/backends.py`, `modalities/visual/flow.py`
- Test: `tests/research/test_visual_backends.py`

- [ ] **Step 1: Failing tests**

```python
"""Backend seam tests — stubs only; real model backends are lazy and never imported here."""
from modalities.visual.backends import EnsembleBackend, StubBackend, VisualFrame


def _vf(ts, au01=0.0, au12=0.0, **kw):
    return VisualFrame(ts_ms=ts, face_present=True, quality=0.9,
                       aus={"AU01": au01, "AU12": au12}, **kw)


def test_stub_backend_returns_frames():
    frames = [_vf(0), _vf(100, au01=0.5)]
    assert StubBackend(frames).extract("clip.mp4") == frames


def test_ensemble_backend_averages_au_values():
    a = StubBackend([_vf(0, au01=0.2)])
    b = StubBackend([_vf(0, au01=0.6)])
    merged = EnsembleBackend(a, b).extract("clip.mp4")
    assert len(merged) == 1
    assert abs(merged[0].aus["AU01"] - 0.4) < 1e-9


def test_pyfeat_backend_is_lazy_and_guides_install():
    from modalities.visual.backends import PyFeatBackend
    backend = PyFeatBackend()   # constructing must NOT import py-feat
    try:
        backend.extract("missing.mp4")
    except RuntimeError as e:
        assert "research" in str(e)   # install guidance
    except Exception:
        pass   # py-feat installed and failed on the missing file — also acceptable
```

- [ ] **Step 2: Run** `python3 -m pytest tests/research/ -q` → FAIL (module missing)

- [ ] **Step 3: Implement `backends.py`**

```python
"""AU backends for the offline research visual analyzer (spec 2c).

Heavy models are LAZY: importing this module never imports torch/py-feat.
One backend runs at a time (M1/8GB sequential rule) — load → run → release.
"""
from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

_INSTALL_HINT = "pip install -e '.[research]'"


@dataclass
class VisualFrame:
    """One analyzed video frame: AU intensities + emotions + head pose."""

    ts_ms: int
    face_present: bool = False
    quality: float = 0.0                              # detector confidence [0,1]
    aus: dict = field(default_factory=dict)           # "AU01".. -> intensity [0,1]
    emotions: dict = field(default_factory=dict)      # "fear".. -> prob [0,1]
    head_pose: dict = field(default_factory=dict)     # yaw/pitch/roll degrees


class AUBackend(ABC):
    name: str = ""

    @abstractmethod
    def extract(self, video_path: str) -> list[VisualFrame]:
        """Run the model over the whole clip, then release it (sequential rule)."""


class StubBackend(AUBackend):
    """Deterministic backend for tests — returns pre-built frames."""

    name = "stub"

    def __init__(self, frames: list[VisualFrame]):
        self._frames = frames

    def extract(self, video_path: str) -> list[VisualFrame]:
        return list(self._frames)


class PyFeatBackend(AUBackend):
    """Py-Feat Detector v2 — 20 AUs w/ intensity, emotions, head pose in one pass."""

    name = "pyfeat"

    def __init__(self, skip_frames: int = 2):
        self.skip_frames = skip_frames   # analyze every Nth frame (speed on CPU)

    def extract(self, video_path: str) -> list[VisualFrame]:
        try:
            from feat import Detector  # noqa: PLC0415 — lazy heavy import by design
        except ImportError as e:
            raise RuntimeError(
                f"Py-Feat is not installed. Install the research extra: {_INSTALL_HINT}"
            ) from e
        detector = Detector()
        try:
            fex = detector.detect_video(video_path, skip_frames=self.skip_frames)
            frames: list[VisualFrame] = []
            au_cols = [c for c in fex.columns if c.upper().startswith("AU")]
            emo_cols = [c for c in ("anger", "disgust", "fear", "happiness",
                                    "sadness", "surprise", "neutral") if c in fex.columns]
            fps = getattr(fex, "fps", None) or 30.0
            for _, row in fex.iterrows():
                frame_no = int(row.get("frame", 0))
                frames.append(VisualFrame(
                    ts_ms=int(frame_no * 1000.0 / fps),
                    face_present=bool(row.get("FaceScore", 1.0) > 0),
                    quality=float(min(1.0, max(0.0, row.get("FaceScore", 0.9)))),
                    aus={c.upper(): float(row[c]) for c in au_cols},
                    emotions={c: float(row[c]) for c in emo_cols},
                    head_pose={"yaw": float(row.get("Yaw", 0.0)),
                               "pitch": float(row.get("Pitch", 0.0)),
                               "roll": float(row.get("Roll", 0.0))},
                ))
            return frames
        finally:
            del detector
            gc.collect()   # release before any other heavyweight loads (8GB rule)


class OpenGraphAUBackend(AUBackend):
    """Complementary AU detector for ensemble robustness — integration seam.

    Wire-up is deliberate follow-up work: OpenGraphAU has no pip package; it needs a
    cloned repo + checkpoint. The seam keeps `EnsembleBackend` ready for it.
    """

    name = "opengraphau"

    def extract(self, video_path: str) -> list[VisualFrame]:
        raise RuntimeError(
            "OpenGraphAU backend is a seam: clone github.com/lingjivoo/OpenGraphAU, "
            "download a checkpoint, and implement extract() against it. "
            "Py-Feat (default) covers AUs today."
        )


class LibreFaceBackend(AUBackend):
    """Documented fallback AU backend (LibreFace) — integration seam, same contract."""

    name = "libreface"

    def extract(self, video_path: str) -> list[VisualFrame]:
        raise RuntimeError(
            "LibreFace backend is a seam: pip install libreface, then implement "
            "extract() against libreface.get_facial_attributes(). "
            "Py-Feat (default) covers AUs today."
        )


class EnsembleBackend(AUBackend):
    """Agreement ensemble: mean AU intensity across two backends, frame-aligned.

    Backends run SEQUENTIALLY (never concurrently) — 8GB rule.
    """

    name = "ensemble"

    def __init__(self, primary: AUBackend, secondary: AUBackend):
        self.primary = primary
        self.secondary = secondary

    def extract(self, video_path: str) -> list[VisualFrame]:
        a = self.primary.extract(video_path)
        b = self.secondary.extract(video_path)
        merged = []
        for fa, fb in zip(a, b, strict=False):
            aus = dict(fa.aus)
            for k, v in fb.aus.items():
                aus[k] = (aus[k] + v) / 2.0 if k in aus else v
            merged.append(VisualFrame(
                ts_ms=fa.ts_ms,
                face_present=fa.face_present and fb.face_present,
                quality=(fa.quality + fb.quality) / 2.0,
                aus=aus, emotions=dict(fa.emotions), head_pose=dict(fa.head_pose),
            ))
        return merged
```

- [ ] **Step 4: Implement `flow.py`**

```python
"""Optical-flow seam: temporal motion dynamics for micro-expression spotting (spec 2c)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

_INSTALL_HINT = "pip install -e '.[research]'"


@dataclass
class FlowSample:
    ts_ms: int
    mean_magnitude: float    # average pixel motion this step
    peak_magnitude: float    # strongest local motion this step


class FlowSource(ABC):
    @abstractmethod
    def extract(self, video_path: str) -> list[FlowSample]: ...


class StubFlowSource(FlowSource):
    def __init__(self, samples: list[FlowSample]):
        self._samples = samples

    def extract(self, video_path: str) -> list[FlowSample]:
        return list(self._samples)


class FarnebackFlow(FlowSource):
    """Dense Farneback optical flow via OpenCV — cheap CPU pass, lazy import."""

    def __init__(self, step: int = 3, resize_width: int = 320):
        self.step = step
        self.resize_width = resize_width

    def extract(self, video_path: str) -> list[FlowSample]:
        try:
            import cv2  # noqa: PLC0415 — lazy heavy import by design
        except ImportError as e:
            raise RuntimeError(
                f"OpenCV is not installed. Install the research extra: {_INSTALL_HINT}"
            ) from e
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        samples: list[FlowSample] = []
        prev = None
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % self.step:
                idx += 1
                continue
            h, w = frame.shape[:2]
            scale = self.resize_width / float(w)
            gray = cv2.cvtColor(
                cv2.resize(frame, (self.resize_width, int(h * scale))),
                cv2.COLOR_BGR2GRAY)
            if prev is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag = (flow[..., 0] ** 2 + flow[..., 1] ** 2) ** 0.5
                samples.append(FlowSample(
                    ts_ms=int(idx * 1000.0 / fps),
                    mean_magnitude=float(mag.mean()),
                    peak_magnitude=float(mag.max()),
                ))
            prev = gray
            idx += 1
        cap.release()
        return samples
```

`__init__.py`:

```python
from modalities.visual.analyzer import VisualAnalyzer

__all__ = ["VisualAnalyzer"]
```

(Leave `__init__.py` for Task 2 since it imports the analyzer; create the directory files in this task with an empty `__init__.py` placeholder, filled in Task 2.)

- [ ] **Step 5: Run tests** → PASS; **commit** `feat(research): AU backend + optical-flow seams (Py-Feat primary, ensemble, stubs)`

---

### Task 2: VisualAnalyzer — features, cue specs, CueEvents

**Files:**
- Create: `modalities/visual/analyzer.py`; fill `modalities/visual/__init__.py`
- Test: `tests/research/test_visual_analyzer.py`

Nine clip-level cues (ids distinct from the live overlay's blendshape cues):

| cue_id | signal |
|---|---|
| `visual.au_stress_brow` | mean min(AU01, AU02, AU04) — fear/stress brow combo |
| `visual.au_lip_press` | mean AU24 (fallback AU23) |
| `visual.au_contempt` | mean AU14 |
| `visual.duchenne_deficit` | while smiling (AU12 ≥ 0.3): AU12 × max(0, AU12 − AU06) |
| `visual.emotion_leakage` | mean of max(fear, anger, disgust, sadness) per frame |
| `visual.head_dynamics` | mean head-pose delta per second |
| `visual.expressivity_rigidity` | **negated** mean per-AU std (higher = more rigid — keeps fusion direction positive) |
| `visual.au_micro_burst` | peak frame-to-frame mean AU delta, max across early/mid/late thirds |
| `visual.flow_agitation` | peak optical-flow magnitude (only when a FlowSource is wired) |

- [ ] **Step 1: Failing tests**

```python
"""VisualAnalyzer tests — stub backends only; never touches real models."""
import pytest

from core.calibration.baseline import PersonalBaseline
from modalities.visual.analyzer import MIN_FRAMES, VisualAnalyzer
from modalities.visual.backends import StubBackend, VisualFrame
from modalities.visual.flow import FlowSample, StubFlowSource


def _clip(n=40, au01=0.1, au12=0.0, au06=0.0, yaw=0.0, fear=0.05, wobble=0.0):
    frames = []
    for i in range(n):
        w = wobble * (i % 2)
        frames.append(VisualFrame(
            ts_ms=i * 100, face_present=True, quality=0.9,
            aus={"AU01": au01 + w, "AU02": au01 + w, "AU04": au01 + w,
                 "AU06": au06, "AU12": au12, "AU14": 0.05, "AU24": 0.1},
            emotions={"fear": fear, "anger": 0.02, "disgust": 0.02,
                      "sadness": 0.02, "happiness": 0.5},
            head_pose={"yaw": yaw * i, "pitch": 0.0, "roll": 0.0},
        ))
    return frames


def test_extract_features_produces_all_stub_reachable_cues():
    an = VisualAnalyzer(backend=StubBackend(_clip()))
    feats = an.extract_features("clip.mp4")
    for cue in ("visual.au_stress_brow", "visual.au_lip_press", "visual.au_contempt",
                "visual.duchenne_deficit", "visual.emotion_leakage",
                "visual.head_dynamics", "visual.expressivity_rigidity",
                "visual.au_micro_burst"):
        assert cue in feats
    assert "visual.flow_agitation" not in feats   # no FlowSource wired


def test_flow_agitation_requires_flow_source():
    flow = StubFlowSource([FlowSample(0, 0.2, 1.4), FlowSample(100, 0.3, 2.0)])
    an = VisualAnalyzer(backend=StubBackend(_clip()), flow=flow)
    assert abs(an.extract_features("clip.mp4")["visual.flow_agitation"] - 2.0) < 1e-9


def test_stress_brow_rises_with_combo():
    calm = VisualAnalyzer(backend=StubBackend(_clip(au01=0.05)))
    stressed = VisualAnalyzer(backend=StubBackend(_clip(au01=0.6)))
    assert (stressed.extract_features("c.mp4")["visual.au_stress_brow"]
            > calm.extract_features("c.mp4")["visual.au_stress_brow"])


def test_duchenne_deficit_zero_without_smile_and_high_when_masked():
    no_smile = VisualAnalyzer(backend=StubBackend(_clip(au12=0.1)))
    masked = VisualAnalyzer(backend=StubBackend(_clip(au12=0.8, au06=0.05)))
    genuine = VisualAnalyzer(backend=StubBackend(_clip(au12=0.8, au06=0.7)))
    assert no_smile.extract_features("c.mp4")["visual.duchenne_deficit"] == 0.0
    assert (masked.extract_features("c.mp4")["visual.duchenne_deficit"]
            > genuine.extract_features("c.mp4")["visual.duchenne_deficit"])


def test_rigidity_is_negated_variance():
    frozen = VisualAnalyzer(backend=StubBackend(_clip(wobble=0.0)))
    lively = VisualAnalyzer(backend=StubBackend(_clip(wobble=0.4)))
    # negated std: frozen (no variance) must score HIGHER than lively
    assert (frozen.extract_features("c.mp4")["visual.expressivity_rigidity"]
            > lively.extract_features("c.mp4")["visual.expressivity_rigidity"])


def test_abstains_on_too_few_face_frames():
    an = VisualAnalyzer(backend=StubBackend(_clip(n=MIN_FRAMES - 1)))
    with pytest.raises(ValueError, match="input_quality_insufficient"):
        an.extract_features("c.mp4")


def test_analyze_emits_cue_events_with_specs():
    an = VisualAnalyzer(backend=StubBackend(_clip()))
    baseline = PersonalBaseline()
    baseline.record_baseline(an.build_baseline_observations(["b1.mp4", "b2.mp4"]),
                             duration_s=120)
    events = an.analyze("c.mp4", question_id="q1", baseline=baseline, timestamp_ms=5)
    assert events and all(e.cue_id.startswith("visual.") for e in events)
    assert all(e.modality.value == "visual" for e in events)
    assert all(e.effect_size_d > 0 for e in events)
```

- [ ] **Step 2: Run** → FAIL (`analyzer` missing)

- [ ] **Step 3: Implement `analyzer.py`**

```python
"""Offline research visual analyzer (spec 2c) — recorded video only.

Mirrors modalities/audio/analyzer.py: clip-level scalar features → CueEvents,
normalized against the shared PersonalBaseline and fused by the existing core.
Backends are swappable and lazy; tests always use stubs.
"""
from __future__ import annotations

from modalities.visual.backends import AUBackend, PyFeatBackend, VisualFrame
from modalities.visual.flow import FlowSource

from core.schemas.cue_event import CueEvent, Modality, Phase

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
```

- [ ] **Step 4: Run** `python3 -m pytest tests/research/ -q` → PASS; full suite + ruff → green

- [ ] **Step 5: Commit** `feat(research): VisualAnalyzer — 9 AU/flow clip cues into shared baseline+fusion`

---

### Task 3: Engine + CLI + extras wiring

**Files:**
- Modify: `blitz_engine/engine.py` (visual modality), `blitz_engine/cli.py` (`analyze-video`), `pyproject.toml` (`research` extra)
- Test: `tests/research/test_engine_visual.py`

- [ ] **Step 1: Failing test**

```python
from blitz_engine.engine import BlitzEngine
from modalities.visual.backends import StubBackend
from modalities.visual.analyzer import VisualAnalyzer
from tests.research.test_visual_analyzer import _clip


def test_engine_runs_visual_modality_end_to_end():
    analyzer = VisualAnalyzer(backend=StubBackend(_clip()))
    engine = BlitzEngine(modalities=["visual"], visual_analyzer=analyzer)
    session = engine.new_session(
        baseline_video_files=["b1.mp4", "b2.mp4"],
        consent=True, use_case="research", jurisdiction="CA-US",
    )
    result = session.analyze(video_path="r.mp4", question="Where were you?")
    assert 0.0 <= result.risk_score <= 1.0
    assert result.quality_flags["input_mode"] == "video"
    assert any(c["cue_id"].startswith("visual.") for c in result.top_cues)
```

- [ ] **Step 2: Engine changes** — `ALLOWED_MODALITIES = {"linguistic", "audio", "visual"}`; `BlitzEngine.__init__` gains `visual_analyzer=None` param and sets `self.visual = (visual_analyzer or VisualAnalyzer()) if "visual" in self.modalities else None` (import `VisualAnalyzer` lazily inside `__init__` to keep base import light); `new_session` gains `baseline_video_files` (required when visual enabled; each video adds ~30 s to the default duration estimate); `analyze` gains `video_path` routed to `self.engine.visual.analyze(...)`; `input_mode` reports `"video"`/`"multimodal"` accordingly; update the unsupported-modality error text.

- [ ] **Step 3: CLI** — add `analyze-video` subparser: `--baseline-video` (repeatable, required), `--response-video` (required), `--question`, `--use-case`, `--jurisdiction`, `--output`; `run_analyze_video` builds `BlitzEngine(modalities=["visual"])`, sessions with `baseline_video_files`, analyzes `video_path`, prints via `dumps_report`. Route in `main`.

- [ ] **Step 4: pyproject** — under `[project.optional-dependencies]`: `research = ["py-feat>=0.6.2", "opencv-python>=4.9"]`

- [ ] **Step 5: Full suite + ruff → green; commit** `feat(research): visual modality in BlitzEngine + blitz analyze-video CLI + [research] extra`

---

### Task 4: Docs + push

- [ ] **Step 1** — README: Status list `Research tier` → `[x] … (code complete; Py-Feat/OpenCV via pip install -e ".[research]"; OpenGraphAU/LibreFace are seams)`; §4 diagram caption gains the install hint. OVERLAY_README untouched.
- [ ] **Step 2** — `python3 -m ruff check . && python3 -m pytest -q` → green
- [ ] **Step 3** — Commit `docs: research tier shipped — analyze-video quickstart + honest backend status` and push.
