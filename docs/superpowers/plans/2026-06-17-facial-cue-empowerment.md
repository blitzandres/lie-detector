# Facial Cue Empowerment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Grow the Visual family from 5 → 11 cues using data MediaPipe already produces (iris landmarks + previously-unused blendshapes), per the 66-cue catalog (`docs/CUE_CATALOG.md`).

**Architecture:** The browser now forwards 6 extra blendshapes (`noseSneer*`, `mouthSmile*`, `cheekSquint*`) and a new `geometry.iris_ratio` (pupil-dilation proxy from iris landmarks 468–477) — already implemented in `apps/overlay-web/js/{schema.js,mediapipe-extractor.js}`. This plan adds 6 Python `CueDetector`s reading that data, their science weights, and tests. The Cue Mixer shows the new lanes automatically (it builds lanes from the cue stream).

**Tech Stack:** Python 3.14 (`python3`, no venv), pytest, ruff.

**Reference:** `docs/CUE_CATALOG.md`. Verify after each task: `python3 -m pytest -q` and `python3 -m ruff check .` stay green. Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## Task 1: Science weights for the 6 new visual cues

**Files:**
- Modify: `blitz_overlay/weights.py`
- Test: `tests/overlay/test_weights.py`

- [ ] **Step 1: Write the failing test** — append to `tests/overlay/test_weights.py`:

```python
def test_new_facial_cue_weights_present():
    from blitz_overlay.weights import CUE_WEIGHTS
    expected = {
        "visual.gaze_fixation": (0.50, 2, "eyes"),
        "visual.pupil_dilation": (0.40, 2, "eyes"),
        "visual.asymmetric_smile": (0.35, 2, "mouth"),
        "visual.nose_wrinkle": (0.28, 3, "mouth"),
        "visual.eye_blocking": (0.28, 3, "eyes"),
        "visual.eye_widen": (0.25, 3, "eyes"),
    }
    for cue_id, (d, tier, region) in expected.items():
        spec = CUE_WEIGHTS[cue_id]
        assert spec["family"] == "visual"
        assert spec["region"] == region
        assert abs(spec["effect_size_d"] - d) < 1e-9
        assert spec["reliability_tier"] == tier
        assert spec["citation"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/overlay/test_weights.py::test_new_facial_cue_weights_present -v`
Expected: FAIL — `KeyError: 'visual.gaze_fixation'`.

- [ ] **Step 3: Implement** — in `blitz_overlay/weights.py`, add inside `CUE_WEIGHTS` (before the closing brace):

```python
    "visual.gaze_fixation": {
        "effect_size_d": 0.50, "reliability_tier": 2,
        "family": "visual", "region": "eyes",
        "citation": (
            "Catalog cue 56 — gaze fixation pattern (count/duration); fabrication = more"
            " frequent, shorter fixations vs recall; 70-80% with ML (CUE_CATALOG.md)."
        ),
    },
    "visual.pupil_dilation": {
        "effect_size_d": 0.40, "reliability_tier": 2,
        "family": "visual", "region": "eyes",
        "citation": (
            "Catalog cue 7/55 — pupil/iris dilation, cognitive-load spike; 65-75% alone."
            " Reliable only at 720p+, so quality is scaled down at low webcam resolution."
        ),
    },
    "visual.asymmetric_smile": {
        "effect_size_d": 0.35, "reliability_tier": 2,
        "family": "visual", "region": "mouth",
        "citation": (
            "Catalog cue 5 — smile asymmetry (Duchenne vs fake), AU6/AU12 left-right"
            " asymmetry (CUE_CATALOG.md)."
        ),
    },
    "visual.nose_wrinkle": {
        "effect_size_d": 0.28, "reliability_tier": 3,
        "family": "visual", "region": "mouth",
        "citation": (
            "Catalog cue 4 — nose wrinkle (AU9), disgust/discomfort; weak-moderate."
        ),
    },
    "visual.eye_blocking": {
        "effect_size_d": 0.28, "reliability_tier": 3,
        "family": "visual", "region": "eyes",
        "citation": (
            "Catalog cue 13 — eye blocking (prolonged eye closure while speaking),"
            " blink-duration classifier; weak-moderate."
        ),
    },
    "visual.eye_widen": {
        "effect_size_d": 0.25, "reliability_tier": 3,
        "family": "visual", "region": "eyes",
        "citation": (
            "Catalog cue 9-adjacent — eye widen (AU5, eyeWide), surprise/fear leakage;"
            " weak single-cue diagnosticity."
        ),
    },
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/overlay/test_weights.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add blitz_overlay/weights.py tests/overlay/test_weights.py
git commit -m "feat(overlay): science weights for 6 new facial cues

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: The 6 facial cue detectors

**Files:**
- Modify: `blitz_overlay/cues/visual.py`
- Test: `tests/overlay/test_visual_cues.py`

- [ ] **Step 1: Write the failing test** — append to `tests/overlay/test_visual_cues.py` (add imports for the new classes at the top alongside the existing visual imports):

```python
def _bs_frame(ts, **bs):
    from blitz_overlay.schemas import FeatureFrame
    return FeatureFrame.from_dict({"ts": ts, "face_present": True, "confidence": 0.9,
                                   "blendshapes": bs, "geometry": {}})


def _geo_frame(ts, **geo):
    from blitz_overlay.schemas import FeatureFrame
    return FeatureFrame.from_dict({"ts": ts, "face_present": True, "confidence": 0.9,
                                   "blendshapes": {}, "geometry": geo})


def test_visual_registry_has_eleven_detectors():
    from blitz_overlay.cues.visual import VISUAL_DETECTORS
    assert len(VISUAL_DETECTORS) == 11
    ids = {d().cue_id for d in VISUAL_DETECTORS}
    assert {"visual.gaze_fixation", "visual.pupil_dilation", "visual.eye_blocking",
            "visual.eye_widen", "visual.nose_wrinkle", "visual.asymmetric_smile"}.issubset(ids)


def test_pupil_dilation_reads_iris_ratio():
    from blitz_overlay.cues.visual import PupilDilation
    d = PupilDilation()
    assert abs(d.measure(_geo_frame(0, iris_ratio=0.42)) - 0.42) < 1e-9
    assert d.measure(_geo_frame(0)) is None                 # no iris_ratio -> abstain
    assert d.measure(_geo_frame(0, iris_ratio=None)) is None


def test_pupil_dilation_quality_scaled_for_low_res():
    from blitz_overlay.cues.visual import PupilDilation
    d = PupilDilation()
    # quality is scaled below raw confidence (catalog: needs 720p+)
    assert d.quality(_geo_frame(0, iris_ratio=0.4)) < 0.9


def test_eye_widen_takes_max_side():
    from blitz_overlay.cues.visual import EyeWiden
    d = EyeWiden()
    assert d.measure(_bs_frame(0, eyeWideLeft=0.2, eyeWideRight=0.7)) == 0.7
    assert d.measure(_bs_frame(0)) is None


def test_nose_wrinkle_takes_max_side():
    from blitz_overlay.cues.visual import NoseWrinkle
    d = NoseWrinkle()
    assert d.measure(_bs_frame(0, noseSneerLeft=0.3, noseSneerRight=0.1)) == 0.3
    assert d.measure(_bs_frame(0)) is None


def test_asymmetric_smile_is_absolute_difference():
    from blitz_overlay.cues.visual import AsymmetricSmile
    d = AsymmetricSmile()
    m = d.measure(_bs_frame(0, mouthSmileLeft=0.7, mouthSmileRight=0.2))
    assert abs(m - 0.5) < 1e-9
    assert d.measure(_bs_frame(0)) is None


def test_eye_blocking_accumulates_closed_duration():
    from blitz_overlay.cues.visual import EyeBlocking
    d = EyeBlocking()
    assert d.measure(_bs_frame(0, eyeBlinkLeft=0.9, eyeBlinkRight=0.9)) == 0.0   # just closed
    held = d.measure(_bs_frame(1500, eyeBlinkLeft=0.9, eyeBlinkRight=0.9))       # 1.5s closed
    assert abs(held - 1.5) < 1e-6
    assert d.measure(_bs_frame(2000, eyeBlinkLeft=0.0, eyeBlinkRight=0.0)) == 0.0  # eyes open -> reset


def test_gaze_fixation_measures_darting_velocity():
    from blitz_overlay.cues.visual import GazeFixation
    d = GazeFixation()
    # steady gaze -> ~0 velocity
    for t in range(0, 1600, 100):
        v = d.measure(_geo_frame(t, gaze_x=0.1, gaze_y=0.0))
    assert v < 0.05
    # darting gaze -> higher velocity
    d2 = GazeFixation()
    last = 0.0
    for i, t in enumerate(range(0, 1600, 100)):
        gx = 0.4 if i % 2 else -0.4
        last = d2.measure(_geo_frame(t, gaze_x=gx, gaze_y=0.0))
    assert last > 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/overlay/test_visual_cues.py -q`
Expected: FAIL — `ImportError`/`AttributeError` for the new classes.

- [ ] **Step 3: Implement** — in `blitz_overlay/cues/visual.py`, add these detectors before the `VISUAL_DETECTORS = [...]` line:

```python
EYE_CLOSED_THRESHOLD = 0.5       # eyeBlink coefficient above this = eye held closed
GAZE_FIX_WINDOW_MS = 1500        # window for gaze darting velocity


class GazeFixation(CueDetector):
    """Gaze darting velocity — fabrication = more frequent, shorter fixations (catalog cue 56).

    Mean per-sample gaze movement over a ~1.5 s window. High = darting (suspicious); low =
    steady fixation. Distinct from gaze_aversion, which measures sustained off-centre duration.
    """

    cue_id = "visual.gaze_fixation"
    direction = 1

    def __init__(self) -> None:
        super().__init__()
        self._hist: deque[tuple[int, float, float]] = deque()

    def measure(self, frame: FeatureFrame) -> float | None:
        g = frame.geometry
        gx, gy = g.get("gaze_x"), g.get("gaze_y")
        if gx is None and gy is None:
            return None
        now = frame.ts
        self._hist.append((now, float(gx or 0.0), float(gy or 0.0)))
        while self._hist and self._hist[0][0] < now - GAZE_FIX_WINDOW_MS:
            self._hist.popleft()
        if len(self._hist) < 2:
            return 0.0
        steps = 0.0
        for (_, x0, y0), (_, x1, y1) in zip(self._hist, list(self._hist)[1:]):
            steps += ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        return steps / (len(self._hist) - 1)


class PupilDilation(CueDetector):
    """Pupil/iris dilation proxy — cognitive-load spike (catalog cue 7/55).

    Reads geometry.iris_ratio (iris diameter ÷ eye width). Quality is scaled down because
    the catalog notes this is only reliable at 720p+.
    """

    cue_id = "visual.pupil_dilation"
    direction = 1

    def measure(self, frame: FeatureFrame) -> float | None:
        val = frame.geometry.get("iris_ratio")
        return None if val is None else float(val)

    def quality(self, frame: FeatureFrame) -> float:
        return 0.5 * max(0.0, min(1.0, frame.confidence))


class EyeBlocking(CueDetector):
    """Eye blocking — prolonged eye closure *duration* while speaking (catalog cue 13)."""

    cue_id = "visual.eye_blocking"
    direction = 1

    def __init__(self) -> None:
        super().__init__()
        self._closed_since: int | None = None

    def measure(self, frame: FeatureFrame) -> float | None:
        bs = frame.blendshapes
        if "eyeBlinkLeft" not in bs and "eyeBlinkRight" not in bs:
            return None
        closed = max(bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0)) >= EYE_CLOSED_THRESHOLD
        now = frame.ts
        if closed:
            if self._closed_since is None:
                self._closed_since = now
            return (now - self._closed_since) / 1000.0
        self._closed_since = None
        return 0.0


class EyeWiden(CueDetector):
    """Eye widen (AU5, eyeWide) — surprise/fear leakage (catalog cue 9-adjacent)."""

    cue_id = "visual.eye_widen"
    direction = 1

    def measure(self, frame: FeatureFrame) -> float | None:
        bs = frame.blendshapes
        if "eyeWideLeft" not in bs and "eyeWideRight" not in bs:
            return None
        return max(bs.get("eyeWideLeft", 0.0), bs.get("eyeWideRight", 0.0))


class NoseWrinkle(CueDetector):
    """Nose wrinkle (AU9, noseSneer) — disgust/discomfort (catalog cue 4)."""

    cue_id = "visual.nose_wrinkle"
    direction = 1

    def measure(self, frame: FeatureFrame) -> float | None:
        bs = frame.blendshapes
        if "noseSneerLeft" not in bs and "noseSneerRight" not in bs:
            return None
        return max(bs.get("noseSneerLeft", 0.0), bs.get("noseSneerRight", 0.0))


class AsymmetricSmile(CueDetector):
    """Smile asymmetry (AU6/AU12 left-right) — fake vs Duchenne (catalog cue 5)."""

    cue_id = "visual.asymmetric_smile"
    direction = 1

    def measure(self, frame: FeatureFrame) -> float | None:
        bs = frame.blendshapes
        if "mouthSmileLeft" not in bs and "mouthSmileRight" not in bs:
            return None
        return abs(bs.get("mouthSmileLeft", 0.0) - bs.get("mouthSmileRight", 0.0))
```

and extend the registry:

```python
VISUAL_DETECTORS = [
    BlinkRate, GazeAversion, BrowFlash, LipPress, JawTension,
    GazeFixation, PupilDilation, EyeBlocking, EyeWiden, NoseWrinkle, AsymmetricSmile,
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/overlay/test_visual_cues.py -q`
Expected: PASS.

- [ ] **Step 5: Run full suite + lint**

Run: `python3 -m pytest -q` then `python3 -m ruff check .`
Expected: all green (the pipeline/consensus/mixer pick up the new detectors automatically via `VISUAL_DETECTORS`).

- [ ] **Step 6: Commit**

```bash
git add blitz_overlay/cues/visual.py tests/overlay/test_visual_cues.py
git commit -m "feat(overlay): 6 new facial cues — gaze fixation, pupil, eye blocking/widen, nose wrinkle, smile asymmetry

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Manual browser verification (user-confirmed)

**Files:** none.

- [ ] **Step 1: Launch** `BLITZ_OVERLAY_BASELINE_SECONDS=20 python3 -m blitz_overlay`; open in Chrome.
- [ ] **Step 2: Verify** the Cue Mixer now shows **11 visual lanes**; widen eyes / wrinkle nose / lopsided smile / dart eyes / hold eyes shut → the matching lanes light up. Pupil lane reacts but stays dimmer (low-res quality scaling).
- [ ] **Step 3: Report to user.** In-browser confirmation gate before merge.
```
