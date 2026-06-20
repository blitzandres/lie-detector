# Tier-1 Cue Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add 10 browser-native visual cues from MediaPipe data we already capture (unused blendshapes + head pose) → **Visual 11 → 21** (~32 cues total), filling the Cue Polygon — no new model, no extra RAM.

**Architecture:** The browser now forwards the previously-unused blendshapes (done in `apps/overlay-web/js/schema.js`) and already streams `head_pose`. This plan adds 10 Python `CueDetector`s + their science weights + tests. The pipeline/consensus/polygon pick them up automatically via `VISUAL_DETECTORS`.

**Tech Stack:** Python 3.14 (`python3`, no venv), pytest, ruff.

**Reference:** `docs/CUE_CATALOG.md`. Verify after each task: `python3 -m pytest -q` + `python3 -m ruff check .` green. Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Do NOT modify `core/*`.

The 10 cues: `head_movement` (head_pose, #14) · `eye_squint` · `mouth_stretch` · `mouth_frown` · `mouth_shrug` · `jaw_shift` · `jaw_drop` · `lip_roll` · `brow_outer_raise` · `contempt_asymmetry` (AU14).

---

## Task 1: Science weights for the 10 Tier-1 cues

**Files:**
- Modify: `blitz_overlay/weights.py`
- Test: `tests/overlay/test_weights.py`

- [ ] **Step 1: Write the failing test** — append to `tests/overlay/test_weights.py`:

```python
def test_tier1_cue_weights_present():
    from blitz_overlay.weights import CUE_WEIGHTS
    expected = {
        "visual.contempt_asymmetry": (0.30, 2, "mouth"),
        "visual.head_movement": (0.30, 3, "head"),
        "visual.mouth_stretch": (0.28, 3, "mouth"),
        "visual.lip_roll": (0.26, 3, "mouth"),
        "visual.eye_squint": (0.25, 3, "eyes"),
        "visual.mouth_frown": (0.25, 3, "mouth"),
        "visual.brow_outer_raise": (0.25, 3, "brow"),
        "visual.mouth_shrug": (0.24, 3, "mouth"),
        "visual.jaw_shift": (0.22, 3, "jaw"),
        "visual.jaw_drop": (0.22, 3, "jaw"),
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

Run: `python3 -m pytest tests/overlay/test_weights.py::test_tier1_cue_weights_present -v`
Expected: FAIL — `KeyError: 'visual.contempt_asymmetry'`.

- [ ] **Step 3: Implement** — in `blitz_overlay/weights.py`, add inside `CUE_WEIGHTS` (before the closing brace). Each is a weak-moderate single-cue affect/tension marker (tier 3 unless noted):

```python
    "visual.head_movement": {
        "effect_size_d": 0.30, "reliability_tier": 3, "family": "visual", "region": "head",
        "citation": "Catalog cue 14 — head-movement increase (restlessness/discomfort) via head-pose variance.",
    },
    "visual.eye_squint": {
        "effect_size_d": 0.25, "reliability_tier": 3, "family": "visual", "region": "eyes",
        "citation": "AU7 eye squint — tension/contempt micro-cue; weak single-cue diagnosticity.",
    },
    "visual.mouth_stretch": {
        "effect_size_d": 0.28, "reliability_tier": 3, "family": "visual", "region": "mouth",
        "citation": "AU20 lip stretch — fear/tension grimace; weak-moderate.",
    },
    "visual.mouth_frown": {
        "effect_size_d": 0.25, "reliability_tier": 3, "family": "visual", "region": "mouth",
        "citation": "AU15 lip-corner depressor — negative-affect leakage; weak.",
    },
    "visual.mouth_shrug": {
        "effect_size_d": 0.24, "reliability_tier": 3, "family": "visual", "region": "mouth",
        "citation": "AU17 chin raise / mouth shrug — doubt / uncertainty emblem; weak.",
    },
    "visual.jaw_shift": {
        "effect_size_d": 0.22, "reliability_tier": 3, "family": "visual", "region": "jaw",
        "citation": "Lateral/forward jaw displacement — jaw tension proxy; weak.",
    },
    "visual.jaw_drop": {
        "effect_size_d": 0.22, "reliability_tier": 3, "family": "visual", "region": "jaw",
        "citation": "AU26 jaw drop / mouth opening — surprise/affect; weak.",
    },
    "visual.lip_roll": {
        "effect_size_d": 0.26, "reliability_tier": 3, "family": "visual", "region": "mouth",
        "citation": "Lip suck/roll (AU28-adjacent) — withholding marker; weak-moderate.",
    },
    "visual.brow_outer_raise": {
        "effect_size_d": 0.25, "reliability_tier": 3, "family": "visual", "region": "brow",
        "citation": "AU2 outer-brow raise — surprise / overemphasis; weak.",
    },
    "visual.contempt_asymmetry": {
        "effect_size_d": 0.30, "reliability_tier": 2, "family": "visual", "region": "mouth",
        "citation": "AU14 unilateral contempt — left-right mouth-dimple asymmetry; moderate micro-expression marker.",
    },
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/overlay/test_weights.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add blitz_overlay/weights.py tests/overlay/test_weights.py
git commit -m "feat(overlay): science weights for 10 Tier-1 visual cues

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: The 10 Tier-1 cue detectors

**Files:**
- Modify: `blitz_overlay/cues/visual.py`
- Test: `tests/overlay/test_visual_cues.py`

- [ ] **Step 1: Write the failing test** — append to `tests/overlay/test_visual_cues.py` (reuse the
`_bs_frame`/`_geo_frame` helpers already added in that file by the facial-cue tests; add a head-pose helper):

```python
def _hp_frame(ts, yaw=0.0, pitch=0.0, roll=0.0):
    from blitz_overlay.schemas import FeatureFrame
    return FeatureFrame.from_dict({"ts": ts, "face_present": True, "confidence": 0.9,
                                   "blendshapes": {}, "geometry": {},
                                   "head_pose": {"yaw": yaw, "pitch": pitch, "roll": roll}})


def test_visual_registry_has_twentyone_detectors():
    from blitz_overlay.cues.visual import VISUAL_DETECTORS
    assert len(VISUAL_DETECTORS) == 21
    ids = {d().cue_id for d in VISUAL_DETECTORS}
    assert {"visual.head_movement", "visual.eye_squint", "visual.mouth_stretch",
            "visual.mouth_frown", "visual.mouth_shrug", "visual.jaw_shift", "visual.jaw_drop",
            "visual.lip_roll", "visual.brow_outer_raise", "visual.contempt_asymmetry"}.issubset(ids)


def test_max_blendshape_cues_take_max_and_abstain():
    from blitz_overlay.cues.visual import EyeSquint, JawShift, JawDrop
    assert EyeSquint().measure(_bs_frame(0, eyeSquintLeft=0.2, eyeSquintRight=0.6)) == 0.6
    assert EyeSquint().measure(_bs_frame(0)) is None
    assert JawShift().measure(_bs_frame(0, jawLeft=0.1, jawRight=0.4, jawForward=0.2)) == 0.4
    assert JawDrop().measure(_bs_frame(0, jawOpen=0.55)) == 0.55


def test_contempt_asymmetry_is_absolute_difference():
    from blitz_overlay.cues.visual import ContemptAsymmetry
    d = ContemptAsymmetry()
    assert abs(d.measure(_bs_frame(0, mouthDimpleLeft=0.6, mouthDimpleRight=0.1)) - 0.5) < 1e-9
    assert d.measure(_bs_frame(0)) is None


def test_head_movement_accumulates_over_window():
    from blitz_overlay.cues.visual import HeadMovement
    d = HeadMovement()
    # steady head -> ~0 movement
    last = 0.0
    for t in range(0, 1600, 100):
        last = d.measure(_hp_frame(t, yaw=5.0, pitch=2.0, roll=1.0))
    assert last < 0.5
    # jerky head -> larger movement
    d2 = HeadMovement()
    last2 = 0.0
    for i, t in enumerate(range(0, 1600, 100)):
        last2 = d2.measure(_hp_frame(t, yaw=20.0 if i % 2 else -20.0, pitch=0.0, roll=0.0))
    assert last2 > 5.0


def test_head_movement_abstains_without_head_pose():
    from blitz_overlay.cues.visual import HeadMovement
    from blitz_overlay.schemas import FeatureFrame
    f = FeatureFrame.from_dict({"ts": 0, "face_present": True, "confidence": 0.9})
    assert HeadMovement().measure(f) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/overlay/test_visual_cues.py -q`
Expected: FAIL — `ImportError` for the new classes / registry length 11 ≠ 21.

- [ ] **Step 3: Implement** — in `blitz_overlay/cues/visual.py`, add before the `VISUAL_DETECTORS = [...]` line:

```python
HEAD_MOVE_WINDOW_MS = 2000


class _MaxBlendshapeCue(CueDetector):
    """Base for cues that are the max of a set of blendshape coefficients."""

    direction = 1
    keys: tuple[str, ...] = ()

    def measure(self, frame: FeatureFrame) -> float | None:
        bs = frame.blendshapes
        if not any(k in bs for k in self.keys):
            return None
        return max(bs.get(k, 0.0) for k in self.keys)


class EyeSquint(_MaxBlendshapeCue):
    cue_id = "visual.eye_squint"
    keys = ("eyeSquintLeft", "eyeSquintRight")


class MouthStretch(_MaxBlendshapeCue):
    cue_id = "visual.mouth_stretch"
    keys = ("mouthStretchLeft", "mouthStretchRight")


class MouthFrown(_MaxBlendshapeCue):
    cue_id = "visual.mouth_frown"
    keys = ("mouthFrownLeft", "mouthFrownRight")


class MouthShrug(_MaxBlendshapeCue):
    cue_id = "visual.mouth_shrug"
    keys = ("mouthShrugUpper", "mouthShrugLower")


class JawShift(_MaxBlendshapeCue):
    cue_id = "visual.jaw_shift"
    keys = ("jawLeft", "jawRight", "jawForward")


class JawDrop(_MaxBlendshapeCue):
    cue_id = "visual.jaw_drop"
    keys = ("jawOpen",)


class LipRoll(_MaxBlendshapeCue):
    cue_id = "visual.lip_roll"
    keys = ("mouthRollUpper", "mouthRollLower")


class BrowOuterRaise(_MaxBlendshapeCue):
    cue_id = "visual.brow_outer_raise"
    keys = ("browOuterUpLeft", "browOuterUpRight")


class ContemptAsymmetry(CueDetector):
    """Unilateral contempt (AU14) — left-right mouth-dimple asymmetry."""

    cue_id = "visual.contempt_asymmetry"
    direction = 1

    def measure(self, frame: FeatureFrame) -> float | None:
        bs = frame.blendshapes
        if "mouthDimpleLeft" not in bs and "mouthDimpleRight" not in bs:
            return None
        return abs(bs.get("mouthDimpleLeft", 0.0) - bs.get("mouthDimpleRight", 0.0))


class HeadMovement(CueDetector):
    """Head-movement magnitude over a ~2 s window — restlessness/discomfort (catalog cue 14)."""

    cue_id = "visual.head_movement"
    direction = 1

    def __init__(self) -> None:
        super().__init__()
        self._hist: deque[tuple[int, float, float, float]] = deque()

    def measure(self, frame: FeatureFrame) -> float | None:
        hp = frame.head_pose
        if not hp or not any(k in hp for k in ("yaw", "pitch", "roll")):
            return None
        now = frame.ts
        self._hist.append((now, float(hp.get("yaw", 0.0)),
                           float(hp.get("pitch", 0.0)), float(hp.get("roll", 0.0))))
        while self._hist and self._hist[0][0] < now - HEAD_MOVE_WINDOW_MS:
            self._hist.popleft()
        if len(self._hist) < 2:
            return 0.0
        steps = 0.0
        hist = list(self._hist)
        for (_, y0, p0, r0), (_, y1, p1, r1) in zip(hist, hist[1:], strict=False):
            steps += ((y1 - y0) ** 2 + (p1 - p0) ** 2 + (r1 - r0) ** 2) ** 0.5
        return steps / (len(hist) - 1)
```

and extend the registry (keep the existing 11, append the 10 new):

```python
VISUAL_DETECTORS = [
    BlinkRate, GazeAversion, BrowFlash, LipPress, JawTension,
    GazeFixation, PupilDilation, EyeBlocking, EyeWiden, NoseWrinkle, AsymmetricSmile,
    HeadMovement, EyeSquint, MouthStretch, MouthFrown, MouthShrug,
    JawShift, JawDrop, LipRoll, BrowOuterRaise, ContemptAsymmetry,
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/overlay/test_visual_cues.py -q`
Expected: PASS.

- [ ] **Step 5: Run full suite + lint**

Run: `python3 -m pytest -q` then `python3 -m ruff check .`
Expected: all green (pipeline/consensus/polygon pick up the new detectors via `VISUAL_DETECTORS`).

- [ ] **Step 6: Commit**

```bash
git add blitz_overlay/cues/visual.py tests/overlay/test_visual_cues.py
git commit -m "feat(overlay): 10 Tier-1 visual cues (head movement, squint, stretch, frown, shrug, jaw, lip roll, brow, contempt)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Manual verification (user-confirmed)

**Files:** none.

- [ ] **Step 1:** Hard-refresh `http://127.0.0.1:8000`. The Cue Polygon should now show **~21 visual vertices**
  (V1…V21) plus audio/linguistic/physio.
- [ ] **Step 2:** Make faces — squint, stretch/frown your mouth, lopsided dimple (contempt), drop your jaw,
  move your head around — and confirm the matching `V*` vertices light + shoot to centre (hover to read which).
- [ ] **Step 3: Report to user.**
```
