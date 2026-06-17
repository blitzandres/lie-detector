# Linguistic Family Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the Linguistic family as the 4th live consensus voter, driven by a browser Web Speech transcript feeding person-relative lexicon cues reused from `modalities/linguistic/analyzer.py`.

**Architecture:** Browser `Transcriber` seam (Web Speech adapter) maintains a rolling word window and sends transcript `{text, seq}` over the existing WebSocket only when it changes. The Python engine runs 7 `CueDetector` lexicon cues over that window through the existing rolling baseline + family fusion + two-gate. A pipeline-level seq de-dup keeps baseline samples per-utterance (not per 30 Hz frame). Surfaced on enneagram slot 7 and an on-screen caption strip.

**Tech Stack:** Python 3.14 (no venv; run via `python3`), pytest, ruff; vanilla ES-module browser app (`webkitSpeechRecognition`, Canvas 2D); FastAPI WebSocket.

**Reference:** spec `docs/superpowers/specs/2026-06-16-linguistic-family-design.md`. Verify after every task: `python3 -m pytest -q` and `python3 -m ruff check .` stay green. Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## File Structure

**Python (engine):**
- Modify `blitz_overlay/schemas.py` — add `transcript: dict | None` to `FeatureFrame` (+ `from_dict`).
- Modify `blitz_overlay/weights.py` — add 7 `linguistic.*` weight entries.
- Create `blitz_overlay/cues/linguistic.py` — `LinguisticCue` base + 7 detectors + `LINGUISTIC_DETECTORS`.
- Modify `blitz_overlay/pipeline.py` — register detectors + per-utterance seq de-dup.
- Modify `blitz_overlay/consensus.py` — add `"linguistic"` to `WIRED_FAMILIES`.

**Browser:**
- Create `apps/overlay-web/js/transcriber.js` — `Transcriber` interface + `WebSpeechTranscriber`.
- Modify `apps/overlay-web/js/main.js` — start transcriber, attach `frame.transcript` on change, render caption.
- Modify `apps/overlay-web/index.html` — caption strip + cloud-STT notice DOM.
- Modify `apps/overlay-web/css/overlay.css` — caption/notice styles.
- Modify `apps/overlay-web/js/enneagram.js` — slot 7 lights on strongest active `linguistic.*` cue.

**Docs:**
- Modify `planning/INDEX.md` + `docs/OVERLAY_README.md` — honest external-network-call correction.

**Tests:**
- Modify `tests/overlay/test_schemas.py`, `tests/overlay/test_weights.py`, `tests/overlay/test_consensus.py`.
- Create `tests/overlay/test_linguistic_cues.py`, `tests/overlay/test_pipeline_linguistic.py`.

---

## Task 1: Schema — `transcript` field on FeatureFrame

**Files:**
- Modify: `blitz_overlay/schemas.py:26-52`
- Test: `tests/overlay/test_schemas.py`

- [ ] **Step 1: Write the failing test** — append to `tests/overlay/test_schemas.py`:

```python
def test_feature_frame_carries_transcript_block():
    from blitz_overlay.schemas import FeatureFrame
    frame = FeatureFrame.from_dict({
        "ts": 10, "face_present": True, "confidence": 0.9,
        "transcript": {"text": "i was at home all night", "seq": 3},
    })
    assert frame.transcript == {"text": "i was at home all night", "seq": 3}


def test_feature_frame_transcript_absent_is_none():
    from blitz_overlay.schemas import FeatureFrame
    frame = FeatureFrame.from_dict({"ts": 10, "face_present": True})
    assert frame.transcript is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/overlay/test_schemas.py::test_feature_frame_carries_transcript_block -v`
Expected: FAIL — `FeatureFrame` has no attribute `transcript` (TypeError/AttributeError).

- [ ] **Step 3: Implement** — in `blitz_overlay/schemas.py`, add the field after the `audio` line (line 37) inside `FeatureFrame`:

```python
    transcript: dict | None = None            # {"text": str, "seq": int} or None
```

and in `from_dict` (after the `audio=...` line ~51):

```python
            transcript=(dict(d["transcript"]) if d.get("transcript") else None),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/overlay/test_schemas.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add blitz_overlay/schemas.py tests/overlay/test_schemas.py
git commit -m "feat(overlay): add transcript block to FeatureFrame schema

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Science-driven linguistic weights

**Files:**
- Modify: `blitz_overlay/weights.py:14-99`
- Test: `tests/overlay/test_weights.py`

- [ ] **Step 1: Write the failing test** — append to `tests/overlay/test_weights.py`:

```python
def test_linguistic_weights_present_with_citations():
    from blitz_overlay.weights import CUE_WEIGHTS
    expected = {
        "linguistic.sensory_detail_poverty": (0.29, 2),
        "linguistic.pronoun_avoidance": (0.27, 2),
        "linguistic.distancing_language": (0.24, 2),
        "linguistic.filler_ratio": (0.23, 3),
        "linguistic.qualifier_overload": (0.21, 3),
        "linguistic.negative_emotion_density": (0.18, 3),
        "linguistic.lexical_diversity_drop": (0.16, 3),
    }
    for cue_id, (d, tier) in expected.items():
        spec = CUE_WEIGHTS[cue_id]
        assert spec["family"] == "linguistic"
        assert spec["region"] == "mouth"
        assert abs(spec["effect_size_d"] - d) < 1e-9
        assert spec["reliability_tier"] == tier
        assert spec["citation"]  # non-empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/overlay/test_weights.py::test_linguistic_weights_present_with_citations -v`
Expected: FAIL — `KeyError: 'linguistic.sensory_detail_poverty'`.

- [ ] **Step 3: Implement** — in `blitz_overlay/weights.py`, add these entries inside `CUE_WEIGHTS` (before the closing `}` at line 99):

```python
    "linguistic.sensory_detail_poverty": {
        "effect_size_d": 0.29, "reliability_tier": 2,
        "family": "linguistic", "region": "mouth",
        "citation": (
            "Catalog cue — Reality Monitoring sensory-detail poverty; truthful accounts carry"
            " more perceptual detail (RESEARCH.md RM/CBCA); person-relative, d~0.29."
        ),
    },
    "linguistic.pronoun_avoidance": {
        "effect_size_d": 0.27, "reliability_tier": 2,
        "family": "linguistic", "region": "mouth",
        "citation": (
            "Catalog cue — reduced first-person pronoun use / self-distancing under deception"
            " (Newman/Pennebaker LIWC; RESEARCH.md), d~0.27."
        ),
    },
    "linguistic.distancing_language": {
        "effect_size_d": 0.24, "reliability_tier": 2,
        "family": "linguistic", "region": "mouth",
        "citation": (
            "Catalog cue — distancing / third-person referents replacing direct reference"
            " (RESEARCH.md distancing language), d~0.24."
        ),
    },
    "linguistic.filler_ratio": {
        "effect_size_d": 0.23, "reliability_tier": 3,
        "family": "linguistic", "region": "mouth",
        "citation": (
            "Catalog cue — filler / hesitation marker rate as cognitive-load proxy"
            " (RESEARCH.md), weak-moderate d~0.23."
        ),
    },
    "linguistic.qualifier_overload": {
        "effect_size_d": 0.21, "reliability_tier": 3,
        "family": "linguistic", "region": "mouth",
        "citation": (
            "Catalog cue — qualifier / hedge overload signalling uncommitted speech"
            " (RESEARCH.md), weak d~0.21."
        ),
    },
    "linguistic.negative_emotion_density": {
        "effect_size_d": 0.18, "reliability_tier": 3,
        "family": "linguistic", "region": "mouth",
        "citation": (
            "Catalog cue — elevated negative-emotion word density (LIWC; RESEARCH.md),"
            " weak d~0.18."
        ),
    },
    "linguistic.lexical_diversity_drop": {
        "effect_size_d": 0.16, "reliability_tier": 3,
        "family": "linguistic", "region": "mouth",
        "citation": (
            "Catalog cue — reduced lexical diversity (type-token ratio) under cognitive load"
            " (RESEARCH.md), weak d~0.16 (anchor-tier)."
        ),
    },
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/overlay/test_weights.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add blitz_overlay/weights.py tests/overlay/test_weights.py
git commit -m "feat(overlay): science-driven linguistic cue weights + citations

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Linguistic cue detectors (reuse analyzer lexicons)

**Files:**
- Create: `blitz_overlay/cues/linguistic.py`
- Test: `tests/overlay/test_linguistic_cues.py`

- [ ] **Step 1: Write the failing test** — create `tests/overlay/test_linguistic_cues.py`:

```python
"""Tests for the linguistic cue detectors (reused analyzer.py lexicons).

Mirrors test_audio_cues.py:
  - a frame WITH a transcript window yields a float measurement
  - a frame WITHOUT a transcript block yields None
  - a too-short window yields None
  - LINGUISTIC_DETECTORS contains exactly the seven expected detectors
"""
from blitz_overlay.cues.linguistic import (
    LINGUISTIC_DETECTORS,
    FillerRatio,
    PronounAvoidance,
)
from blitz_overlay.schemas import FeatureFrame
from core.calibration import RollingBaseline


def _frame(ts, text, seq=1):
    return FeatureFrame.from_dict({
        "ts": ts, "face_present": True, "confidence": 0.9,
        "transcript": {"text": text, "seq": seq},
    })


def _no_transcript_frame(ts):
    return FeatureFrame.from_dict({"ts": ts, "face_present": True, "confidence": 0.9})


def test_registry_has_seven_linguistic_detectors():
    assert len(LINGUISTIC_DETECTORS) == 7
    ids = {d().cue_id for d in LINGUISTIC_DETECTORS}
    assert ids == {
        "linguistic.sensory_detail_poverty", "linguistic.pronoun_avoidance",
        "linguistic.distancing_language", "linguistic.filler_ratio",
        "linguistic.qualifier_overload", "linguistic.negative_emotion_density",
        "linguistic.lexical_diversity_drop",
    }


def test_filler_ratio_measures_window():
    d = FillerRatio()
    # 8 tokens, 2 fillers ("um", "like") -> ratio 0.25
    m = d.measure(_frame(0, "um i was like at the store yesterday"))
    assert abs(m - 0.25) < 1e-9


def test_returns_none_without_transcript():
    assert FillerRatio().measure(_no_transcript_frame(0)) is None


def test_returns_none_when_window_too_short():
    # < MIN_WORDS (5) tokens
    assert PronounAvoidance().measure(_frame(0, "i was home")) is None


def test_linguistic_cue_emits_event_after_calibration():
    """FillerRatio fires once a person-relative deviation exceeds z_threshold.

    Linguistic features are bounded ~[0,1], so the flat-baseline fallback (z = value - median,
    capped near 1.0) cannot reach z>=2. We therefore calibrate on speech with a small but
    NONZERO filler-ratio spread (alternating 0.0 and ~0.11), giving a real MAD, then inject a
    filler-saturated window whose ratio sits many MADs out -> z >> 2.
    """
    d = FillerRatio()
    rb = RollingBaseline(baseline_seconds=0, window_seconds=600)

    # Two calm windows: ratio 0.0 and ~0.111 -> median ~0.055, MAD ~0.055 (nonzero spread).
    calm = [
        "i went to the store and bought some food there",   # 10 tokens, 0 fillers -> 0.0
        "um i went to the store and bought food",            # 9 tokens, 1 filler  -> ~0.111
    ]
    seq = 0
    for i, t in enumerate(range(0, 6000, 200)):
        seq += 1
        v = d.measure(_frame(t, calm[i % 2], seq=seq))
        rb.update({"linguistic.filler_ratio": v if v is not None else 0.0}, ts_ms=t)

    # Filler-saturated window: 8 fillers / 11 tokens -> ~0.727, many MADs above baseline.
    hot = "um uh like you know basically i was actually literally there"
    ts, event = 6000, None
    for _ in range(20):
        seq += 1
        frame = _frame(ts, hot, seq=seq)
        v = d.measure(frame)
        rb.update({"linguistic.filler_ratio": v if v is not None else 0.0}, ts_ms=ts)
        event = d.update(frame, rb, value=v)
        ts += 200

    assert event is not None
    assert event.cue_id == "linguistic.filler_ratio"
    assert event.region == "mouth"
    assert event.z_score >= d.z_threshold
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/overlay/test_linguistic_cues.py -q`
Expected: FAIL — `ModuleNotFoundError: blitz_overlay.cues.linguistic`.

- [ ] **Step 3: Implement** — create `blitz_overlay/cues/linguistic.py`:

```python
"""Linguistic cue detectors reading a browser-supplied transcript window.

Each detector reads frame.transcript {"text", "seq"} and computes one lexicon feature by
reusing modalities/linguistic/analyzer.py (single source of truth for the word-lists and
feature math). When frame.transcript is absent (mic/recognizer off) or the window is too
short, measure() returns None and the linguistic family is simply absent from consensus.

Person-relative via the rolling baseline (catalog: linguistic cues are baseline-relative).
Honest framing: Web Speech transcription is a browser approximation; WhisperX (fully local)
is the documented upgrade behind the browser Transcriber seam.
"""
from __future__ import annotations

from blitz_overlay.cues.base import CueDetector
from blitz_overlay.schemas import FeatureFrame
from modalities.linguistic.analyzer import LinguisticAnalyzer

_ANALYZER = LinguisticAnalyzer()
MIN_WORDS = 5  # below this a window is too short to score reliably


class LinguisticCue(CueDetector):
    """Base for transcript-window lexicon cues. Subclasses set cue_id only.

    direction +1: the lexicon feature is defined so that the suspicious pattern is the
    HIGH direction (more fillers / more distancing / less first-person / less sensory detail).
    """

    direction = 1

    def measure(self, frame: FeatureFrame) -> float | None:
        t = frame.transcript
        if not t:
            return None
        text = t.get("text") or ""
        if len(_ANALYZER.tokenize(text)) < MIN_WORDS:
            return None
        return float(_ANALYZER.extract_features(text)[self.cue_id])

    def quality(self, frame: FeatureFrame) -> float:
        """Confidence scales with window length — short windows are noisy."""
        t = frame.transcript or {}
        n = len(_ANALYZER.tokenize(t.get("text") or ""))
        return max(0.0, min(1.0, n / 15.0))


class SensoryDetailPoverty(LinguisticCue):
    cue_id = "linguistic.sensory_detail_poverty"


class PronounAvoidance(LinguisticCue):
    cue_id = "linguistic.pronoun_avoidance"


class DistancingLanguage(LinguisticCue):
    cue_id = "linguistic.distancing_language"


class FillerRatio(LinguisticCue):
    cue_id = "linguistic.filler_ratio"


class QualifierOverload(LinguisticCue):
    cue_id = "linguistic.qualifier_overload"


class NegativeEmotionDensity(LinguisticCue):
    cue_id = "linguistic.negative_emotion_density"


class LexicalDiversityDrop(LinguisticCue):
    cue_id = "linguistic.lexical_diversity_drop"


LINGUISTIC_DETECTORS = [
    SensoryDetailPoverty, PronounAvoidance, DistancingLanguage, FillerRatio,
    QualifierOverload, NegativeEmotionDensity, LexicalDiversityDrop,
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/overlay/test_linguistic_cues.py -q`
Expected: PASS (7 tests). If `test_filler_ratio_measures_window` mismatches, check the analyzer's `FILLERS` set covers "um"/"like" (it does) and that the token count is 8.

- [ ] **Step 5: Commit**

```bash
git add blitz_overlay/cues/linguistic.py tests/overlay/test_linguistic_cues.py
git commit -m "feat(overlay): linguistic cue detectors reusing analyzer lexicons

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Pipeline registration + per-utterance seq de-dup

**Files:**
- Modify: `blitz_overlay/pipeline.py:7-79`
- Test: `tests/overlay/test_pipeline_linguistic.py`

**Why the de-dup:** linguistic windows change ~2 Hz but frames arrive ~30 Hz. Feeding the
same window every frame would collapse the baseline MAD and manufacture false spikes. The
session remembers the last processed transcript `seq`; a repeated `seq` is treated as absent
(its `transcript` is cleared before measuring), so baseline samples stay per-utterance.

- [ ] **Step 1: Write the failing test** — create `tests/overlay/test_pipeline_linguistic.py`:

```python
"""Pipeline integration for the linguistic family + per-utterance seq de-dup."""
from blitz_overlay.pipeline import OverlaySession


def _frame(ts, text, seq):
    return {
        "ts": ts, "face_present": True, "confidence": 0.9,
        "transcript": {"text": text, "seq": seq},
    }


def test_linguistic_detectors_registered():
    s = OverlaySession(baseline_seconds=0)
    ids = {d.cue_id for d in s.detectors}
    assert "linguistic.filler_ratio" in ids
    assert "linguistic.pronoun_avoidance" in ids


def test_duplicate_seq_not_refed_to_baseline():
    """A repeated transcript seq must not add another baseline observation."""
    s = OverlaySession(baseline_seconds=0)
    s.process(_frame(0, "um i was like at the store yesterday", seq=1))
    n_after_first = s.baseline.observation_count("linguistic.filler_ratio")
    assert n_after_first >= 1
    # Same seq again on the next frame -> treated as absent -> no new observation
    s.process(_frame(33, "um i was like at the store yesterday", seq=1))
    assert s.baseline.observation_count("linguistic.filler_ratio") == n_after_first
    # New seq -> a new observation is added
    s.process(_frame(66, "the person took that thing away from someone", seq=2))
    assert s.baseline.observation_count("linguistic.filler_ratio") == n_after_first + 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/overlay/test_pipeline_linguistic.py -q`
Expected: FAIL — `linguistic.filler_ratio` not registered / duplicate seq re-fed.

- [ ] **Step 3: Implement** — edit `blitz_overlay/pipeline.py`:

(a) Add the import near the other cue imports (after line 9):

```python
from blitz_overlay.cues.linguistic import LINGUISTIC_DETECTORS
```

(b) Extend the detector list in `__init__` (the `self.detectors = (...)` block, ~line 22):

```python
        self.detectors = (
            [cls() for cls in VISUAL_DETECTORS]
            + [cls() for cls in AUDIO_DETECTORS]
            + [cls() for cls in LINGUISTIC_DETECTORS]
            + [RppgHeartRate(fps=fps)]
        )
```

(c) Add a seq tracker at the end of `__init__`:

```python
        self._last_transcript_seq: int | None = None
```

(d) In `process`, immediately after `frame = FeatureFrame.from_dict(raw)` (line 35), add the de-dup gate:

```python
        # Per-utterance gate: a repeated transcript seq is treated as absent so the
        # rolling baseline samples linguistic cues per utterance, not per 30 Hz frame.
        if frame.transcript is not None:
            seq = frame.transcript.get("seq")
            if seq is not None and seq == self._last_transcript_seq:
                frame.transcript = None
            else:
                self._last_transcript_seq = seq
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/overlay/test_pipeline_linguistic.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add blitz_overlay/pipeline.py tests/overlay/test_pipeline_linguistic.py
git commit -m "feat(overlay): register linguistic detectors + per-utterance seq de-dup

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Wire linguistic as a consensus voter

**Files:**
- Modify: `blitz_overlay/consensus.py:8`
- Test: `tests/overlay/test_consensus.py:50-58` (update) + new FLAG test

- [ ] **Step 1: Update the failing assertion + add the new test** — in `tests/overlay/test_consensus.py`:

Change the existing `test_unwired_families_shown_not_fresh` line:

```python
    assert names["linguistic"].wired is False
```

to:

```python
    assert names["linguistic"].wired is True   # linguistic is now a wired family
```

Then append a new test proving FLAG no longer requires rPPG:

```python
def test_flag_reachable_via_visual_and_linguistic():
    cb = ConsensusBuilder()
    cues = [_cue("visual.gaze_aversion", Modality.VISUAL, 7.0, "eyes", d=0.7),
            _cue("linguistic.pronoun_avoidance", Modality.LINGUISTIC, 7.0, "mouth", d=0.27)]
    out = cb.build(cues=cues, calibrating=False, ts=1000,
                   regions={"visual.gaze_aversion": "eyes",
                            "linguistic.pronoun_avoidance": "mouth"})
    assert out.status == "FLAG"
    assert out.flag is True
    assert out.n_agree == 2
```

- [ ] **Step 2: Run tests to verify the new one fails**

Run: `python3 -m pytest tests/overlay/test_consensus.py::test_flag_reachable_via_visual_and_linguistic -v`
Expected: FAIL — linguistic not wired, so its vote is suppressed (`vote = ... and wired`), `n_agree == 1`, status `WATCH`.

- [ ] **Step 3: Implement** — in `blitz_overlay/consensus.py` line 8:

```python
WIRED_FAMILIES = {"visual", "physio", "audio", "linguistic"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/overlay/test_consensus.py -q`
Expected: PASS (including the updated `test_unwired_families_shown_not_fresh`).

- [ ] **Step 5: Commit**

```bash
git add blitz_overlay/consensus.py tests/overlay/test_consensus.py
git commit -m "feat(overlay): wire linguistic as 4th consensus voter

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Full suite + lint gate

**Files:** none (verification task).

- [ ] **Step 1: Run the whole suite**

Run: `python3 -m pytest -q`
Expected: PASS — all prior tests plus the new linguistic/schema/pipeline/consensus tests.

- [ ] **Step 2: Run the linter**

Run: `python3 -m ruff check .`
Expected: `All checks passed!`

- [ ] **Step 3: Fix anything red, then re-run both.** No commit if nothing changed; otherwise:

```bash
git add -A
git commit -m "chore(overlay): lint + full-suite green after linguistic family

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Browser — Transcriber seam (Web Speech adapter)

**Files:**
- Create: `apps/overlay-web/js/transcriber.js`

> No JS test runner exists in this repo; browser tasks are verified manually in-browser
> (Task 11). Keep the adapter small and defensive — a missing/denied recognizer must never throw.

- [ ] **Step 1: Implement** — create `apps/overlay-web/js/transcriber.js`:

```javascript
/**
 * Transcriber — live transcript source for the Linguistic family.
 *
 * Interface (the seam): start(), stop(), latest() -> { text, seq }, available, supported.
 * `seq` increments only when the rolling window text changes, so the engine can sample
 * linguistic cues per-utterance instead of per video frame.
 *
 * WebSpeechTranscriber wraps Chrome's webkitSpeechRecognition. HONEST FRAMING: Chrome
 * streams mic audio to Google for transcription — the only path by which audio leaves the
 * device. A fully-local LocalWhisperTranscriber can drop in behind this same interface
 * later with no engine change.
 */

const WINDOW_WORDS = 40;       // rolling window cap (~last 12s of speech)
const WINDOW_MS = 12000;       // drop words older than this

export class WebSpeechTranscriber {
  constructor() {
    this.supported = !!(window.SpeechRecognition || window.webkitSpeechRecognition);
    this.available = false;
    this._recog = null;
    this._words = [];           // [{ word, ts }]
    this._seq = 0;
    this._text = "";
  }

  start() {
    if (!this.supported) {
      console.warn("[Transcriber] Web Speech API unsupported — linguistic family disabled.");
      return;
    }
    try {
      const Ctor = window.SpeechRecognition || window.webkitSpeechRecognition;
      const r = new Ctor();
      r.continuous = true;
      r.interimResults = true;
      r.lang = "en-US";
      r.onresult = (e) => this._onResult(e);
      r.onerror = (e) => console.warn("[Transcriber] error:", e.error);
      r.onend = () => { if (this.available) { try { r.start(); } catch { /* already starting */ } } };
      r.start();
      this._recog = r;
      this.available = true;
    } catch (err) {
      console.warn("[Transcriber] start failed (non-fatal):", err.message);
      this.available = false;
    }
  }

  stop() {
    this.available = false;
    if (this._recog) { try { this._recog.stop(); } catch { /* noop */ } }
  }

  /** Latest rolling-window snapshot. seq advances only when the window text changes. */
  latest() {
    return { text: this._text, seq: this._seq };
  }

  _onResult(event) {
    const now = Date.now();
    // Collect the newest transcript fragment (interim or final) and split into words.
    let fragment = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      fragment += event.results[i][0].transcript + " ";
    }
    const newWords = fragment.trim().split(/\s+/).filter(Boolean);
    if (newWords.length === 0) return;

    // For interim results we replace the tail; simplest robust approach: append finals only,
    // and keep a short interim overlay. Here we append all and let the window cap bound growth.
    for (const w of newWords) this._words.push({ word: w, ts: now });

    // Trim by age and count
    const cutoff = now - WINDOW_MS;
    this._words = this._words.filter((x) => x.ts >= cutoff);
    if (this._words.length > WINDOW_WORDS) {
      this._words = this._words.slice(this._words.length - WINDOW_WORDS);
    }

    const text = this._words.map((x) => x.word).join(" ");
    if (text !== this._text) {
      this._text = text;
      this._seq += 1;
    }
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add apps/overlay-web/js/transcriber.js
git commit -m "feat(overlay): browser Transcriber seam — Web Speech adapter

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Browser — wire transcriber into main loop

**Files:**
- Modify: `apps/overlay-web/js/main.js:1-68`

- [ ] **Step 1: Implement** — edit `apps/overlay-web/js/main.js`:

(a) Add the import after the `AudioCapture` import (line 4):

```javascript
import { WebSpeechTranscriber } from "./transcriber.js";
```

(b) Construct it near the other captures (after `const audio = ...`, ~line 23):

```javascript
const transcriber = new WebSpeechTranscriber();
let _lastTranscriptSeq = -1;
```

(c) Start it inside the existing mic try/catch in `start()` (right after `await audio.start();`):

```javascript
    transcriber.start();
```

(d) In `loop()`, after the audio attach block (~line 62), attach transcript only when it changed:

```javascript
  // Attach transcript only when the window changed (per-utterance, not per frame).
  if (transcriber.available) {
    const t = transcriber.latest();
    if (t.seq !== _lastTranscriptSeq && t.text) {
      frame.transcript = t;
      _lastTranscriptSeq = t.seq;
    }
  }
```

- [ ] **Step 2: Verify import resolves** — start the server and load the page (full check in Task 11):

Run: `BLITZ_OVERLAY_OPEN_BROWSER=0 python3 -m blitz_overlay &` then open `http://127.0.0.1:8000` in Chrome and confirm no console error about `transcriber.js`. Stop the server after (`kill %1`).

- [ ] **Step 3: Commit**

```bash
git add apps/overlay-web/js/main.js
git commit -m "feat(overlay): attach live transcript to feature frames

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: Browser — caption strip + honest cloud-STT notice

**Files:**
- Modify: `apps/overlay-web/index.html`
- Modify: `apps/overlay-web/css/overlay.css`
- Modify: `apps/overlay-web/js/main.js`

- [ ] **Step 1: Add DOM** — in `apps/overlay-web/index.html`, add a caption strip + notice below the video container (place near the `<canvas id="overlay">`; match existing element nesting):

```html
    <div id="caption" class="caption" aria-live="polite"></div>
    <div id="stt-notice" class="stt-notice">Transcript via Chrome cloud STT
      <button id="stt-toggle" type="button">disable</button>
    </div>
```

- [ ] **Step 2: Style it** — append to `apps/overlay-web/css/overlay.css`:

```css
.caption {
  position: absolute;
  left: 50%;
  bottom: 56px;
  transform: translateX(-50%);
  max-width: 80%;
  padding: 6px 12px;
  font: 14px/1.4 monospace;
  color: #e6edf3;
  background: rgba(11, 15, 20, 0.62);
  border-radius: 6px;
  text-align: center;
  pointer-events: none;
  min-height: 1.4em;
}
.stt-notice {
  position: absolute;
  left: 50%;
  bottom: 28px;
  transform: translateX(-50%);
  font: 11px/1 monospace;
  color: #7d8da3;
}
.stt-notice button {
  margin-left: 6px;
  font: 11px/1 monospace;
  color: #5b8def;
  background: none;
  border: none;
  cursor: pointer;
  text-decoration: underline;
}
```

> If `index.html`'s video wrapper is not `position: relative`, add `position: relative;` to its
> rule so the absolutely-positioned caption anchors to the video, matching how the overlay
> canvas is already positioned.

- [ ] **Step 3: Update the caption + toggle in JS** — in `apps/overlay-web/js/main.js`:

(a) Add to the `panel` object (or near it) references:

```javascript
const caption = document.getElementById("caption");
const sttNotice = document.getElementById("stt-notice");
document.getElementById("stt-toggle").addEventListener("click", () => {
  if (transcriber.available) { transcriber.stop(); sttNotice.style.display = "none"; caption.textContent = ""; }
});
```

(b) In `loop()`, inside the `if (transcriber.available)` block, mirror the window to the caption:

```javascript
      caption.textContent = t.text;
```

(c) Hide the notice when the recognizer never came online (end of `start()`):

```javascript
  if (!transcriber.available) sttNotice.style.display = "none";
```

- [ ] **Step 4: Commit**

```bash
git add apps/overlay-web/index.html apps/overlay-web/css/overlay.css apps/overlay-web/js/main.js
git commit -m "feat(overlay): live caption strip + honest cloud-STT notice/toggle

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 10: Enneagram — slot 7 lights on strongest linguistic cue

**Files:**
- Modify: `apps/overlay-web/js/enneagram.js:104-112`

The 9 slots reserve slot index 7 for `linguistic.verbal`, but the family emits several
`linguistic.*` cue ids. Generalize the slot's pull so it tracks the strongest active
`linguistic.*` cue (mirrors how audio slot 6 reacts to its family's representative cue).

- [ ] **Step 1: Implement** — in `apps/overlay-web/js/enneagram.js`, replace the per-point pull loop body (lines ~105-112) so slot 7 aggregates the linguistic family:

```javascript
    for (let i = 0; i < 9; i++) {
      const slotId = CUE_SLOTS[i];
      let hit;
      if (slotId === "linguistic.verbal") {
        // Aggregate: strongest active linguistic.* cue drives the verbal slot.
        hit = cues
          .filter((cu) => cu.cue_id.startsWith("linguistic."))
          .sort((a, b) => Math.abs(b.z) - Math.abs(a.z))[0];
      } else {
        hit = cues.find((cu) => cu.cue_id === slotId);
      }
      const targetPull = hit ? Math.max(0, Math.min(1, hit.z / 6)) : 0;
      this._pull[i] += (targetPull - this._pull[i]) * EASE;
      this._glow[i] += (targetPull - this._glow[i]) * (EASE * 1.5);
    }
```

- [ ] **Step 2: Commit**

```bash
git add apps/overlay-web/js/enneagram.js
git commit -m "feat(overlay): enneagram verbal slot tracks strongest linguistic cue

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 11: Honest doc correction (environment invariant)

**Files:**
- Modify: `planning/INDEX.md` (START HERE environment bullet, lines ~40-41)
- Modify: `docs/OVERLAY_README.md` (privacy/network section)

- [ ] **Step 1: Update INDEX** — in `planning/INDEX.md`, replace the "Only external network call" bullet with:

```markdown
- External network calls: (1) the browser fetches the **MediaPipe model from a CDN once**; (2)
  when the **Linguistic transcriber is enabled, Chrome's Web Speech API streams mic audio to
  Google for transcription** — the only path by which audio leaves the device. All cue detection /
  fusion / consensus / rPPG run **on-device**; raw **video** never leaves the browser. Web Speech is
  the pragmatic transcript source (zero install/login); a fully-local Whisper adapter (no external
  call) is the documented upgrade behind the `Transcriber` seam.
```

- [ ] **Step 2: Update OVERLAY_README** — find the privacy/"raw video + audio never leave" statement in `docs/OVERLAY_README.md` and amend it with the same honest caveat (mic audio leaves only while the transcriber is on; video never leaves; local-Whisper is the upgrade path). Keep wording consistent with the INDEX bullet.

- [ ] **Step 3: Commit**

```bash
git add planning/INDEX.md docs/OVERLAY_README.md
git commit -m "docs: honest external-network-call correction for Web Speech transcript

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 12: Manual browser verification (user-confirmed)

**Files:** none.

- [ ] **Step 1: Launch with a short baseline for a quick demo**

Run: `BLITZ_OVERLAY_BASELINE_SECONDS=20 python3 -m blitz_overlay`
Open `http://127.0.0.1:8000` in Chrome; allow camera + mic.

- [ ] **Step 2: Verify**
  - Caption strip shows live words while speaking; "Transcript via Chrome cloud STT" notice + disable toggle present.
  - Linguistic family row shows online/activity once you talk for ~20s (past calibration).
  - Enneagram slot 7 ("verbal") deforms when you use distanced/filler-heavy speech.
  - No console errors; the visual overlay still works if you deny the mic.
  - Disable toggle stops the caption and hides the notice.

- [ ] **Step 3: Report results to the user.** This is the in-browser confirmation gate before
  the whole `feat/audio-linguistic` branch (audio + linguistic) merges to `main` and pushes —
  which also still depends on the user's earlier **audio mic test**. Do not merge/push until the
  user confirms both audio and linguistic read correctly.
```
