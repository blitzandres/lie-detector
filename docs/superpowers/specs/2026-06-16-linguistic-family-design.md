# Linguistic Family — Live Consensus Overlay (Design Spec)

> Date: 2026-06-16 · Branch: `feat/audio-linguistic` · Status: Approved
> Supersedes nothing; extends the Stage-1 overlay with a 4th consensus voter.

## Goal

Add the **Linguistic family** as the 4th live consensus voter (Visual · Physio · Audio ·
**Linguistic**) in the Live Consensus Overlay. A live transcript drives a small set of
person-relative lexicon cues reused from `modalities/linguistic/analyzer.py`, wired through the
existing rolling baseline + family fusion + two-gate, and surfaced on enneagram slot 7 plus an
on-screen caption strip.

With a 4th wired family, an **earned two-gate FLAG becomes reachable via any 2 of the 4 families**
(no longer forced to require rPPG).

## Locked decisions (from brainstorming)

1. **Transcript source = Chrome Web Speech API now, behind a `Transcriber` seam.** Zero install,
   zero login, zero API key, lowest memory/latency — Chrome offloads STT to Google. A fully-local
   `LocalWhisperTranscriber` (no external call) is the documented upgrade path behind the same
   interface, to be built only when we choose to spend the RAM.
2. **Lexicon cues run in the Python engine**, not the browser. The browser sends rolling transcript
   **text**; Python reuses the validated `analyzer.py` word-lists as `CueDetector`s. Single source
   of truth; science weights stay in `weights.py`. (User does not require text to stay off the wire.)
3. **On-screen live caption strip** under the video — honest about what is being read; makes the
   family feel alive.

## Honest-framing / environment correction (REQUIRED scope)

Web Speech breaks the documented invariant in `planning/INDEX.md` ("Only external network call:
MediaPipe CDN … Raw video + audio never leave the browser"). Because honest framing is a locked
constraint, this spec **includes updating that environment note** (INDEX START HERE + `docs/OVERLAY_README.md`)
to state accurately:

> External network calls: (1) MediaPipe model from CDN once; (2) **when the Linguistic transcriber is
> enabled, Chrome's Web Speech API streams mic audio to Google for transcription** — the only path by
> which audio leaves the device. All cue detection / fusion / consensus / rPPG run on-device; raw
> **video** never leaves the browser. A fully-local Whisper adapter (no external call) is the
> documented upgrade behind the `Transcriber` seam.

The browser UI shows a clear, honest notice while the transcriber is active (e.g. "Transcript via
Chrome cloud STT") with a toggle to disable it.

## Architecture / data flow

```
Browser:  Web Speech API → rolling word window (~last 12s / 40 words)
          → frame.transcript {text, seq}     [attached only when seq advances; ~2 Hz]
            │ localhost WebSocket (tiny: a few words of text)
Python:   linguistic CueDetectors read frame.transcript.text
          → analyzer.py lexicons → rolling baseline (per-utterance) → family fusion → consensus
          → enneagram slot 7 + caption strip
```

### Browser — `apps/overlay-web/js/transcriber.js` (new)

- `Transcriber` interface (the seam). One adapter now: `WebSpeechTranscriber` wrapping
  `webkitSpeechRecognition` (`continuous = true`, `interimResults = true`).
- Maintains a rolling window of recent words (cap ~40 words / ~12 s). `latest() → {text, seq}`.
  `seq` increments **only when the window text changes**.
- Graceful: unsupported browser (no `webkitSpeechRecognition`) or denied permission →
  `available = false`, no throw, family simply absent.
- Throttle outbound updates to ≤ ~2 Hz / on meaningful change.

### Browser — `main.js`

- Construct + `start()` the transcriber inside the existing mic try/catch (a failed transcriber
  must never kill the visual overlay).
- In `loop()`, attach `frame.transcript = transcriber.latest()` **only on frames where `seq`
  advanced** (mirrors how `frame.audio` is attached only when mic is live; nulls omitted so the
  Python side sees absence cleanly).
- Render the caption strip + the honest cloud-STT notice/toggle (small DOM, `overlay.css`).

### Engine — `blitz_overlay/cues/linguistic.py` (new)

Seven `CueDetector` subclasses, each reusing the `analyzer.py` word-lists, computed over the
current transcript window:

| cue_id | direction | effect_size_d | tier |
|---|---|---|---|
| `linguistic.sensory_detail_poverty` | +1 | 0.29 | 2 |
| `linguistic.pronoun_avoidance` | +1 | 0.27 | 2 |
| `linguistic.distancing_language` | +1 | 0.24 | 2 |
| `linguistic.filler_ratio` | +1 | 0.23 | 3 |
| `linguistic.qualifier_overload` | +1 | 0.21 | 3 |
| `linguistic.negative_emotion_density` | +1 | 0.18 | 3 |
| `linguistic.lexical_diversity_drop` | +1 | 0.16 | 3 |

- **Dropped:** `linguistic.response_delay_ms` — needs question→answer turn timing; the live
  overlay is free-form with no `question_id`. (Re-add if a question track is ever introduced.)
- `measure(frame)` returns the feature over `frame.transcript.text`, or **`None`** when: no
  transcript block, the window is unchanged since last processed (`seq` not advanced), or fewer
  than `MIN_WORDS` (~5) words. → Linguistic samples therefore feed the rolling baseline
  **per-utterance, not per 30 Hz frame**, so the baseline MAD is not poisoned by duplicate static
  windows. (Key correctness point.)
- Person-relative via the existing rolling baseline; `quality` scales with word count (short
  window → low quality), reusing the analyzer's confidence intuition.
- Export `LINGUISTIC_DETECTORS`.

The lexicon constants live in `analyzer.py` and are imported, not copied. If a shared feature
helper is cleaner than re-deriving counts, factor a small pure function in `analyzer.py` and reuse
it from both places (no logic duplication).

### Wiring (mirrors the audio family)

- `weights.py`: add the 7 linguistic weights + citations (from `analyzer.py` cue_specs and
  `modalities/linguistic/RESEARCH.md`); these are science-driven, never learned.
- `consensus.py`: add `"linguistic"` to `WIRED_FAMILIES`. It is already in `PANEL_FAMILIES`.
- `pipeline.py`: append `[cls() for cls in LINGUISTIC_DETECTORS]` to the session detectors.
- `schemas.py`: add `transcript: dict | None` to `FeatureFrame` (+ `from_dict`), same nullable
  pattern as `audio`. Additive/backward-compatible → keep `SCHEMA_VERSION` unchanged (matches how
  the `audio` field was added); no `schema.js` version change needed.
- `enneagram.js`: slot 7 (`linguistic.*`) lights on the **strongest active `linguistic.*` cue**
  (small generalization so the family's several cues all map to the one reserved slot). Slot 8
  (`cbca.content`) stays a placeholder. Caption strip wired here or in `overlay-renderer.js`.

### Honest degradation

- No speech during calibration → linguistic baseline never populates → family absent → system
  honestly caps at WATCH (same as audio/rPPG today).
- Web Speech unsupported (non-Chrome) or denied → family absent, no error.

## Testing (TDD)

- Each detector: correct feature value from a known transcript window; `None` on absent / unchanged
  (`seq` not advanced) / too-short (< MIN_WORDS) windows.
- Baseline integration: a deviating window fires a `CueEvent` with the right modality/region;
  repeated identical windows do **not** add baseline samples.
- Schema round-trip: `FeatureFrame.from_dict` carries `transcript`; absent → `None`.
- Consensus: `linguistic` reported `wired = True`; a deterministic replay reaches a **Visual +
  Linguistic** two-gate FLAG (proving FLAG no longer requires rPPG).
- `python3 -m ruff check .` and `python3 -m pytest -q` stay green.

## Out of scope

- Local Whisper adapter (seam only).
- CBCA/RM content cues (slot 8 placeholder).
- Question/answer turn tracking and `response_delay_ms`.
- Cross-modal coherence meta-cue, bell/trust-log, Body/Posture family (later queue items).
