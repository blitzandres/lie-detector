# Parallel Cue Verifier + Synchrony Bell (Design Spec)

> Date: 2026-06-16 · Branch: `feat/audio-linguistic` · Status: Approved
> A "put-them-all-together" layer on top of the existing cue engine. Adds the living-checklist
> experience + a synchrony-driven bell. The detection engine is NOT modified.

## Goal

Turn the overlay into a **live "Parallel Cue Verifier"**: a checklist where every cue is a row that
**lights up in real time**, and when **multiple cues across independent channels co-fire in the same
brief moment** (temporal *synchrony*), an **earned bell** rings. This makes the reasoning legible
("it tells you") and operationalizes the user's insight that cue *convergence in time* is a strong
deception signal — while staying honest.

## Hard constraint — engine is sacred

**Zero changes** to: `blitz_overlay/cues/*` (detectors), `blitz_overlay/weights.py` (science weights),
`core/fusion/bayesian_fusion.py` (fusion/two-gate), `core/calibration/*` (baseline). Every new piece
**reads** their existing outputs (per-cue z-scores, family votes, two-gate result). This automatically
honors the locked "science-driven weights, never learned / never let one cue dominate" rules.

In-scope to modify (these are the aggregation/presentation layer, not the detection engine):
`blitz_overlay/pipeline.py` (orchestration), `blitz_overlay/consensus.py` (payload builder),
`blitz_overlay/schemas.py` (additive fields), `blitz_overlay/logger.py` (additive bell log), and the
browser app.

## Honest-framing (locked)

- Synchrony is the **headline experience**; the existing Bayesian score + two-gate stays underneath as
  the **referee** that keeps the bell *earned* (prevents random-coincidence false alarms).
- Bell label: **"strong deception-pattern convergence."** Never "LIE." Statuses stay
  CALIBRATING→CLEAR→WATCH→FLAG.
- The sensitivity slider tunes the **decision operating point only** (K / lit-threshold / hold), never
  the science cue weights, and is transparently labeled **"↑ raises false alarms."**

## Components

### 1. SynchronyDetector — `blitz_overlay/synchrony.py` (new)

Pure aggregation over the per-cue z-scores the pipeline already computes each frame.

- Keeps a short **rolling lit-window** (`window_ms`, default 1000 ms): for each cue, the last time its
  directed/absolute z was ≥ `lit_z` (default 2.0). A cue counts as "recently lit" if seen within the
  window — this captures "the same moment" robustly across ~10–30 Hz frames (cues rarely peak on the
  exact same frame).
- Each frame returns a **convergence snapshot**:
  `{n_lit, n_families, lit_cue_ids: [...], families_lit: [...], peak_z}`.
- A **convergence burst** is defined as `n_lit >= K` (default 3) **and** `n_families >= 2`.
- Stateful, lives one session. No knowledge of detection internals — just `(cue_id, family, z)` tuples.

### 2. BellController — `blitz_overlay/bell.py` (new)

The bell has its **own honest earned-gate** that *reads* the engine's outputs but owns its operating
point (so the sensitivity slider can make it reachable without ever touching the engine):
- **Convergence gate:** a synchrony burst — `n_lit >= K` lit cues across `n_families >= 2` independent
  channels. The ≥2-families requirement is the honest guardrail (same spirit as the engine's two-gate
  gate-2): it prevents single-channel / random-coincidence alarms and can never be faked by the slider.
- **Risk gate:** the existing fused posterior (read-only from fusion) `>= risk_floor`. `risk_floor` is
  the slider-tunable operating point (default **0.65** = today's conservative point; max sensitivity
  lowers it toward ~0.45 with the explicit false-alarm caveat). This is the "score as referee."
- Rings only when **both** gates hold continuously for `hold_ms` (default 1500 ms) — debounced, not a
  1-frame flicker. (Calibrating ⇒ no lit cues ⇒ silent.)
- Emits `{ringing, just_rang, sustained_ms, label}`. `just_rang` is True for exactly one emission per
  episode (the browser plays the chime on that edge). Re-arms only after the condition drops.
- On each ring, returns a bell record `{ts, cue_ids, families, risk}` for the trust log.

The engine's own `two_gate.flag`/status is unchanged and still drives the red pulse; the bell is a
stricter, sustained, synchrony-gated annotation layered on top.

### 3. Payload extension — `blitz_overlay/schemas.py`

Add additive fields to `Consensus` (and a `CueRow` dataclass); detection unchanged:
- `cue_rows: list[CueRow]` — **every registered cue** with `{cue_id, family, region, label, z, lit,
  online}` so the checklist can render idle vs lit rows.
- `convergence: dict` — the SynchronyDetector snapshot.
- `bell: dict` — the BellController state.
`SCHEMA_VERSION` stays "1.0" (additive, backward-compatible — matches prior precedent).

### 4. Pipeline wiring — `blitz_overlay/pipeline.py`

- Already computes per-cue `z` in the liveness loop; collect `(cue_id, family, z)` for **all** measured
  detectors and an `online` flag, plus idle rows for unmeasured cues.
- Instantiate `SynchronyDetector` + `BellController`; feed them the lit set + the existing `two_gate`
  result each frame; pass results into `consensus.build(...)`.
- Read an optional `frame.config = {"sensitivity": 0..1}` (sent by the browser slider) and map it to
  `(K, lit_z, risk_floor, hold_ms)` — default sensitivity = today's conservative point
  (`K=3, lit_z=2.0, risk_floor=0.65`); max sensitivity → `K=2, lit_z=1.5, risk_floor≈0.45`. Never maps
  to the science weights or the two-family requirement. No new WS message type; reuses the frame channel.

### 5. Consensus builder — `blitz_overlay/consensus.py`

Assemble `cue_rows` (labels/regions from the detector registry), attach `convergence` + `bell` to the
`Consensus`. The status/risk/two-gate logic is unchanged; the bell does **not** alter status — it is a
sustained-FLAG annotation.

### 6. Trust log — `blitz_overlay/logger.py`

On `bell.just_rang`, append the bell record to the prediction log (additive field). Honest audit slot.

### 7. Browser

- `apps/overlay-web/js/cue-verifier.js` (new) — renders the **live checklist**: a row per cue grouped
  by channel, lighting up with intensity = z; a convergence counter ("4 cues · 3 channels firing");
  the honest verdict readout (climbs with risk, visible at WATCH).
- `apps/overlay-web/js/bell.js` (new) — **WebAudio synth chime** (no asset files); plays on
  `bell.just_rang`. A browser-side **trust meter** = bell-frequency over a rolling window.
- `apps/overlay-web/js/main.js` — wire both; render a **sensitivity slider** that attaches
  `frame.config.sensitivity` each loop.
- `apps/overlay-web/index.html` + `css/overlay.css` — checklist panel, verdict line, slider, trust
  meter, bell flash. The enneagram stays; the checklist is the new primary "reasoning" view.

## Data flow

```
existing engine (UNTOUCHED): cues -> per-cue z -> fusion(posterior) -> two_gate
        │  (pipeline reads these, read-only)
        ▼
SynchronyDetector (lit-window, convergence burst: >=K lit, >=2 families) ─┐
                                                                          ├─► BellController
fused posterior ──────────────────────────────────────────────────────────┘   (burst AND
        │                                                                        posterior>=risk_floor,
        │                                                                        held 1.5s)
        ▼                                                                            │
ConsensusBuilder → Consensus{ ..., cue_rows, convergence, bell }  ◄──────────────────┘
        │  ws (additive fields)
        ▼
browser: Live Cue Verifier checklist + verdict + WebAudio bell + trust meter + sensitivity slider
```

## Honest degradation

- Calibrating → no lit cues permitted (z stays 0 while `is_calibrating`); convergence = 0; bell silent.
- Fewer than 2 families ever lit → bell unreachable; verdict honestly caps at WATCH (existing behavior).
- Sensitivity slider at max still requires ≥2 independent families to co-fire — it lowers the bar, it
  never fabricates a second channel.

## Testing (TDD, Python)

- **SynchronyDetector:** burst detected only at `n_lit>=K` AND `n_families>=2`; rolling-window expiry
  drops stale lit cues; same-family-only co-firing does NOT count as ≥2 families.
- **BellController:** silent until (burst AND posterior≥risk_floor) held `hold_ms`; `just_rang` fires
  exactly once per episode; re-arms after drop; respects sensitivity-mapped params; never rings with
  <2 families lit or posterior<risk_floor (even at max sensitivity).
- **Consensus:** `cue_rows` lists every registered cue with correct lit/online; `convergence` + `bell`
  present and correctly shaped.
- **Pipeline:** `frame.config.sensitivity` maps to params and changes burst behavior; a deterministic
  replay where ≥2 families co-fire for ≥1.5s produces `bell.just_rang` exactly once and logs it; a
  single-family replay never rings.
- `python3 -m pytest -q` and `python3 -m ruff check .` stay green.

## Defaults (all slider-tunable)

`lit_z = 2.0` · `K = 3` lit cues · `≥2` families · `window_ms = 1000` · `hold_ms = 1500` ·
default sensitivity = conservative.

## Out of scope (Phase 2 — cue accuracy)

Cross-modal audio×movement *coherence* meta-cue (needs literature WebSearch), extraction-sensitivity
improvements, Body/Posture family. The detection engine changes live there, not here.
