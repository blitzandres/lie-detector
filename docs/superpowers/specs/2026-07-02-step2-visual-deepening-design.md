# Step 2 — Visual Deepening: enneagram rewire, live temporal cues, offline research tier

Date: 2026-07-02 · Status: approved scope, spec for review

Step 2 of the "finish the project" plan (step 1 = merge + cleanup, done; the README
architecture organigram was also redrawn to match the built engine). Three parts, built in
order: **2a** enneagram canvas rewire + redesign, **2b** eight new live MediaPipe-derived
cues (Visual 21 → 29), **2c** offline research-tier visual analyzer for recorded video.

Invariants that apply to all three parts:

- Honest framing everywhere: statuses/labels never say "LIE"; micro-expression cues carry
  a low-base-rate caveat into governance docs.
- Science weights are fixed, citation-traceable, and never move at runtime.
- The live path stays browser-extraction-only: raw frames never reach Python.
- ruff + pytest stay green; every new detector gets TDD tests.

## 2a — Enneagram: the family organigram

**Problem.** `apps/overlay-web/js/enneagram.js` is wired to the Stage-1 world: its 9 points
are hardcoded to 5 early visual cues + `physio.heart_rate` + `audio.tremor` + a linguistic
aggregate + `cbca.content` (a family that does not exist in the payload). 16 of 21 visual
cues and 2 of 3 audio cues can never move it. Deformation is multiplied by `risk`, which the
conservative engine keeps near 0, so the figure barely moves. Inner lines light only on
family votes (z > 2 spikes) and ignore the continuous `online`/`activity` fields the engine
already publishes.

**Design.** The enneagram becomes the **family-level** view (the Cue Polygon is the per-cue
view). The 9 points get permanent meanings:

| Point | Meaning | Driven by |
|---|---|---|
| 1 | Visual | family activity + max \|z\| of visual.* active cues |
| 2 | Audio | same, audio.* |
| 3 | Linguistic | same, linguistic.* |
| 4 | Physio | same, physio.* |
| 5 | Content | last `turn_result` content risk, fading over ~30 s |
| 6 | Body | reserved — dim "offline" until step 3 builds the family |
| 7 | Synchrony | `convergence` (n_families-led burst level; flares on burst) |
| 8 | Consensus | fused posterior (risk) |
| 9 | Trust | bell frequency (inverse trust from BellController) |

Behavior:

- **Pull (deformation)** per family point = blend of continuous `activity` (base) and the
  strongest active cue z in that family (spike). **Decoupled from risk** — risk instead
  drives fill alpha and color temperature of the whole figure.
- **Idle life**: points breathe subtly (small sinusoidal radius wobble scaled by `online`)
  so the figure visibly lives even at CLEAR/quiet.
- **3-6-9 triangle** lights when the two-gate is satisfied (≥2 families agree); **hexad
  lines** brighten with pairwise family co-activity (product of endpoint activities).
- **Design polish**: per-family accent colors (consistent with the Cue Polygon's family
  arcs), gradient stroke on the outer ring, labels on all 9 points (abbrev + hover-style
  emphasis on the strongest), status color still governs the overall tone.
- `main.js` additionally forwards `turn_result` messages to the enneagram (content point).

No engine changes: everything needed (`families[].online/activity/vote`, `active_cues`,
`convergence`, `bell`, `turn_result`) is already in the payload.

## 2b — Eight new live cues (Visual 21 → 29)

All derived from the existing blendshape/landmark/iris stream — no new models, no new RAM.
Each gets: a browser-side feature (only where a new scalar is needed), a `CueDetector` in
`blitz_overlay/cues/visual.py`, a science weight with citation in `weights.py`, and tests.

| Cue | Signal | Source |
|---|---|---|
| `visual.duchenne_absence` | smile without eye involvement (AU12 high, AU6 low) — social/masked smile | mouthSmile vs cheekSquint blendshapes |
| `visual.stress_brow` | AU1+AU2+AU4 co-occurrence (fear/stress brow) | browInnerUp + browOuterUp + browDown |
| `visual.head_velocity` | head rotation speed (nods/shakes/tilts) | head pose deltas (split from head_movement) |
| `visual.head_acceleration` | sudden head-movement onsets | second derivative of head pose |
| `visual.face_asymmetry` | multi-region L/R deviation (eye + brow + mouth), beyond smile | paired L/R blendshapes |
| `visual.facial_rigidity` | expressivity collapse — low blendshape trajectory variance over the window (freezing) | rolling variance of blendshape vector |
| `visual.microexpression_burst` | short-window blendshape intensity velocity + variance spikes (honest label: proxy) | frame-to-frame blendshape deltas |
| `visual.blink_duration` | long-blink duration + rebound timing (complements rate + eye_blocking) | eyeBlink blendshape time series |

Question-timed gaze aversion (gaze pattern relative to Q&A turn boundaries) is folded into
the **content fusion** layer instead of a new detector: `fuse_turn` already pulls the cue
timeline for `[t0, t1]`; it gains per-window gaze-aversion weighting. This avoids a live cue
that is meaningless outside Q&A mode.

Where a detector needs a windowed statistic (rigidity, microexpression, blink duration), the
computation lives Python-side off the streamed per-frame scalars — the browser only ever
adds cheap raw scalars to the FeatureFrame.

Governance: `docs/CUE_CATALOG.md` / governance notes get a paragraph stating
micro-expressions have low base rates and modest real-world effect sizes; weights set
accordingly (low d).

## 2c — Offline research tier: `modalities/visual/analyzer.py`

For **recorded video files only** (research mode). These models need raw frames and are too
heavy for the live path. Hard rule on the M1/8GB dev machine: **one heavyweight model loaded
at a time** — load → run over the whole clip → release → next model; artifacts cached to
disk between passes.

- **Py-Feat Detector v2** (primary): one multi-task pass → 20 AUs with intensity, emotions,
  valence/arousal, gaze, head pose, MediaPipe-compatible landmarks.
- **OpenGraphAU** (optional ensemble): complementary AU detector; when enabled, AU cues use
  agreement-weighted values (robustness), behind a flag — not required for the default run.
- **LibreFace**: documented fallback backend behind the same AU interface (not integrated by
  default; seam only).
- **OpenCV optical flow (Farneback)**: dense flow on face crops → temporal dynamics +
  micro-expression spotting features (onset velocity), cheap CPU pass.

Output: typed `CueEvent`s (`cue_id`, `value`, `effect_size`, `reliability_weight`,
timestamps) feeding the **same** MAD baseline + log-odds fusion core, with instantaneous and
3-phase window features per the original spec. Entry point: `blitz analyze-video <clip>`
(extends the existing CLI), baseline clip required, abstain honestly on quality failures.

Dependencies (`py-feat`, `opencv-python`, torch) go in an optional extra
(`pip install -e ".[research]"`) so the lean live overlay install stays light. Tests use
small synthetic/fixture inputs and a stubbed AU backend — CI never downloads model weights.

## Non-goals

- Body family (step 3), REST API, thermal.
- No live-path use of Py-Feat/OpenGraphAU/optical flow.
- No learning/training of weights from any output.

## Testing

- 2a: visual behavior is browser-side; add a payload-shape regression test if any Python
  field is touched (none expected). Manual in-browser verification.
- 2b: TDD per detector (baseline-neutral, spike, abstain-on-missing-input cases) +
  weights-table completeness test (every cue_id has a weight + citation).
- 2c: analyzer unit tests with stubbed backends; an integration test on a tiny fixture clip
  gated behind the `research` extra being installed.
