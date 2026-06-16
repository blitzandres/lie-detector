# Design Spec — Live Consensus Overlay ("AI-Vision")

> Blitz Engine | Brainstormed & written June 16, 2026
> Status: DESIGN APPROVED (sections 1–3) — ready for implementation planning.
> Companions: `planning/READINESS.md`, `planning/EXECUTION_ARCHITECTURE.md`,
> `modalities/linguistic/RESEARCH.md`, `docs/CUE_CATALOG.md`.

---

## 1. What this is (one paragraph)

A real-time "AI-vision" overlay for the Blitz Engine, modeled on sports tactical-analysis tools
(LiveTag.Pro / telestrator + tagged timeline). MediaPipe maps a speaker's face/body as moving points
(478 landmarks + 52 blendshapes + head pose + body pose) — the "players." The engine detects deception
**cues** anchored to specific landmark regions, tags them with timestamps (`CueEvent`s), and a **live
consensus mechanism** (the engine's Bayesian fusion + two-gate convergence) aggregates all cues in real
time. The overlay draws a telestrator circle on the active region and shows a **consensus panel** of cue
"voters." Raw video stays on the device; only tiny feature vectors reach the engine.

## 1b. Project staging (explicit)

- **Stage 1 (this spec) = webcam only.** Lean walking skeleton: prove capture → cue detection → consensus
  → overlay on one source, on the M1/8 GB.
- **Ultimate goal = the full suite:** all modality families (visual + audio + linguistic + physiological),
  all capture sources (webcam → in-browser screen-region → native draw-anywhere), and eventually the full
  66-cue catalog with the eval/accuracy track. Stage 1 is deliberately a slice of that end state, built so
  every later stage plugs into the same engine + adapter interfaces with no rework.

## 2. Goals / non-goals

**Goals**
- Prove the *entire* pipeline end-to-end: capture → cue detection → consensus → overlay (the walking
  skeleton, READINESS #5).
- Real-time on the 8 GB M1 (browser does GPU vision; Python engine does the thinking).
- Honest framing: risk + cue activity + consensus, never a binary "LIE."
- Reuse the real Python core (`core/`), not a throwaway — the overlay is an adapter.

**Non-goals (v1)**
- Full 66 cues. Live audio/linguistic cues. Accuracy/eval-harness validation (separate track).
- Chrome-extension packaging. Native draw-anywhere capture (that's a later build).

## 3. Architecture (Approach 2 — browser capture + Python engine)

```
BROWSER (client)
  capture (webcam | screen-region)  →  MediaPipe (landmarks · blendshapes · head pose · body pose)
  rPPG ROI sample (forehead/cheek color)
  → feature vector {landmarks, blendshapes, pose, ts}   (few KB, NO raw video leaves the page)
        │ WebSocket (localhost) ▲ consensus payload
PYTHON ENGINE (local — the real Blitz core)
  cue detectors → CueEvent(region, ts)
  rolling baseline (robust-Z / median-MAD) → family-grouped fusion → two-gate convergence
  consensus-builder {per-family vote, freshness, risk, status} → prediction log
        ⇧ back to browser → overlay-renderer (telestrator + consensus panel)
```

**Why this split:** browser is best at GPU vision + rendering; the Python engine stays the single source of
truth for cue detection + consensus (matches the platform-agnostic "engine + adapters" philosophy). Only
feature vectors cross the wire → privacy (READINESS #15) + low bandwidth.

## 4. Components (each isolated + unit-testable)

**browser/**
- `capture` — source adapter: `WebcamSource` (build 1), `ScreenRegionSource` (build 2). Emits frames.
- `mediapipe-extractor` — frame → feature vector (Face Landmarker w/ blendshapes + head pose; Pose).
- `rppg-sampler` — samples forehead/cheek ROI color from landmarks (optional v1).
- `ws-client` — sends feature vectors, receives consensus.
- `overlay-renderer` — telestrator circles on active regions; consensus panel (collapsed ↔ expanded);
  risk meter; earned red pulse on FLAG.

**engine/** (Python, builds on existing `core/`)
- `ws-server` — receives feature frames, streams consensus.
- `cue-detectors/` — one detector per cue; feature vector → `CueEvent | None`.
- `baseline` — rolling robust-Z (reuse/extend `core/calibration/baseline.py`, mode="rolling", READINESS #7).
- `fusion` — family-grouped log-odds + two-gate (reuse/extend `core/fusion/bayesian_fusion.py`;
  independence/decorrelation fix, READINESS #11).
- `consensus-builder` — assembles per-family votes + freshness + combined risk + status.
- `prediction-logger` — append-only audit log (READINESS #8): inputs, cue contributions, posterior, status.

**shared/**
- `feature-frame` schema and `consensus` schema — the WebSocket contract (versioned).

## 5. Cue → landmark mapping (the "map players, but cues")

| Cue | Mapped region ("player") | Detected from | Family |
|---|---|---|---|
| Blink rate / eye widen | eye landmarks + `eyeBlink`,`eyeWide` | blendshape rate vs baseline | Visual |
| Gaze aversion | iris landmarks + head pose | gaze direction vs baseline | Visual |
| Brow flash (AU1/2/4) | `browInnerUp`,`browDown` | blendshape spikes | Visual |
| Lip press / pucker | `mouthPress`,`mouthPucker` | blendshape over time | Visual |
| Facial asymmetry (micro-expr) | left vs right blendshapes | L/R delta | Visual |
| **Jaw tension** | jaw landmark distances | distance ratio over time — **resolves Blocker 2 (AU28)** | Visual |
| Head freeze / shifts | head-pose matrix | motion variance | Visual |
| Posture / self-touch | Pose: hand→face proximity | landmark distance | Visual/Body |
| Heart rate (rPPG) | forehead/cheek ROI | color signal → BPM | Physio |

Each detection = a `CueEvent(region, ts, value, robust_z, confidence)` → the tag + the telestrator anchor.

## 6. Consensus mechanism (live)

- Each **cue family** (Visual, Audio, Linguistic, Physio) is a "voter." A family votes "flag" when its
  cues' fused contribution exceeds its threshold.
- **Independence (READINESS #11):** only *independent families* count toward consensus; correlated cues
  within a family are decorrelated/capped so repeated signals don't fake agreement.
- **Two-gate FLAG** = (≥2 independent families agree) AND (combined risk ≥ gate, e.g. 0.65).
- **Freshness:** each voter carries fresh/stale state (Linguistic/Physio lag or aren't wired in v1) →
  honestly excluded from the live count when stale/absent.
- Output statuses: `CALIBRATING → CLEAR → WATCH → FLAG`.

## 7. Framing & UX (honest, governance-compliant)

- **Never binary "LIE."** Statuses only; red pulse fires *only* on a two-gate FLAG (drama earned by
  agreement). Confidence + which families always visible.
- **Hybrid display** (matches the planned collapsed/expanded widget):
  - *Collapsed (default):* master consensus glow + risk meter — clean, cinematic.
  - *Expanded (tap/hover):* full voter panel — families, "N of M agree," per-cue freshness.

## 8. Error handling / graceful degradation (READINESS #12)

- No camera / permission denied → clear message, no fake data.
- No face detected → "no subject," cues pause (do not emit).
- Low landmark confidence / low light → cues low-confidence, uncertainty widened.
- WebSocket drop → "engine offline" + auto-reconnect.
- Baseline filling (first 90–180 s) → status `CALIBRATING`, no flags permitted.
- < 2 wired families → FLAG unreachable; capped at WATCH.
- Linguistic/Physio not wired → shown "—", excluded from family count honestly.

## 9. Capture source roadmap (swappable adapter)

- **Build 1 — Webcam** (`WebcamSource`): the walking skeleton.
- **Build 2 — In-browser screen region** (`ScreenRegionSource`): `getDisplayMedia` + mouse-drawn box →
  canvas crop → same pipeline. Pure browser; analyzes any tab/window (e.g., a podcast). Caveat: browser
  requires the user to pick the share-source each session.
- **Build 3 — Native draw-anywhere** (later): Electron wrapper or a small Python helper providing a global
  hotkey + draw-a-square-over-any-app + screen grab, feeding the same feature pipeline. Delivers the
  "press a key, draw a square anywhere" vision the browser alone cannot.
- The `capture` source-adapter interface means Builds 2 & 3 plug in with **zero engine changes**.

## 10. Testing

- Unit test per cue detector: synthetic blendshape/landmark input → expected `CueEvent`.
- Fusion + baseline math tests (READINESS #6).
- Recorded feature-stream replay harness → deterministic consensus output (reproducibility, READINESS #17).
- Prediction logging verified (READINESS #8).

## 11. First-build scope (definition of done for v1)

Wire for real: ~5 visual cues (blink, gaze aversion, brow flash, lip-press, **jaw tension**) + rPPG heart
rate. **rPPG is recommended (not optional) for v1** because it provides the *second independent family* —
without it the two-gate can never be met and the overlay can only ever reach WATCH, so a FLAG would not be
demonstrable. Rolling baseline + family-grouped fusion + two-gate + consensus payload + prediction log.
Browser overlay: telestrator + collapsed/expanded consensus panel + risk meter + earned red pulse.
Audio/Linguistic voters shown "not wired." Webcam source only. Runs real-time on M1/8 GB.

**Done when:** a live webcam session shows cues detected from real facial measurements, a live consensus
panel that only FLAGs under the two-gate rule, and a deterministic replay test passes.
