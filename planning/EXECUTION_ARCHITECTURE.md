# Execution Architecture — Memory Budget & Browser-Hybrid

> Blitz Engine | Written June 15, 2026
> Covers two linked decisions: (A) how the engine runs inside a tight RAM budget, and (B) a browser-hybrid
> deployment that pushes heavy work to the GPU/WASM and the Claude API instead of local system RAM.
> Companion to `planning/READINESS.md` (esp. items 3, 8, 15, 17) and `planning/PROJECT_MAP.md`.

---

## Part A — Hardware reality & memory budget

### A.1 The constraint (primary dev machine)
- **Apple M1 MacBook Pro, 8 GB unified RAM, 8 cores (4P/4E). RAM is soldered — not upgradeable.**
- Unified memory: CPU + GPU + Neural Engine share the same 8 GB. **No separate GPU VRAM.**
- macOS + normal apps consume **~3–4 GB at idle** → real working budget is **~4–5 GB**, not 8.

This 8 GB ceiling is the tightest constraint in the project and drives the decisions below.

### A.2 Per-model memory footprint (loaded, CPU)
| Component | Approx RAM | Note |
|---|---|---|
| CrisperWhisper (1.55B) | **~6–7 GB** | ❌ alone exceeds the working budget |
| WhisperX / faster-whisper (int8) | ~2.5–3 GB | ✅ M1-friendly swap (also resolves Blocker 1) |
| pyannote.audio diarization (#10) | ~1–2 GB | torch + segmentation/embedding models |
| spaCy `en_core_web_trf` | ~0.5–1 GB | |
| NLI model (bart-large-mnli, cue 53) | ~1.6 GB | or offload to Claude API |
| OpenGraphAU (ResNet50 face AUs) | ~0.5–1 GB | |
| MMPose RTMPose (body, ONNX) | ~0.3–0.5 GB | light |
| vitallens rPPG | ~0.5 GB | |
| Python + torch baseline overhead | ~1–2 GB | always-on |
| Lexicons (NRCLex / VADER) | negligible | |

### A.3 The make-or-break rule: SEQUENTIAL execution
- **Concurrent (all models loaded): ~12–14 GB** → on 8 GB this thrashes to SSD swap, beachballs, may
  OOM-crash, and wears the SSD. **Not viable.**
- **Sequential (load → run → release → gc → next): peak RAM = the single largest model**, not the sum.
  With WhisperX that is ~3–4 GB peak → **fits in 8 GB, runs smooth.**

**ARCHITECTURE RULE (hard):** never hold two heavyweight models in memory at once. Stage the pipeline and
**cache intermediate artifacts to disk** (transcript JSON → cue events → fused output). Each stage loads
its model, emits its artifact, frees the model before the next stage begins.

### A.4 "Go smooth on M1 / 8 GB" recipe
1. **Sequential staging + disk caching** — the single most important design choice.
2. **Drop CrisperWhisper → WhisperX / faster-whisper (int8).** RAM decision, not just licensing → see §C.1.
3. **Offload heavy reasoning to the Claude API** (CBCA/RM scoring, contradiction) — network cost, ~0 local RAM.
   Optionally skip the local NLI model entirely and let Claude do contradiction (cue 53).
4. **Quantize (int8/fp16), `batch_size=1`, ONNX + CoreML execution provider** where possible (RTMPose) to
   use the Neural Engine.
5. **Process recorded clips offline, not real-time.** Full-stack real-time is not realistic on 8 GB.
6. **For full 66-cue runs, use the free 24 GB cloud box** (Oracle Cloud Always Free, 4 ARM cores / 24 GB).
   Dev + light/single-modality runs local; heavy all-modality runs in the cloud.

### A.5 Honest bottom line
- The M1 / 8 GB **can** run this for development and lean runs *if* execution is sequential and models are
  light. It **cannot** comfortably run the full concurrent stack. Design for that now, not later.

---

## Part B — Browser-hybrid execution (the "light but super complex" path)

### B.1 The idea
Modern browsers run heavy compute *lightly* by offloading to the GPU via **WebGL / WebGPU / WASM**, instead
of competing for the machine's 8 GB of system RAM (cf. WebGL/Three.js games like messenger.abeto.co). A large
share of the engine's per-frame work — facial landmarks, pose, gaze, rPPG signal extraction, and the UI —
can run **client-side**, leaving only heavy reasoning for the Claude API and a few heavy cues for the cloud.

This is not a detour: **the Chrome Extension adapter is already planned, and an extension is exactly
browser JS + WebGL.** The browser-hybrid simply brings that platform forward as the primary lean runtime.

### B.2 Modality placement — where each layer runs
| Layer / cue group | Runs on | Tech | Notes |
|---|---|---|---|
| Facial AUs | **Client (browser)** | **MediaPipe Face Landmarker** (WASM/WebGL) — 478 landmarks + **52 blendshape coefficients** | blendshapes map onto many catalog AUs in real time, free ⭐ |
| Head pose / gaze | **Client** | MediaPipe FaceMesh / Face Landmarker | real-time, light |
| Body / posture | **Client** | MediaPipe Pose | real-time |
| rPPG (HR / HRV) | **Client** | POS/CHROM signal processing in JS on the face ROI | pure DSP, no model needed |
| Live transcript | **Client** | whisper-tiny/base via `transformers.js` or whisper.cpp-WASM | tiny/base only on-device |
| VHS "signal monitor" UI | **Client** | **Three.js / WebGL** | matches the planned aesthetic ⭐ |
| Linguistic holistic (CBCA / RM / verifiability) | **Claude API** | API call from the browser | zero local compute |
| Contradiction (cue 53) | **Claude API** | API (replaces local NLI) | drop the 1.6 GB NLI model |
| Diarization (#10) | **Cloud / server** | pyannote.audio | poor in-browser → server-side |
| High-accuracy AUs (OpenGraphAU), CrisperWhisper, large NLI | **Cloud (24 GB box)** | Python | only when max accuracy is wanted |
| Fusion + calibration + verdict | **Either** | thin compute | runs anywhere; keep core platform-agnostic |

### B.3 Why this fits the project specifically
1. **Dodges the 8 GB wall** — GPU/WASM work does not draw on the Python RAM budget.
2. **Privacy = the strongest ethics win** — client-side cue extraction means **the raw video never leaves
   the device.** Only derived cue features (or nothing) go to the API. This directly satisfies the PII /
   data-handling goal (READINESS #15, `governance/DATA_HANDLING.md`) and the EU AI Act posture.
3. **It IS the Chrome Extension** — Phase 2's extension adapter becomes the primary lean runtime, not extra work.
4. **Engine stays platform-agnostic** — per the core philosophy (adapters on top of one engine). The
   browser is one adapter; CLI and cloud Python are others. Cue contracts (`CueEvent`) stay identical.

### B.4 Honest limits
- Browser-light models are **less accurate** than heavy server models. Acceptable: the honest target is
  70–75%, and the browser's strength (landmark-derived visual cues) holds up well.
- **WebGPU** is strong in Chrome, still maturing in Safari; MediaPipe's WASM/WebGL path works broadly today.
- **Heavy NLP and diarization are poor in-browser** — keep them on the Claude API / cloud.
- Client-side ML burns the user's battery/GPU; fine for short clips, not for hours of continuous capture.

### B.5 Recommended split (summary)
- **Browser-native:** visual cues (MediaPipe blendshapes→AUs, pose, gaze), rPPG, tiny-Whisper transcript,
  Three.js UI.
- **Claude API:** linguistic/CBCA/RM holistic + contradiction.
- **Cloud (free 24 GB ARM box):** diarization + the few heavy cues, for max-accuracy runs.

---

## Part C — Implications & decisions

### C.1 Resolve Blocker 1 → WhisperX (now a RAM decision, not only licensing)
CrisperWhisper (~6–7 GB) does not fit the 8 GB working budget. **Use WhisperX / faster-whisper (int8) for
the local/cloud Python path; use whisper-tiny/base (transformers.js / whisper.cpp-WASM) for the browser
path.** This also clears the CC-BY-NC licensing concern. Update `RESEARCH.md` Blocker 1 to "resolved: WhisperX".

### C.2 Deployment tiers
| Tier | Runtime | Use |
|---|---|---|
| **Lean (default)** | Browser (MediaPipe + Three.js + tiny-Whisper) + Claude API | private, real-time-ish, fits 8 GB |
| **Local Python** | M1, sequential staging, WhisperX | dev + single-modality/light runs |
| **Cloud full** | Oracle Always Free 24 GB ARM | full 66-cue, max-accuracy, diarization, heavy cues |

### C.3 Plan hooks (small follow-ups to fold in)
- READINESS #8/#17: record runtime tier + model versions per prediction (browser vs API vs cloud differ).
- READINESS #12: graceful degradation already covers "modality unavailable" — applies when a browser lacks
  WebGPU or a camera.
- `BLITZ_ENGINE_SPEC.md`: note the browser is a first-class adapter; cue contracts are runtime-independent.

> Net effect: "light but super complex" done right — GPU/WASM + API do the heavy lifting, the M1 stays
> within budget, and client-side extraction turns the privacy constraint into the product's biggest strength.
