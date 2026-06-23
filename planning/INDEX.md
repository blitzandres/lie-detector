# Blitz Engine — Complete Project Documentation
> Consolidated April 1, 2026 | Canonical planning set tracked in the repo

---

## 🚦 START HERE — current focus (updated June 17, 2026)

**What this is:** the **Live Consensus Overlay** ("AI-vision") — a real-time, sports-analysis-style
overlay where MediaPipe maps the face/body, the Python engine detects deception **cues** and aggregates
them via a live **consensus mechanism** (Bayesian fusion + two-gate), and the browser draws a telestrator
+ consensus panel on top of the video. Raw video never leaves the device. Run it: `blitz-overlay`.

**⭐ STRATEGIC PIVOT (June 19, 2026) — CONTENT-FIRST, Q&A:** the *meaning of speech* is now the
**primary** layer; behavioral cues become the **secondary, time-aligned confirmation**. A local LLM
(**Ollama**, behind a swappable `ContentJudge` seam) reads each **answer in a Q&A interview** and
scores **content-pattern** deception markers (consistency · Reality-Monitoring richness · verifiability
· evasion) — NOT factual truth-checking (locked honest boundary; still never "LIE"). Two engines run in
parallel, loosely coupled: the fast **cue engine** keeps the real-time rhythm + a timestamped timeline;
the slow **content engine** judges each answer then *pulls the cue activity for that answer's time
window* and fuses (content-primary, cue-confirm). Plus a **calibration reading phase** + **True/False
dev scripts**. Full design: `docs/superpowers/specs/2026-06-19-content-first-qa-architecture-design.md`;
plan `…/plans/2026-06-19-content-engine-phase1.md`. **✅ BUILT (June 19):** `blitz_overlay/content/`
(`ContentJudge` seam · deterministic `StubContentJudge` · `OllamaContentJudge` · `TimelineBuffer` ·
content-primary `fuse_turn` · `session.judge_turn` · WS `turn` route) + browser Q&A panel + True/False/read
dev scripts. **Ollama installed** (`llama3.2:3b`, ~2 GB) and verified live (TRUE script risk 0.10 vs FALSE
0.40). Env-gated: server uses `OllamaContentJudge` only when `BLITZ_OVERLAY_CONTENT=ollama`; tests always
use the stub (never hit a live LLM). Degrades gracefully to the cue engine when Ollama is down.

**STATUS (updated June 19, 2026):**
- ✅ **DONE & on `main` (GitHub):** Stage 1 walking skeleton (Visual + Physio families, 5 visual cues +
  rPPG, rolling baseline, family fusion + two-gate, consensus, prediction log, FastAPI one-command server,
  browser telestrator). Plus the **enneagram viz**, **right-hand dashboard layout**, and **live family
  activity bars**. 80 tests.
- 🔄 **Built, on branch `feat/audio-linguistic` (committed + pushed as a BRANCH; NOT merged to `main` —
  awaiting user's in-browser confirmation of audio + linguistic + verifier):**
  - **Audio family** (browser Web Audio: pitch/pause/tremor → `cues/audio.py`, voter + enneagram slot 6).
  - **Linguistic family** (4th voter): browser **Web Speech** transcript → `cues/linguistic.py` reusing
    `modalities/linguistic/analyzer.py` lexicons (7 cues), per-utterance seq de-dup, live caption strip +
    honest cloud-STT notice. Spec/plan: `docs/superpowers/{specs,plans}/2026-06-16-linguistic-family*`.
  - **Parallel Cue Verifier + synchrony Bell** (engine UNTOUCHED — pure aggregation layer):
    `blitz_overlay/synchrony.py` (co-firing burst: ≥K lit cues across ≥2 families in a ~1s window) +
    `blitz_overlay/bell.py` (earned debounced bell on burst + posterior≥risk_floor held ~1.5s, honest
    label "strong deception-pattern convergence") + `cue_rows`/`convergence`/`bell` payload. Browser:
    live **cue checklist**, **Cue Mixer** scrolling multi-track timeline (`cue-timeline.js`, synchrony =
    vertical co-firing column), WebAudio chime, **trust meter**, **sensitivity slider** (moves operating
    point K/lit_z/risk_floor only — never science weights). Spec/plan: `…/2026-06-16-parallel-cue-verifier*`.
  - **Facial cue empowerment** (BUILT): +6 MediaPipe cues (gaze_fixation #56, pupil_dilation #7/#55,
    eye_blocking #13, eye_widen, nose_wrinkle #4, asymmetric_smile #5) via browser iris radius + unused
    blendshapes → **Visual 5→11**. Plan `…/2026-06-17-facial-cue-empowerment.md`.
  - **Hard-gated active calibration** (BUILT, option 2): `calibration_status.py` — calibration won't
    complete until every *producing* cue has a base (≥8 samples); zero-signal cues don't block; max-timeout
    escape. Browser calibration card with per-channel checklist + guidance.
  - **rPPG honesty fix** (BUILT): `estimate_bpm` now **abstains unless a real pulse dominates** (peak-SNR
    gate) instead of reporting noise as a heart rate; UI relabeled **rPPG·cam** (camera estimate, not a
    sensor). Cue Mixer lanes fixed-height + scroll (no overlap).
  - **Content engine (CONTENT-FIRST Q&A)** (BUILT June 19 — see pivot block above).
  - **Cue Polygon viz** (BUILT June 19): the linear Cue Mixer was replaced by a radial **N-sided polygon**
    (`cue-polygon.js`) — each cue is a vertex (grouped by family into arcs), lit cues shoot a light to the
    centre, the centre brightens with `convergence.n_lit` (synchrony) and flares + rings on an earned burst.
    Static, scales toward ~300 cues. Vertices show **abbreviations (V1/A2/L3…)** with the full cue id on
    **hover**. The **enneagram stays** (family-level viz). `cue-timeline.js` retired (unwired).
  - **Tier-1 cue expansion** (BUILT June 19): +10 browser-native visual cues from unused blendshapes +
    head pose (head_movement, eye_squint, mouth_stretch/frown/shrug, jaw_shift/drop, lip_roll,
    brow_outer_raise, contempt_asymmetry) → **Visual 11→21**. Plan `…/2026-06-19-tier1-cue-expansion.md`.
  - **Cue views simplified** (June 20): text checklist RETIRED — **Polygon + Enneagram only**. Convergence
    now **leads with CHANNELS** (independent families) not raw cue count, and the **polygon centre
    brightness is channel-driven** (80% families / 20% count) — so many correlated face cues can't oversell.
  - **rPPG skin-aware** (BUILT June 20): browser averages **only skin-toned pixels** (YCbCr per-pixel mask,
    no new model) + reports `skin_fraction`; the physio cue's **quality scales with skin_fraction**. On top
    of the earlier **peak-SNR abstain gate** (no fake BPM from noise) and the **rPPG·cam** honest label.
  - **159 tests, ruff+pytest green.** 32 cues live: 21 visual · 3 audio · 7 linguistic · 1 physio.
  - **⏸️ PROJECT ON HOLD (June 21)** at user's request — all work committed + pushed to branch
    `feat/audio-linguistic` (NOT merged to `main`). Server run cmd: `BLITZ_OVERLAY_CONTENT=ollama
    BLITZ_OVERLAY_BASELINE_SECONDS=20 python3 -m blitz_overlay`. Resume from the roadmap below.

**📈 CUE-EXPANSION ROADMAP (toward the ~300-cue polygon — "more cues every time"):** now at 32 cues.
- **Tier 1 — ✅ PARTLY DONE (June 19):** 10 blendshape/head-pose cues added (Visual 11→21). Still on the
  table for free: the *remaining* unused blendshapes (cheekPuff, mouthFunnel, tongueOut…) + **478-landmark
  geometry** (lip-part distance, mouth-corner asymmetry, brow height, nostril flare/width). More no-RAM cues.
- **Tier 2 — MediaPipe Pose/Holistic (Body family):** torso/hands/neck/self-adaptors/shrug — browser-native,
  low weight. Adds `B1…Bn` to the polygon.
- **Tier 3 — heavy models (deferred, fight 8 GB):** **OpenGraphAU (41 facial AUs)** = the true "detailed AU"
  layer; **Parselmouth** (jitter/shimmer/HNR/VOT audio); **MMPose** 133-keypoint body. Python-side, big RAM.
- **Tier 4 — RF-DETR (researched June 19):** Apache-2.0 real-time DETR (detection/segmentation/keypoints);
  30–126M params, **GPU/TensorRT-oriented, rough on Apple Silicon, no browser path.** Unique value =
  **object-manipulation cue** (phone/cup/pen pacifying, catalog #20) via ONNX at low fps — OPTIONAL later,
  behind a seam, accepting memory cost. Its pose duplicates MediaPipe (skip). Honest note: gross-body/object
  cues are weak (d≈0.3–0.5); content + facial/voice dominate accuracy — more cues mainly enrich the polygon.
- **🧊 3D RECONSTRUCTION ("Tesla-style vision") — research chapter (June 23):**
  `docs/research/2026-06-23-3d-reconstruction-vision.md`. Key finding: **we ALREADY have a real-time 3D
  face model** (MediaPipe 478 pts +z, head-pose matrix, 52 blendshapes = a 3D expression vector space) —
  we just read it as 2D. Cheap next = *use* that 3D (depth motion, full 3D pose, 3D asymmetry). Real
  upgrade = **EMOCA/SPARK** 3D-expression (Tier-3, after more RAM). Photoreal **Gaussian-splat** avatar =
  low-ROI for lies, needs a GPU. Honest: a better 3D *view* sharpens cues, it does NOT make a lie certain
  (ceiling stays ~70–75%). Slots in as a swappable visual front-end behind the existing `CueDetector`.
- **Tier 2.5 — semantic-skin rPPG:** ✅ a *light* version shipped (YCbCr per-pixel skin mask in the browser,
  no model). A heavier MediaPipe ImageSegmenter skin mask is the optional upgrade if rPPG still struggles.
  (Instance segmentation stays Tier-4 / object cues only.)
- ⏭️ **NEXT when resumed, in priority order:** (1) **Body family** (MediaPipe Pose/Holistic → `B1…Bn`,
  torso/hands/neck, low weight) — the next "more cues" win and the 5th voter. (2) more Tier-1 landmark-geometry
  cues. (3) Cross-modal coherence meta-cue. Deferred (fight 8 GB → "after more RAM"): OpenGraphAU 41 AUs,
  Parselmouth audio, MMPose, RF-DETR object cue. **Possible accelerant the user raised: a stronger model
  (e.g. Fable 5) to build faster + sharper.**
- Consensus voters: **Visual · Audio · Linguistic · Physio** (all 4 live on the branch; Body would make 5).

**ENVIRONMENT / where + how it operates (so we never redo this):**
- Machine: **macOS (darwin), shell zsh, Apple M1 / 8 GB RAM** — everything runs **locally**.
- Python: **`/opt/homebrew/bin/python3` = 3.14, NO venv**. Deps installed globally via homebrew pip
  (numpy, fastapi, uvicorn, websockets, starlette, pytest, ruff, httpx — all present).
- Run it: **`python3 -m blitz_overlay`** (works as-is) or **`blitz-overlay`** (console script; needs
  `pip install -e .` first — NOT yet run). Serves **http://127.0.0.1:8000**.
- Config via `.env` / `BLITZ_OVERLAY_*`: `PORT` (8000), `GATE` (0.65), `BASELINE_SECONDS` (90; use **20**
  for quick demos), `OPEN_BROWSER` (1; set 0 to not auto-open), `LOG_DIR` (logs/). No API keys needed.
- Git: remote `origin` = **github.com/blitzandres/lie-detector**. Work on feature branches; after the
  feature is confirmed working in the browser, **merge to `main` + push**. Commit trailer:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Verify: **`python3 -m pytest -q`** and **`python3 -m ruff check .`** — both MUST stay green.
- External network calls: (1) the browser fetches the **MediaPipe model from a CDN once**; (2) when
  the **Linguistic transcriber is enabled, Chrome's Web Speech API streams mic audio to Google for
  transcription** — the only path by which audio leaves the device. All cue detection / fusion /
  consensus / rPPG run **on-device**; raw **video** never leaves the browser. Web Speech is the
  pragmatic transcript source (zero install/login); a fully-local Whisper adapter (no external call)
  is the documented upgrade behind the `Transcriber` seam.
- Browser app served from `apps/overlay-web/` (static, vanilla ES modules, no bundler); Python engine
  package is `blitz_overlay/`; reusable core math in `core/` (calibration, fusion, schemas).

**AI agents — read in this order before building:**
1. `docs/superpowers/specs/2026-06-16-live-consensus-overlay-design.md` — the approved design spec (WHAT to build).
2. `planning/READINESS.md` — the 21 definition-of-ready items (gate before/at build start; the overlay IS item #5).
3. `planning/EXECUTION_ARCHITECTURE.md` — M1/8 GB memory budget (**sequential execution**) + browser-hybrid.
4. `docs/CUE_CATALOG.md` + `modalities/linguistic/RESEARCH.md` — the cues and their evidence.
5. `docs/superpowers/plans/2026-06-16-live-consensus-overlay.md` + `docs/OVERLAY_README.md` — the implementation plan + one-command quickstart (Stage 1 built).

**Key locked decisions:** Approach 2 (browser capture + Python engine, WebSocket, feature vectors only) ·
science-driven cue weights (not learned) · honest framing (statuses, never binary "LIE") · WhisperX not
CrisperWhisper · capture is a swappable source adapter (webcam → screen-region → native draw-anywhere).

---

## 📋 Document Overview

This folder contains the complete planning phase for the **Blitz Engine** — a modular, open-source behavioral deception detection system.

Treat this folder as the authoritative planning copy. External mirrors should be synced from here, not edited independently.

### Files Included

| File | Purpose | Size | Status |
|------|---------|------|--------|
| **PROJECT_MAP.md** | Canonical map of the repo, source-of-truth rules, implementation order, and current build status | 8KB | ✅ Final |
| **BLITZ_ENGINE_SPEC.md** | Complete technical specification with 5-layer architecture, 66 cues, Bayesian fusion, output schema | 30KB | ✅ Final |
| **LIE_DETECTOR_BLUEPRINT.md** | High-level project blueprint with vision, cue catalog (40→66), libraries, constraints, build phases | 31KB | ✅ Final |
| **ACCURACY_PLAN.md** | Accuracy expectations, quality gates, baseline normalization, scoring formula, Claude prompt strategy | 14KB | ✅ Final |
| **RESEARCH.md** | Implementation research: 6 gaps resolved, 2 blockers identified, library install methods | 14KB | ✅ Final |
| **COMPETITIVE_RESEARCH.md** | GitHub landscape survey (Apr 2026): 10 top repos, novel techniques, free datasets, accuracy benchmarks | ~20KB | ✅ Final |
| **READINESS.md** | Definition of Ready — 21 loose ends before/at start of dev: 9 core (manifest, plugin interface, eval harness, walking skeleton, tests, science-driven weights) + 12 cheap high-value (diarization, cue-independence fusion fix, graceful degradation, calibration, PII/data handling, secrets, reproducibility, language gate, CI) | 14KB | ⬜ Action list |
| **signal_preview.py** | VHS signal monitor preview (terminal animation demo) | 4.1KB | ✅ Runnable |
| **EXECUTION_ARCHITECTURE.md** | Memory budget (M1/8GB, sequential-execution rule) + browser-hybrid deployment (MediaPipe/WebGL client, Claude API, cloud) + per-modality placement + privacy win | 11KB | ✅ Reference |
| **../docs/superpowers/specs/2026-06-16-live-consensus-overlay-design.md** | ⭐ CURRENT BUILD — approved design for the Live Consensus Overlay: architecture, cue→landmark mapping, consensus mechanism, capture source roadmap, first-build scope | 9KB | ✅ Approved design |
| **../modalities/linguistic/RESEARCH.md** | Deep research on the linguistic/NLP layer — RM/CBCA/VA frameworks, cue-by-cue evidence, the no-baseline (podcast) case, transformer caveats | 13KB | ✅ Reference |

**Total:** ~120 KB of documentation, research, and code scaffolding

---

## 🎯 Quick Start — Read in This Order

1. **PROJECT_MAP.md** — Start here to understand what is real vs planned
   - Canonical source of truth for repo structure
   - Current implementation status by directory
   - Recommended build sequence
   - Rules for syncing mirror copies

2. **BLITZ_ENGINE_SPEC.md** — Canonical architecture source of truth
   - 5-layer design (ingestion, feature extraction, calibration, fusion, output)
   - All 66 behavioral cues across 5 modality families
   - Personal baseline calibration + Bayesian log-odds fusion
   - Output schema + API examples

3. **ACCURACY_PLAN.md** — How to achieve 70-75% accuracy
   - Quality gates per cue
   - Baseline normalization (robust Z-score)
   - Temporal feature extraction
   - Cue reliability weights + scoring formula
   - Claude prompt strategy

4. **RESEARCH.md** — Technical implementation details
   - 6 implementation gaps: all resolved with specific code examples
   - 2 blockers: CrisperWhisper license, AU28 missing from OpenGraphAU
   - Library-by-library install instructions
   - Local-first deployment notes + optional remote fallback references
   - Chrome Extension MV3 architecture

5. **LIE_DETECTOR_BLUEPRINT.md** — Historical blueprint + cue catalog
   - Original vision and product framing
   - Full 66-cue catalog and reliability ranking
   - Library decisions and constraints history
   - Use for context, not as the architecture source of truth

6. **signal_preview.py** — Run to see the VHS signal UI
   ```bash
   python signal_preview.py
   ```

---

## 🏗️ Architecture at a Glance

```
┌────────────────────────────────────────────────────────────┐
│  LAYER 1: INGESTION                                         │
│  File / CLI / API / Extension adapters → continuous clip   │
├────────────────────────────────────────────────────────────┤
│  LAYER 2: FEATURE EXTRACTION (66 Cues)                     │
│  Visual (20) | Audio (13) | Linguistic (18) | Physio (5)   │
│  CBCA/RM (10) → CueEvent objects with timestamps           │
├────────────────────────────────────────────────────────────┤
│  LAYER 3: PERSONAL BASELINE CALIBRATION                    │
│  90-180s baseline → Robust Z normalization per cue         │
├────────────────────────────────────────────────────────────┤
│  LAYER 4: BAYESIAN FUSION                                  │
│  Cue-level experts → Cross-modal attention → Log-odds      │
│  Convergence gate: 2+ modality families required           │
├────────────────────────────────────────────────────────────┤
│  LAYER 5: OUTPUT                                            │
│  risk_score | uncertainty | cue_attribution | narrative    │
└────────────────────────────────────────────────────────────┘
```

---

## 🎯 The 66 Deception Cues

Organized by modality:

### Visual (20 cues)
Blink rate, micro-expressions, lip compression, pupil dilation, gaze aversion, asymmetric smile, jaw tension, self-touching, head movement, postural shifts, emblematic slip, gaze fixation, blink rebound, transdermal blood flow, eye blocking, eyebrow raise, nose wrinkle, speech-gesture mismatch, reduced hand gestures, and more.

### Audio (13 cues)
Pitch increase, voice tremor, HNR drop, speech rate change, filler words, pause duration, volume drop, VOT shortening (AUC 0.89), articulation vs speaking rate, spectral tilt, Contact Quotient, formant dispersion, energy-per-syllable irregularity.

### Linguistic (18 cues)
Response delay, distancing language, qualifier overload, sensory detail poverty, tense inconsistencies, pronoun avoidance, abnormal answer length, negative emotion density, narrative coherence, verifiable entity poverty, spontaneous corrections (72-74% standalone), cognitive operations density, lexical diversity, syntactic complexity, direct quote ratio, narrative proportion imbalance, statement contradiction (NLI), complication events (CBCA).

### Physiological (5 cues)
Heart rate spike, HRV drop, skin color / blood flow change, multi-ROI rPPG divergence, contactless EDA proxy.

### CBCA/RM (10 cues)
Unusual peripheral details, admitted perceptual uncertainty, cognitive operations contrast, logical structure, contextual embedding, interactions/conversations, reproduction of conversation, unexpected complications, superfluous details, spontaneous corrections.

---

## 📊 Accuracy Expectations

| System | Accuracy | Conditions |
|--------|----------|-----------|
| Human judges (baseline) | 54% | 24,483 judges, 206 studies |
| SVC 2025 competition winner | ~62% | Hard cross-domain setting |
| **Blitz Engine Phase 1** | **70-75%** | With quality gating + baseline + abstain option |
| **Blitz Engine Phase 2** | **75-80%** | With online drift correction |
| Phase 3 + thermal camera | ~83-87% | Validated lab conditions only |

**Key insight:** Low-70s is achievable only with:
- Good A/V quality (720p+, stable face, clean audio)
- Abstain option (don't force verdict on bad footage)
- Robust baseline normalization per session
- Quality-aware cue weighting

---

## 🔧 Tech Stack (100% Free)

### Visual Layer
- **MediaPipe** (Apache 2.0) — Face mesh, blink detection, hand tracking
- **OpenGraphAU** (Apache 2.0) — 41 facial action units (micro-expressions)
- **rtmlib/MMPose** (Apache 2.0) — 133-point body pose keypoints
- **InsightFace** (MIT code, non-commercial models) — Gaze, head pose

### Audio Layer
- **librosa** (ISC) — Pitch (pyin), energy (rms)
- **Parselmouth** (GPL-3.0) — Jitter, shimmer, HNR, voice tremor
- **CrisperWhisper** (CC-BY-NC-4.0) — Filler words, word-level timestamps
- **vitallens-python** (MIT) — Heart rate + HRV from video (rPPG)

### Linguistic Layer
- **spaCy** (MIT) — POS, NER, morphology, dependency parsing
- **VADER** (MIT) — Sentiment for conversational text
- **NRCLex** (MIT) — 8-emotion breakdown per word
- **TextDescriptives** (MIT) — Coherence, readability metrics
- **sentence-transformers** — Semantic distance for CBCA criteria

### Backend
- **FastAPI** (MIT) — REST API wrapper
- **Claude API** (Anthropic) — Structured scoring + narration
- **Python 3.11+** — Runtime
- **Localhost first** — Zero-cost research target
- **Oracle Cloud / HF Spaces** — Optional free remote fallback

---

## ⚠️ Key Blockers (Resolved)

### Blocker 1: CrisperWhisper License
**Issue:** CrisperWhisper is CC-BY-NC (non-commercial only)

**Resolution:** Use WhisperX (BSD-2) as fallback for commercial; CrisperWhisper fine for Phase 1 personal research.

### Blocker 2: AU28 Not in OpenGraphAU
**Issue:** OpenGraphAU skips AU28 (jaw tension)

**Resolution:** Use MediaPipe Face Mesh landmark distances (jaw width ratio) as fallback.

---

## 🚀 Build Phases

### Phase 0 — Foundation ✅
- Complete blueprint research ✅
- Write Blitz Engine spec ✅
- Finalize cue catalog ✅
- Setup GitHub repo ✅

### Phase 1 — Core Engine (ready to start)
1. Linguistic module (spaCy + VADER + NRCLex)
2. Audio module (librosa + Parselmouth + CrisperWhisper)
3. Visual module (MediaPipe + OpenGraphAU + rtmlib)
4. rPPG module (vitallens-python)
5. Fusion + Claude scoring
6. VHS signal bar UI

### Phase 2 — Applications
- FastAPI REST wrapper
- Python SDK (pip installable)
- Chrome Extension widget
- UI improvements

### Phase 3 — Validation
- Benchmark on Real-Life Trial dataset
- Fairness audit (subgroup accuracy)
- Publish model card

### Phase 4 — Hardware
- Thermal camera adapter
- Validated accuracy ceiling: ~83-87%

---

## 📝 Key Decisions

| Decision | Choice | Evidence |
|----------|--------|----------|
| Fusion | Hybrid (experts + attention + Bayesian) | DOLOS/PECL cross-domain benchmark |
| Baseline | 90-180s (not 30-60s) | Bogaard et al. 2024 |
| Normalization | Robust Z (median/MAD) | Outlier resistance |
| Gender | Person-relative delta (NOT gender-stratified) | Hall et al. 2025 |
| Prior probability | 0.30 (not 0.50) | Avoid overconfidence |
| Convergence gate | 2+ independent modalities | Prevent single-modality false alarms |
| License | Apache 2.0 | Academic + personal use |
| Prohibited uses | Hiring, law enforcement, healthcare, EU high-risk | EU AI Act 2024/1689 |

---

## 📚 Research Foundation

**Literature:**
- Bond & DePaulo (2006) — Human deception detection baseline (24,483 judges)
- Bogaard et al. (2024) — Baseline adequacy for deception cues
- DOLOS (2023) — Multi-modal deception benchmark
- SVC 2025 — Deception detection competition (best: 62%)
- EU AI Act 2024/1689 — High-risk AI application classification
- Hall et al. (2025) — Gender fairness in deception detection

**Datasets:**
- DOLOS (1,675 clips) — TV gameshow deception
- Real-Life Trial (121 clips) — Courtroom footage
- MU3D (320 clips) — Demographic-balanced
- CASME2 (247 samples) — Micro-expressions @ 200fps

---

## 🎬 VHS Signal Monitor Preview

Run the signal preview to see the proposed UI:

```bash
python signal_preview.py
```

This shows:
- 8 active behavioral cues
- Real-time signal bars (green → yellow → orange → red)
- Overall stress score
- AI narration at the bottom
- Status transitions over time

---

## ✅ What's Complete

- [x] Canonical planning map (PROJECT_MAP.md)
- [x] Full technical specification (BLITZ_ENGINE_SPEC.md)
- [x] Project blueprint with 66 cues (LIE_DETECTOR_BLUEPRINT.md)
- [x] Accuracy planning + scoring formula (ACCURACY_PLAN.md)
- [x] Implementation research + blockers resolved (RESEARCH.md)
- [x] Library verification (all MIT/Apache/free)
- [x] Architecture validation
- [x] Ethical framework (Apache 2.0 + prohibited uses)
- [x] VHS signal UI preview (signal_preview.py)
- [x] GitHub repo initialized + planning folder tracked
- [x] Governance docs present in `governance/`

## ⏭️ Next Steps

1. **Keep `planning/` canonical** — Update repo docs first, then sync any external mirrors
2. **Start Phase 1, Task 1** — Implement linguistic module (`modalities/linguistic/`)
3. **Build local-first API/CLI scaffolding** — `apps/web-api/` + `apps/cli/`
4. **Create test harness** — Unit tests and dataset fixtures for each modality module
5. **Begin audio module** — librosa + Parselmouth + CrisperWhisper/WhisperX fallback

---

## 📞 Questions?

Refer to:
- **"How do I use [library]?"** → RESEARCH.md (install + minimal code)
- **"What's the accuracy?"** → ACCURACY_PLAN.md + BLITZ_ENGINE_SPEC.md
- **"What are the 66 cues?"** → LIE_DETECTOR_BLUEPRINT.md (complete table)
- **"How does fusion work?"** → BLITZ_ENGINE_SPEC.md, Layer 4
- **"What's the output format?"** → BLITZ_ENGINE_SPEC.md, Layer 5

---

*Consolidated from earlier lie detector planning into the tracked Blitz Engine repo — April 1, 2026*
*Ready for Phase 1 implementation sequencing*
