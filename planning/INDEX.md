# Blitz Engine — Complete Project Documentation
> Consolidated April 1, 2026 | Canonical planning set tracked in the repo

---

## 🚦 START HERE — current focus (updated June 16, 2026)

**What this is:** the **Live Consensus Overlay** ("AI-vision") — a real-time, sports-analysis-style
overlay where MediaPipe maps the face/body, the Python engine detects deception **cues** and aggregates
them via a live **consensus mechanism** (Bayesian fusion + two-gate), and the browser draws a telestrator
+ consensus panel on top of the video. Raw video never leaves the device. Run it: `blitz-overlay`.

**STATUS (updated June 16, 2026):**
- ✅ **DONE & on `main` (GitHub):** Stage 1 walking skeleton (Visual + Physio families, 5 visual cues +
  rPPG, rolling baseline, family fusion + two-gate, consensus, prediction log, FastAPI one-command server,
  browser telestrator). Plus the **enneagram viz**, **right-hand dashboard layout**, and **live family
  activity bars** (online + activity per family). 80 tests, ruff+pytest CI.
- 🔄 **Built, on branch `feat/audio-linguistic` (NOT pushed — awaiting user mic test):** the **Audio family**
  (browser Web Audio: pitch/pause/tremor → `blitz_overlay/cues/audio.py`, wired voter + enneagram slot 6).
- ⏭️ **NEXT, in priority order** (details in the dev memory `project_blitz_overlay.md`): (1) **Linguistic family**
  (live transcript via Web Speech API → reuse `modalities/linguistic/analyzer.py` lexicon cues → `cues/linguistic.py`);
  (2) **Cross-modal coherence meta-cue** (audio×movement sync = channel discrepancy / speech-gesture mismatch —
  WebSearch the literature first, then a low-weight cross-family modifier); (3) **"Bell" alarm + trust log**
  (earned bell only on a sustained all-families two-gate FLAG; bell frequency over time = inverse trust meter);
  (4) **Body/Posture as a 5th family** (MediaPipe Pose: postural shifts, self-adaptors, shrug, reduced gestures — low weight).
- The 4 consensus voters are **Visual · Audio · Linguistic · Physio** (Audio now live; Body would make 5).

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
