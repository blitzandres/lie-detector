# Blitz Engine

> The open-source behavioral deception detection core.

Blitz Engine is a modular, research-driven engine that analyzes live webcam + mic input (plus text and WAV audio) to detect behavioral deception signals. The flagship app is the **Live Consensus Overlay**: 40 real-time cues across 4 voting families (visual · audio · linguistic · physio), fused with a Bayesian log-odds architecture, personal baseline calibration, and a two-gate consensus — plus an optional local-LLM **content engine** that judges the meaning of each Q&A answer and confirms it against the time-aligned cue activity.

**Not a lie detector. A behavioral signal analyzer.**

**▶ Run the Live Consensus Overlay:** see [docs/OVERLAY_README.md](docs/OVERLAY_README.md) — one command: `blitz-overlay`.

---

## What makes it different

| Feature | Other systems | Blitz Engine |
|---|---|---|
| Baseline | Population thresholds | 90-180s personal calibration |
| Temporal analysis | Averages over clip | 3-phase window per question |
| Fusion | Sum of scores | Bayesian log-odds + convergence gate |
| Correlated cues | Double-counted | Family grouping + effective dimensionality |
| Gender handling | Fixed or stratified weights | Person-relative delta, fairness-audited |
| Output | Single score | Score + uncertainty + cue attribution + compliance |
| Architecture | Monolithic | Modular plugins + canonical CueEvent schema |

---

## 📚 Planning Phase Documentation

All planning artifacts consolidated in [`planning/`](planning/) folder:

| Document | Purpose |
|---|---|
| **[planning/PROJECT_MAP.md](planning/PROJECT_MAP.md)** | Canonical project map — repo reality, source-of-truth rules, implementation order |
| **[planning/INDEX.md](planning/INDEX.md)** | Start here — overview of all planning documents and quick navigation |
| **[planning/BLITZ_ENGINE_SPEC.md](planning/BLITZ_ENGINE_SPEC.md)** | Complete technical specification (30KB) — all 5 layers, 66 cues, APIs, output schema |
| **[planning/LIE_DETECTOR_BLUEPRINT.md](planning/LIE_DETECTOR_BLUEPRINT.md)** | Project blueprint (31KB) — vision, constraints, tech stack, library verification |
| **[planning/ACCURACY_PLAN.md](planning/ACCURACY_PLAN.md)** | Accuracy strategy (14KB) — quality gates, baseline normalization, scoring formula |
| **[planning/RESEARCH.md](planning/RESEARCH.md)** | Implementation research (14KB) — 6 gaps resolved, 2 blockers, library install methods |
| **[planning/COMPETITIVE_RESEARCH.md](planning/COMPETITIVE_RESEARCH.md)** | Competitive landscape (Apr 2026) — 10 top repos, novel techniques, free datasets, accuracy benchmarks |
| **[planning/signal_preview.py](planning/signal_preview.py)** | VHS signal UI demo — run with `python planning/signal_preview.py` |

`planning/` is the canonical research source tracked in Git. If you keep external mirror copies, sync them from here.

**Total:** ~100 KB planning documentation. Ready for Phase 1 implementation sequencing.

## Current Build Status

The flagship build is the **Live Consensus Overlay** (`blitz_overlay/` + `apps/overlay-web/`):

- **40 live cues · 4 voting families** — 29 visual (MediaPipe blendshapes, iris, landmark geometry), 3 audio (browser Web Audio scalars), 7 linguistic (live transcript lexicons), 1 physio (skin-aware webcam rPPG with an honest abstain gate)
- **Two-gate consensus** — a FLAG requires ≥2 independent families AND posterior ≥ 0.65; statuses are CALIBRATING → CLEAR → WATCH → FLAG, never a binary "LIE"
- **Content engine (optional)** — a local Ollama LLM judges each Q&A answer for content-pattern markers and fuses content-first with the cue timeline for that answer's window; degrades gracefully when Ollama is absent
- **Live visualizations** — deforming enneagram (family view), radial Cue Polygon (per-cue synchrony view), synchrony bell + trust meter, hard-gated calibration card
- **Hard-gated calibration** — the rolling personal baseline won't complete until every producing cue has enough samples

The repository also includes the earlier **text + WAV audio engine** (`blitz_engine/engine.py`, `modalities/`), which shares the same calibration and fusion core.

The repository also ships the **research tier**: `blitz analyze-video` runs recorded clips through Py-Feat AUs + optional Farneback optical flow (`pip install -e ".[research]"`), emitting the same CueEvents into the shared fusion core.

Not yet implemented:

- Body family (MediaPipe Pose — 5th voter, planned in `planning/STAGE2_CUE_DETECTION_PLAN.md`)
- REST API adapter

Install locally from the repo root:

```bash
pip install -e .
```

CLI usage:

```bash
blitz analyze-text \
  --baseline-file baseline.txt \
  --response-file response.txt \
  --question "Where were you Tuesday night?"
```

Minimal text-only example:

```python
from blitz_engine import BlitzEngine

engine = BlitzEngine(modalities=["linguistic"])
session = engine.new_session(
    baseline_texts=[
        "I drove to work and grabbed coffee before the meeting.",
        "My usual breakfast is eggs, toast, and tea at home.",
        "Last weekend I cleaned the apartment and watched a movie.",
        "I usually walk to the store in the afternoon for groceries.",
        "My morning routine starts with stretching and checking messages.",
    ],
    consent=True,
    use_case="research",
    jurisdiction="CA-US",
    baseline_duration_s=120,
)

result = session.analyze_text(
    response_text="Honestly, I really do not know, um, I was basically with someone at that place.",
    question="Where were you Tuesday night?",
    response_latency_ms=900,
)

print(result.risk_score)
print(result.narrative)
```

The engine also supports a WAV-based audio modality from Python via `BlitzSession.analyze(...)`. The CLI still targets the text-first workflow.

---

## Accuracy (Honest)

| System | Accuracy |
|---|---|
| Human judges (Bond & DePaulo 2006, 24k judges) | 54% |
| SVC 2025 competition winner (cross-domain) | ~62% |
| Blitz Engine Phase 1 target | 70-75% |
| Blitz Engine Phase 2 target (with drift correction) | 75-80% |

The 85-99% numbers in papers are lab overfitting on tiny datasets (121-320 clips). We don't claim those numbers.

---

## How It Works

### 1 · Live Consensus Overlay — dataflow (as built)

All heavy extraction happens in the browser; only tiny feature vectors cross a localhost WebSocket. Raw video never leaves the device.

```mermaid
flowchart LR
    subgraph BROWSER ["BROWSER — all extraction, raw media never leaves"]
        direction TB
        CAM["Webcam"] --> MP["MediaPipe Face Landmarker\n478 landmarks · 52 blendshapes\niris radius · head pose"]
        CAM --> RPPG["rPPG sampler\nskin-masked ROI means (YCbCr)\n+ skin_fraction quality"]
        MIC["Mic"] --> WA["Web Audio scalars\nF0 · energy · pause ratio · tremor proxy"]
        MIC -.->|"optional — in-page notice + toggle\n(Chrome cloud STT)"| STT["Web Speech API\nlive transcript"]
        MP --> FF["FeatureFrame (~30 Hz)\nfeature vectors only"]
        RPPG --> FF
        WA --> FF
        STT -.-> FF
    end

    FF -->|"localhost WebSocket"| CUES

    subgraph ENGINE ["PYTHON ENGINE — localhost, local math only"]
        direction TB
        CUES["40 cue detectors\n29 visual · 3 audio · 7 linguistic · 1 physio"]
        CUES --> CAL["Rolling personal baseline\nz = (x − median) / (1.4826 × MAD)\nhard-gated calibration\n(every producing cue needs ≥8 samples)"]
        CAL --> FUSE["Science-weighted family fusion\nlogit(P) = logit(0.30) + Σ w·(d·z − d²/2)\nweights fixed from published effect sizes"]
        FUSE --> GATE{"Two-gate consensus\nposterior ≥ 0.65\nAND ≥ 2 independent families"}
        CUES --> SYNC["Synchrony detector\nburst = ≥K lit cues across ≥2 families\nwithin ~1s window"]
        SYNC --> BELL["Bell + trust meter\nrings only on burst + sustained risk\n(honest label, never 'LIE')"]
        GATE --> BELL
    end

    GATE --> OUT["Consensus payload (~10 Hz)\nstatus · risk · families (online/activity/vote)\nactive cues · convergence · bell"]
    OUT -->|"WebSocket"| UI["Overlay UI\ntelestrator · consensus panel\nenneagram (family view)\nCue Polygon (per-cue view)"]
```

Statuses are **CALIBRATING → CLEAR → WATCH → FLAG** — never a binary "LIE". A FLAG requires two independent modality families to agree while combined risk clears the gate.

---

### 2 · Content Engine — Q&A layer (as built, optional)

The meaning of speech is the primary layer; behavioral cues are secondary, time-aligned confirmation. A local Ollama LLM judges each answer's content patterns — never factual truth.

```mermaid
flowchart LR
    Q["Q&A panel\nquestion asked"] --> ANS["Answer transcript\n+ [t0, t1] window"]
    ANS -->|"WS 'turn' message\n(off-thread, never blocks cues)"| JUDGE["ContentJudge — local Ollama LLM\nconsistency · RM richness\nverifiability · evasion"]
    TLB["TimelineBuffer\nper-frame cue rhythm"] -->|"window(t0, t1)\ncue activity during the answer"| CFUSE["Content-primary fusion\ncontent leads · cues confirm"]
    JUDGE --> CFUSE
    CFUSE --> VERDICT["Turn verdict → browser\ncontent risk + cue confirmation"]
```

Enable with `BLITZ_OVERLAY_CONTENT=ollama` (small local model, e.g. `llama3.2:3b`). Without Ollama the system degrades gracefully to the cue engine alone.

---

### 3 · Fusion & Consensus — the math (as built)

```mermaid
flowchart TD
    IN(["FeatureFrame stream"])

    IN --> S1["STAGE 1 · Cue detection\n─────────────────────\n32 detectors emit raw cue values\neach with a science weight:\nd (effect size) + reliability w\ncitation-traceable, fixed — never learned"]

    S1 --> S2["STAGE 2 · Robust baseline normalization\n─────────────────────\nz_i = (x − median) / (1.4826 × MAD + ε)\nrolling personal window — person-relative,\nnot population thresholds"]

    S2 --> S3["STAGE 3 · Bayesian log-odds accumulation\n─────────────────────\nlogit(P) = logit(0.30) + Σ [ w_i × (d_i × z_i − d_i²/2) ]\nprior 0.30 · grouped by family\n(visual · audio · linguistic · physio)"]

    S3 --> GATE{"STAGE 4 · Two-gate convergence\n─────────────────────\nposterior ≥ 0.65\nAND ≥ 2 independent families agree"}

    GATE -->|"not passed"| STATUS["Status: CLEAR or WATCH\n(honest cap: FLAG unreachable\nwith < 2 fresh families)"]
    GATE -->|"passed"| FLAG["Status: FLAG\n⚠ high deception-pattern risk\nnot_for_sole_decision: true"]

    STATUS --> LOGN["Prediction log\nderived decisions only —\nnever raw biometric data"]
    FLAG --> LOGN
```

The sensitivity slider in the UI moves only the bell/burst operating point (K, lit-z, risk floor). **The science weights never move**, and nothing is ever trained on the system's own predictions.

---

### 4 · Research Tier — offline visual analyzer (built)

For recorded video only — these models need raw frames and are too heavy for the live path (M1/8GB rule: one model loaded at a time, sequential). Install the heavy backends with `pip install -e ".[research]"`, then:

```bash
blitz analyze-video \
  --baseline-video b1.mp4 --baseline-video b2.mp4 --baseline-video b3.mp4 \
  --response-video response.mp4 --question "Where were you Tuesday?" \
  --optical-flow
```

At least 3 baseline clips are required (the personal baseline needs 3+ observations per cue).

```mermaid
flowchart LR
    VID["Recorded video file"] --> PF["Py-Feat Detector v2\n20 AUs w/ intensity · emotions\nvalence/arousal · gaze · head pose"]
    VID --> OG["OpenGraphAU\ncomplementary AU detector\n(ensemble robustness)"]
    VID --> OF["OpenCV optical flow\n(Farneback) temporal dynamics\nmicro-expression spotting"]
    PF --> CE["Typed CueEvents\ncue_id · value · effect_size\nreliability_w · timestamps"]
    OG --> CE
    OF --> CE
    CE --> SAME["Same fusion core\nMAD baseline → log-odds → two-gate"]
```

Nine clip-level cues (AU combos, Duchenne deficit, emotion leakage, head dynamics, expressivity rigidity, micro-burst proxy, flow agitation) feed the same personal-baseline + log-odds fusion as text/audio. Py-Feat is the working backend; OpenGraphAU (ensemble) and LibreFace (fallback) are ready seams behind the same `AUBackend` contract. Honest caveat carried into governance docs: micro-expressions have low base rates and modest real-world effect sizes.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: INGESTION — video + audio normalization            │
├─────────────────────────────────────────────────────────────┤
│  LAYER 2: FEATURE EXTRACTION — 5 modality plugins           │
│  Visual | Audio | Linguistic | Physiological | CBCA/RM       │
├─────────────────────────────────────────────────────────────┤
│  LAYER 3: PERSONAL CALIBRATION — 90-180s baseline           │
│  Trait + state stress + deception residual                   │
├─────────────────────────────────────────────────────────────┤
│  LAYER 4: BAYESIAN FUSION — hybrid architecture             │
│  Cue experts → cross-modal attention → log-odds → gate      │
├─────────────────────────────────────────────────────────────┤
│  LAYER 5: OUTPUT — risk score + uncertainty + attribution    │
└─────────────────────────────────────────────────────────────┘
```

Full specification: [planning/BLITZ_ENGINE_SPEC.md](planning/BLITZ_ENGINE_SPEC.md)

---

## The 66-Cue Catalog

The research catalog spans 66 cues; **40 are live in the overlay today** (29 visual · 3 audio · 7 linguistic · 1 physio). The rest arrive via the Body family, the offline research tier, and deeper audio.

| Domain | Catalog count | Key signals |
|---|---|---|
| Visual / facial | 20 | Micro-expressions, gaze fixation, blink rebound, pupil dilation |
| Audio / vocal | 13 | VOT shortening (AUC 0.89), voice tremor, formant dispersion |
| Linguistic / NLP | 18 | Spontaneous corrections, MTLD diversity, NLI contradiction |
| Physiological rPPG | 5 | Multi-ROI divergence, contactless EDA proxy |
| CBCA / RM | 10 | Peripheral details (g=0.64), cognitive operations density |

Full catalog: [docs/CUE_CATALOG.md](docs/CUE_CATALOG.md)

---

## Quick Start

Run the live overlay (webcam + mic, everything local):

```bash
pip install -e .
blitz-overlay        # opens http://127.0.0.1:8000
```

With the content engine (needs [Ollama](https://ollama.com) + a small model):

```bash
ollama pull llama3.2:3b
BLITZ_OVERLAY_CONTENT=ollama blitz-overlay
```

Full guide (config, tuning, privacy): [docs/OVERLAY_README.md](docs/OVERLAY_README.md)

### Video-file SDK (planned — research tier)

```python
from blitz_engine import BlitzEngine

engine = BlitzEngine(modalities=["visual", "audio", "linguistic"])

session = engine.new_session(
    baseline_video="baseline_90s.mp4",
    consent=True,
    use_case="research",
    jurisdiction="CA-US"
)

result = session.analyze(
    video_clip="response.mp4",
    question="Where were you on Tuesday?"
)

print(result.risk_score)     # 0.72
print(result.uncertainty)    # 0.15
print(result.narrative)      # "At 14.2s, VOT shortening + jaw tension..."
```

---

## Repo Structure

```
blitz-engine/
├── core/              Engine: schemas, calibration, fusion, scoring, quality
├── modalities/        Plugins: visual, audio, linguistic, physiological, cbca_rm
├── apps/              Adapters: chrome-extension, web-api, cli
├── evaluation/        Benchmarks, fairness audits, baselines
├── governance/        Ethics, intended use, prohibited uses, model card
├── docs/              Architecture spec, cue catalog, guides
└── planning/          Canonical research bundle and implementation map
```

---

## Ethics & Legal

This software is for **research, education, and journalism only**.

- Accuracy is 70-75% — false positive rate is ~25-30%
- Every API call requires declared consent, use case, and jurisdiction
- `not_for_sole_decision` flag is always true and cannot be disabled
- **Prohibited:** hiring, law enforcement, insurance, healthcare, education discipline

See [governance/PROHIBITED_USES.md](governance/PROHIBITED_USES.md) and [governance/ETHICS.md](governance/ETHICS.md).

EU AI Act (Regulation 2024/1689) applies. Do not deploy for high-risk uses in EU without legal review.

---

## Status

- [x] Phase 0 — Research complete, planning consolidated, repo initialized
- [x] Live Consensus Overlay — 40 cues, 4 voting families, two-gate consensus, hard-gated calibration, skin-aware rPPG, synchrony bell
- [x] Content engine — local Ollama LLM judges Q&A answers, content-primary fusion with the cue timeline
- [x] Text + WAV engine — CLI `blitz analyze-text`, linguistic + audio analyzers on the shared fusion core
- [ ] Body family — MediaPipe Pose as the 5th voter (`planning/STAGE2_CUE_DETECTION_PLAN.md`)
- [x] Research tier — offline visual analyzer (`blitz analyze-video`: Py-Feat v2 + Farneback optical flow via `.[research]`; OpenGraphAU/LibreFace are ready seams)
- [ ] Validation — benchmark + fairness audit + model card
- [ ] Hardware extension — thermal camera

---

## License

Apache 2.0 — free for academic, research, and personal use. Attribution required.
See [governance/PROHIBITED_USES.md](governance/PROHIBITED_USES.md) for use restrictions.

---

## Competitive Landscape

Key open-source repositories in this field (full analysis in [planning/COMPETITIVE_RESEARCH.md](planning/COMPETITIVE_RESEARCH.md)):

| Repo | What it does | Relevance |
|---|---|---|
| [RH-Lin/MMPDA](https://github.com/RH-Lin/MMPDA) | SVC 2025 winner — cross-domain adaptation | Phase 2 domain shift fix |
| [dclay0324/ATSFace](https://github.com/dclay0324/ATSFace) | LoRA per-subject calibration → 92% | Phase 2 calibration upgrade |
| [NMS05/DOLOS-PECL](https://github.com/NMS05/Audio-Visual-Deception-Detection-DOLOS-Dataset-and-Parameter-Efficient-Crossmodal-Learning) | Temporal adapter + crossmodal fusion (ICCV 2023) | Architecture reference |
| [cai-cong/MDPE](https://github.com/cai-cong/MDPE) | 104hr dataset + Big Five personality (HuggingFace) | Free training data |
| [ubicomplab/rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) | NeurIPS 2023 rPPG — HR/HRV from webcam | vitallens upgrade path |
| [Redaimao/awesome-deception](https://github.com/Redaimao/awesome-multimodal-deception-detection) | Curated field index | Research tracking |

**Why we're different:** Every published system uses population thresholds on tiny datasets (Michigan = 121 clips). The SVC 2025 cross-domain winner scored 60.43%. Blitz Engine's 70-75% target is achievable specifically because of personal calibration — a measurement advantage, not a model advantage.

---

## Research Foundation

- DePaulo et al., 2003 — Cues to deception (PMID: 12555795)
- Bond & DePaulo, 2006 — Accuracy of deception judgments (PMID: 16859438)
- Bogaard et al., 2024 — Baselining efficacy (doi:10.1016/j.actpsy.2023.104112)
- Guo et al., 2023 — DOLOS + PECL (arXiv:2303.12745)
- Lin et al., 2025 — SVC 2025 challenge results (arXiv:2508.04129)
- Cai et al., 2024 — MDPE dataset + personality modulation (arXiv:2407.12274)
- Lee et al., 2023 — ATSFace LoRA calibration (arXiv:2309.01383)
- Liu et al., 2023 — rPPG-Toolbox (NeurIPS 2023)
- EU AI Act, 2024 — Regulation 2024/1689
