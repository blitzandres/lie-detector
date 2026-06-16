# Blitz Engine — Project Map & Research Execution Plan
> Canonical planning map, April 1, 2026

---

## Purpose

This file turns the planning set into one coherent handoff:

- what is already present in the repo
- what is still only planned
- which document is authoritative for each topic
- what order implementation should follow

Use this before adding code or making architecture decisions.

---

## Source of Truth

- **Canonical tracked planning copy:** `blitz-engine/planning/`
- **Repository root:** `blitz-engine/` is the GitHub-backed project
- **Standalone mirror:** `/Users/andresblitz/lie-detector-consolidated/` is a synced export of the planning set

If the standalone mirror and the repo ever diverge, treat the copy inside `blitz-engine/planning/` as the authoritative version.

---

## What Exists Right Now

| Area | Status | Notes |
|---|---|---|
| `planning/` | Present and consolidated | Complete research bundle, cue catalog, accuracy plan, implementation notes |
| `docs/` | Present | Public-facing spec mirror, quickstart, cue catalog |
| `governance/` | Present | Ethics, intended use, prohibited uses, model card, license |
| `core/schemas/` | Scaffolded | `CueEvent` and `BlitzOutput` schema are stubbed |
| `core/calibration/` | Scaffolded | Baseline normalization placeholder exists |
| `core/fusion/` | Scaffolded | Bayesian fusion placeholder exists |
| `modalities/` | Planned, empty | No implementation files yet |
| `apps/` | Planned, empty | No API, CLI, or extension code yet |
| `evaluation/` | Planned, empty | No dataset loaders or benchmark harness yet |

The repo is therefore **planning-complete but implementation-light**. Only a few core placeholders exist; the engine is not built yet.

---

## Document Authority Map

| Topic | Canonical file | Why it matters |
|---|---|---|
| Repo map + execution order | `PROJECT_MAP.md` | This file tells you what is real vs planned and what to build next |
| System architecture | `BLITZ_ENGINE_SPEC.md` | Canonical 5-layer design, schemas, API shape, repo layout |
| Accuracy strategy | `ACCURACY_PLAN.md` | Quality gates, abstain logic, weighting, calibration assumptions |
| Library decisions + blockers | `RESEARCH.md` | Install notes, dataset notes, licensing blockers, deployment references |
| Cue catalog + origin story | `LIE_DETECTOR_BLUEPRINT.md` | Historical blueprint, 66-cue catalog, visual/product framing |
| Navigation overview | `INDEX.md` | Entry point for humans reading the planning set |
| UI mood / terminal concept | `signal_preview.py` | Demo of the VHS signal aesthetic |

**Important:** `LIE_DETECTOR_BLUEPRINT.md` is valuable, but it is not the architecture source of truth anymore. Use `BLITZ_ENGINE_SPEC.md` and this file when there is a conflict.

---

## Repo Map

```
blitz-engine/
├── README.md                      # Top-level project overview
├── planning/                      # Canonical research and planning bundle
├── docs/                          # Public-facing spec mirror and guides
├── governance/                    # Ethics, intended use, model card, prohibited uses
├── core/                          # Current implementation scaffolding
│   ├── schemas/                   # Present
│   ├── calibration/               # Present
│   └── fusion/                    # Present
├── modalities/                    # Planned implementation area
│   ├── visual/
│   ├── audio/
│   ├── linguistic/
│   ├── physiological/
│   └── cbca_rm/
├── apps/                          # Planned adapters
│   ├── cli/
│   ├── web-api/
│   ├── chrome-extension/
│   └── desktop/
└── evaluation/                    # Planned dataset, benchmark, fairness work
    ├── datasets/
    ├── benchmarks/
    ├── fairness/
    └── baselines/
```

---

## Recommended Read Order

1. `INDEX.md`
2. `PROJECT_MAP.md`
3. `BLITZ_ENGINE_SPEC.md`
4. `ACCURACY_PLAN.md`
5. `RESEARCH.md`
6. `LIE_DETECTOR_BLUEPRINT.md`

That order moves from orientation -> implementation reality -> architecture -> accuracy constraints -> tooling -> historical context.

---

## Execution Plan

### Phase 0 — Already Complete

- Research consolidated into the tracked repo
- Repo initialized on GitHub
- Governance docs created
- Planning copy mirrored outside the repo
- Core placeholder modules created for schemas, calibration, and fusion

### Phase 1 — Build the Engine Core

> ⚠️ **Gate: complete `planning/READINESS.md` first.** Before any cue code: (1) dependency manifest,
> (2) `ModalityPlugin` abstract interface, (3) resolve the 2 RESEARCH blockers. Then the real entry point
> is the **eval harness + one cue end-to-end (walking skeleton) + math tests** — NOT building all modules
> breadth-first. See READINESS.md for the full 8-item list and order of attack.

1. **Linguistic module**
   Target: `modalities/linguistic/`
   Why first: fastest path to a testable signal pipeline with no video dependencies

2. **Audio module**
   Target: `modalities/audio/`
   Why second: testable from audio files before the full video stack exists

3. **Visual module**
   Target: `modalities/visual/`
   Why third: more moving parts, more model dependencies, higher failure surface

4. **Physiological / rPPG module**
   Target: `modalities/physiological/`
   Why fourth: depends on clip-based capture and stronger quality gating

5. **Fusion + output assembly**
   Target: `core/fusion/`, `core/scoring/`, `core/quality/`
   Why fifth: only makes sense once upstream cues are emitting stable `CueEvent`s

6. **CLI + local API**
   Target: `apps/cli/`, `apps/web-api/`
   Why sixth: local-first research workflow is the active deployment path

### Phase 2 — Product Adapters

- Chrome Extension
- Python SDK polish
- VHS signal interface refinement

### Phase 3 — Evaluation

- Dataset loaders
- Cross-dataset benchmark harness
- Fairness audit
- Model card updates with empirical results

### Phase 4 — Hardware Extension

- Thermal adapter
- Controlled-condition validation only

---

## Dependency Logic

| Build item | Depends on | Reason |
|---|---|---|
| Linguistic module | None beyond planning | Pure text pipeline |
| Audio module | Minimal ingestion utilities | Can run on extracted audio alone |
| Visual module | Clip ingestion + model setup | Needs stable frame pipeline |
| rPPG | Clip ingestion + quality gating | Sensitive to compression, FPS, face stability |
| Fusion | CueEvent emitters | Needs normalized upstream cues |
| Extension | Local API | Adapter should not define engine behavior |
| Evaluation | Stable outputs | Benchmarking before schema stability wastes effort |

---

## Open Decisions That Still Matter

| Topic | Current answer |
|---|---|
| Canonical planning home | `blitz-engine/planning/` |
| Deployment target | 3 tiers — browser-hybrid (lean default), local Python (M1 dev), cloud 24GB (full runs). See `EXECUTION_ARCHITECTURE.md` |
| Memory budget | M1/8GB unified; **sequential execution, one heavyweight model at a time** (concurrent = OOM/thrash). See `EXECUTION_ARCHITECTURE.md` Part A |
| Browser-hybrid | Visual/rPPG/UI client-side (MediaPipe/WebGL); linguistic+contradiction via Claude API; heavy cues on cloud. Privacy: raw video stays on device. See Part B |
| CrisperWhisper | **Resolve to WhisperX** — RAM (6–7GB) + licensing both rule it out for 8GB; whisper-tiny/base in browser |
| AU28 | Custom landmark fallback required |
| Baseline duration | 90-180 seconds |
| Verdict policy | Probability + uncertainty + abstain, never binary certainty |

---

## Consolidation Definition

This project should now be treated as consolidated when all of the following stay true:

- the repo copy in `blitz-engine/planning/` is the canonical planning set
- the standalone mirror matches the repo copy
- `BLITZ_ENGINE_SPEC.md` is the architecture source of truth
- outdated references to "repo not set up yet" or "Railway is the main target" are not driving decisions
- implementation work follows the module order above

---

*Status: consolidated, mapped, and ready for Phase 1 implementation planning.*
