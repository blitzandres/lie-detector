# Definition of Ready — Before Any Development

> Blitz Engine | Written June 15, 2026
> Gate between **planning** (done) and **building** (not started). Nothing in Phase 1 starts until
> items 1–3 are checked. Items 4–6 ARE Phase 1's real entry point.
>
> **Reality check (June 15, 2026):** ~100 KB of planning docs exist, but only **263 lines of real code**
> (`core/calibration/baseline.py`, `core/fusion/bayesian_fusion.py`, `core/schemas/cue_event.py`).
> All `modalities/`, `evaluation/`, `core/quality`, `core/scoring`, and all `apps/` are empty.
> **No tests. No dependency manifest.** The project is over-planned relative to its engineering substrate.
> The gap to "ready to build" is the **skeleton**, not more research.

---

## Must-have BEFORE writing any cue code

### 1. Dependency manifest + reproducible environment  ⬜
- Add `pyproject.toml` (preferred) or `requirements.txt`, pin Python version.
- Without this, the environment is not reproducible. **Step zero.**
- **Done when:** a fresh clone can `pip install` and import `core/`.

### 2. The `ModalityPlugin` abstract interface  ⬜  *(most important open decision)*
- `cue_event.py` defines the *output* schema but there is no shared *plugin contract*.
- Define ONE abstract interface before writing any of the 5 plugins:
  `extract(inputs) -> list[CueEvent]`, declares its cue IDs, declares per-cue `baseline_needed` flag.
- All modality plugins (visual, audio, linguistic, physiological, cbca_rm) implement this identically.
- **Done when:** the interface lives in `core/` and a dummy plugin passes through fusion.

### 3. Resolve the 2 open blockers (from `RESEARCH.md`)  ⬜
- **Blocker 1 — CrisperWhisper license** (CC-BY-NC). Decide: WhisperX (BSD-2) for Phase 1, or keep
  CrisperWhisper for personal research and swap before any commercial use. Recommendation: WhisperX now.
- **Blocker 2 — AU28 (jaw tension)** missing from OpenGraphAU. Confirm MediaPipe jaw-width fallback.
- **Done when:** both decisions are written into `RESEARCH.md` as resolved.

---

## This IS Phase 1's real entry point (do 4 + 5 together, test-driven)

### 4. Evaluation harness + a real labeled dataset — BEFORE the cues  ⬜  *(highest leverage)*
- Documented #1 project risk is accuracy/domain shift collapsing to ~62%. You cannot develop a single
  cue responsibly without a way to score it. `evaluation/` is currently empty.
- Need: (a) a labeled dataset actually in hand — Real-Life Trial corpus / MU3D / DOLOS (check access +
  licensing NOW), (b) a benchmark runner, (c) one metric printed (accuracy/AUC).
- **Building 66 cues before this = flying blind.**
- **Done when:** `evaluation/` can score a (dummy) detector against a real labeled set and print a number.

### 5. Walking skeleton — one vertical slice end-to-end  ⬜
- Pick ONE cue (recommend **verifiability ratio**, cue 37 — see `modalities/linguistic/RESEARCH.md`).
- Wire the FULL pipeline for that one cue:
  `input → transcript → cue → CueEvent → robust-Z normalize → Bayesian fusion → output`.
- One cue, full depth. Proves the architecture before scaling to breadth; surfaces interface problems
  while they are cheap to fix.
- **Done when:** one real cue runs end-to-end and produces a scored `BlitzOutput`.

### 6. Tests for the correctness-critical math  ⬜
- `bayesian_fusion.py` and `baseline.py` are load-bearing: a bug in log-odds fusion silently corrupts
  every verdict. No tests exist yet.
- **Done when:** unit tests cover fusion (log-odds accumulation, two-gate convergence) and calibration
  (robust-Z / median-MAD), and they pass.

---

## Also fold in (from the no-baseline / podcast discussion)

### 7. Specify the rolling self-baseline as a calibration MODE  ⬜
- For the podcast / "no prior recognition of the person" case, the engine builds a baseline from the
  first 90–180 s of the recording instead of an enrollment clip, then scores robust-Z deviations.
- This must be a defined calibration mode in `core/calibration/`, not just prose.
- No-baseline mode outputs **"elevated risk / anomaly," never a binary verdict**; expected ceiling
  ~60–68% vs 70–77% with enrollment. (Detail in `modalities/linguistic/RESEARCH.md` §4.)
- **Done when:** `baseline.py` exposes `mode="enrollment"` and `mode="rolling"`.

### 8. Science-driven weights — NOT learned from trials  ⬜  *(DECISION, June 15 2026)*
**Decision: the engine is grounded in the published science, not trained on trial data.**

- **Cue weights `w_i` are set from published meta-analytic effect sizes** (Cohen's *d* / Hedges' *g*),
  each traceable to a citation (see `modalities/linguistic/RESEARCH.md` and the cue catalog). Fixed,
  transparent, auditable. The weight for "complications" is g≈0.5 *because the literature says so*, not
  because a dataset tuned it.
- **Labeled corpora (Real-Life Trial, MU3D, DOLOS) are used ONLY to validate** — measure accuracy and run
  the fairness audit. They MUST NOT move the weights. This prevents overfitting to one small, biased
  courtroom dataset and keeps the engine explainable.
- **No supervised learning loop in the core.** Adaptation is limited to the legitimate, science-based kind:
  the **rolling per-person baseline** (§7) — that normalizes the signal for *who is speaking*, it does not
  learn from outcomes. That is the answer to domain shift, not a trained model.

**HARD RULE — never train on unlabeled / own-prediction data** (bias amplification / model collapse), and
**do not train weights on the trial corpora either** (overfitting + dataset-specific demographic bias).
Trials measure; science decides the weights.

**Honest tradeoff:** fixed weights cannot auto-adapt to domain shift (#1 risk, ~62% cross-domain). Accept
it as the price of an explainable, defensible, bias-resistant engine; mitigate with per-person baselining.

**Optional / deferred (off by default):** a human-in-the-loop supervised recalibration could exist LATER,
fed ONLY by verified outcomes (confessions, verdicts, fact-checks) and validated against the benchmark
before shipping. Not part of the core; not built in Phase 1. Keep cheap **prediction logging** (input id,
per-cue contributions, posterior, verdict, baseline mode, empty `ground_truth` slot) purely for audits and
transparency — logging ≠ learning.
- **Done when:** weights live in a documented, citation-annotated config (each `w_i` ↔ source); the eval
  harness reads the trial corpora as read-only validation; predictions are logged for audit.

### 9. Promote cue 37 from NER-count to verifiable-detail RATIO  ⬜
- Cue 37 is currently "verifiable entity poverty" (NER entity count). Upgrade to the full Verifiability
  Approach ratio (verifiable details ÷ total details) — it is the most person-independent linguistic
  signal and the backbone of the no-baseline path.
- **Done when:** reflected in `CUE_CATALOG.md` and implemented as the cue 37 in the walking skeleton (#5).

---

## Additional loose ends — cheap, high-value (review June 15 2026)
Found in a second completeness pass. All small to add now, expensive to retrofit. Numbered 10+.

### 10. Speaker diarization  ⬜  *(biggest gap for the podcast use case)*
- **Why:** Whisper transcribes but does not say WHO spoke. Without it you cannot attribute cues to the
  right guest or build a per-person baseline — the whole podcast/no-baseline path breaks.
- **Action:** add **pyannote.audio** (MIT) as a required pre-step for any multi-speaker audio; tag every
  `CueEvent` with a `speaker_id`; baseline + scoring run per speaker.

### 11. Inter-cue independence assumption in fusion  ⬜  *(math-correctness, currently silent bug)*
- **Why:** naive log-odds ADDS each cue's evidence assuming independence. Many cues measure the same
  construct (pause + hedging + cognitive load; sensory-poverty + cognitive-ops both from RM). Summing them
  double-counts evidence → overconfident posteriors.
- **Action:** group cues into independent **families**; cap or decorrelate within-family contribution
  (e.g. take strongest-in-family, or apply a correlation down-weight). Document the assumption + mitigation
  in ACCURACY_PLAN. (The two-gate already references modality families — reuse that grouping.)

### 12. Graceful degradation / partial input  ⬜
- **Why:** audio-only podcast (no face), rPPG failing, empty transcript — must degrade, not crash or
  silently bias.
- **Action:** plugin contract returns "unavailable" cleanly; fusion drops the dead modality and widens
  uncertainty; **abstain when < 2 modality families are available** (ties to two-gate convergence).

### 13. Abstain / threshold derivation  ⬜
- **Why:** verdict policy says "abstain, never binary," but the threshold is undefined.
- **Action:** derive the abstain band + decision thresholds from the validation-set ROC; document the
  method and the band in ACCURACY_PLAN. (Validation only — does not touch cue weights.)

### 14. Probability calibration  ⬜
- **Why:** fixed literature weights do NOT guarantee a posterior of 0.70 means 70% real. Uncalibrated
  numbers mislead.
- **Action:** add a reliability-diagram check on the validation set; **report ordinal risk bands, not
  false-precise percentages.** Calibrate only the readout, never the cue weights (stays science-driven).

### 15. PII retention & deletion policy  ⬜
- **Why:** the engine processes faces, voices, words — biometric/personal data. Governance covers WHO may
  use it, not what happens to the DATA. EU AI Act expects a data lifecycle.
- **Action:** add `governance/DATA_HANDLING.md` — raw media ephemeral and auto-deleted after processing;
  persist only derived cue features + audit logs; no raw biometric retention; encryption at rest if stored.

### 16. Config / secrets handling  ⬜
- **Why:** HF_TOKEN and Claude API key must not be hardcoded or committed.
- **Action:** `.env` + `.gitignore` + a typed config schema; load secrets from env only.

### 17. Reproducibility — pin model versions + seeds  ⬜
- **Why:** Whisper / spaCy / NLI outputs drift across versions; the same video could score differently.
- **Action:** pin model versions in the manifest; set seeds; stamp model versions + seed into every
  prediction log (extends item 8).

### 18. Cue-weight provenance / versioning  ⬜
- **Why:** weights come from papers — each needs a traceable source, and changing one changes the engine.
- **Action:** `core/weights.yaml` with a citation per weight + a `weight_set_version`; stamp that version
  into every `BlitzOutput` (extends items 8 + 17).

### 19. Language scope gate  ⬜
- **Why:** cue science is English-centric (LIWC, RM lexicons). Non-English input (e.g. Spanish podcasts)
  produces confident nonsense.
- **Action:** language-detect first; **English-only for v1**, abstain/flag otherwise. Record detected
  language in the log.

### 20. Audio/video sync assertion  ⬜
- **Why:** cross-modal cues are timestamped; if streams drift you fuse the wrong moment.
- **Action:** assert a single source timebase; validate A/V sync at ingestion; fail-flag on drift.

### 21. Minimal CI + lint  ⬜
- **Why:** solo project rots fast without a guardrail.
- **Action:** one GitHub Action running `ruff` + `pytest` on push.

> Codex review was attempted but unavailable (account does not support the Codex models). This pass was
> done manually against the full planning set.

---

## Can wait (do NOT start yet)
- `apps/chrome-extension`, `apps/web-api`, VHS UI → Phase 2.
- Thermal / hardware cues (41–43) → Phase 4.
- The other 4 modality plugins (visual, audio, physiological, cbca_rm) → AFTER the skeleton (#5) proves
  the pattern. Build depth-first on one, then replicate.

---

## Suggested order of attack
1. Half-day: items **1, 2, 3** (manifest, plugin interface, resolve blockers).
2. Real Phase 1 start: items **4 + 5 + 6** together, test-driven — eval harness + one cue (37) end-to-end + math tests.
3. While building #5: add item **8** plumbing (prediction logging + updatable `w_i`/`P_0`) — cheap now, impossible to retrofit.
4. Then items **7, 9**, then replicate the plugin pattern to the remaining modalities.
5. Supervised learning loop itself (the learner in #8) is Phase 2/3 — only after a labeled benchmark exists.

> One vertical slice (eval harness + one cue end-to-end) de-risks this project more than any amount of
> additional cue research. Stop planning, build the skeleton.
