# Content-First Q&A Architecture (Design Spec)

> Date: 2026-06-19 · Branch: `feat/audio-linguistic` · Status: Approved (architecture); spec under review
> A strategic re-layering: the *meaning of speech* becomes the primary signal; behavioral cues
> become the secondary, time-aligned confirmation. Local LLM (Ollama) is the content engine.

## Vision — the layering flip

We are detecting deception on a **speech basis**. The single strongest signal is the **content**
of what's said, not the behavioral cues. So:

- **Layer 1 (PRIMARY) — Content engine:** an LLM reads each answer and scores *content-pattern*
  deception markers. This drives the verdict.
- **Layer 2 (SECONDARY) — Cue engine:** the existing real-time behavioral pipeline (face + voice +
  rPPG, and later body) provides the *rhythm* and **confirms** the content verdict in time.

The current "linguistic family" (shallow lexical counts — filler ratio, pronoun rate) is a weak
proxy and is **superseded** as the primary content signal by this engine (the lexical cues remain a
fast secondary cue, not the content verdict).

## Honest boundary (LOCKED)

We build **(A) content-pattern analysis**, NOT factual truth-checking:
- **Consistency** — does the account contradict itself / earlier answers?
- **Reality-Monitoring richness (CBCA/RM)** — sensory/contextual/peripheral detail, spontaneous
  self-corrections, complications (truthful accounts carry more; fabricated ones are thinner).
- **Verifiability** — checkable names/places/times the account offers.
- **Evasion/relevance** — does the answer actually address the question?

We do **NOT** claim to know if a statement is factually true about the world (no external fact base).
Output stays honest: **"deception-pattern risk,"** never "LIE." Statuses and the earned bell carry
over from the cue system.

## Interaction model — Q&A interview (primary)

- The unit of analysis is a **turn**: `{question, answer_transcript, [t0, t1]}`.
- An operator advances **question to question** (interview). Questions come from a script/list (and,
  for dev/calibration, the reading scripts below). Free-monologue auto-segmentation is a later mode.
- The question gives the LLM the **context** to judge the answer (relevance, evasion, consistency) —
  this is where content analysis is strongest.

## Two engines, time-aligned (loosely coupled, non-blocking)

```
Browser (fast): face + (later) body + audio + transcript ──► WS ──►┐
                                                                    ▼
  CUE ENGINE (fast, ~10-30 Hz, EXISTING):
    per-frame cues → synchrony → mixer timeline → live consensus
    keeps a rolling TIMELINE buffer of cue events keyed by timestamp   ← the "rhythm"
                                                                    ▲ pull [t0,t1]
  CONTENT ENGINE (slow, per-answer, async, NEW):
    answer transcript [t0,t1] → ContentJudge (Ollama) → content verdict + flagged phrases
    → PULLS the cue timeline for [t0,t1] → FUSE: content = PRIMARY, cue-rhythm = CONFIRM
```

- The cue engine **never blocks** on the LLM; it runs free to preserve the real-time rhythm.
- When an answer completes (turn boundary), the content engine judges it, then **pulls the cue
  activity from exactly that answer's time window** and fuses — the honest "context-aware" move:
  *"while they said this evasive sentence, what were the face/voice/body doing at that moment?"*
- **Fusion (content-primary):**
  - Content + cue-rhythm both flag the same turn → **high-confidence convergence**.
  - Content only → "account thin / inconsistent / unverifiable."
  - Cue only → "behavioral arousal, content fine."
  - Never binary LIE; high-confidence requires alignment (two-gate spirit preserved).

**Topology:** one Python server, two concurrent flows — the fast frame handler (cue engine) and an
**async per-turn worker** (content engine) calling **Ollama as a separate local model server**
(`localhost:11434`), sharing the session's timeline buffer.

## ContentJudge seam

```
ContentJudge.judge(question, answer, history, baseline) -> ContentVerdict {
    risk: 0..1,
    scores: { consistency, richness_rm, verifiability, relevance },
    flagged_phrases: [ {text, reason} ],
    rationale: str,
}
```

- **OllamaContentJudge** (first adapter): calls `localhost:11434`, a **small quantized model**
  (default `llama3.2:3b` or `qwen2.5:3b`, Q4 ~2 GB — fits the 8 GB budget alongside Chrome + engine),
  with a tight rubric prompt + few-shot examples, requesting **structured JSON** output. Robust JSON
  parsing with a safe fallback.
- **Graceful degradation:** if Ollama isn't installed/running, the content layer reports **offline**;
  the system falls back to the cue engine and says so honestly. (Lets us build + test the layer now;
  it lights up once Ollama is installed.)
- **Swap seam:** a `ClaudeContentJudge` adapter can drop in later for sharper judgment — no engine
  changes (mirrors the browser `Transcriber` seam pattern).
- Honest tradeoff documented: a 3B local model is solid at *structured rubric scoring* but less sharp
  than a frontier model.

## Calibration — reading phase

- During calibration, present **neutral text for the subject to read aloud**. This:
  - feeds the existing per-channel hard-gate baseline (voice, cue rhythm), AND
  - establishes a **content/voice baseline** of their truthful narrative style.
- Honest caveat baked into the UI: a person's *reading* voice/cadence differs from *spontaneous*
  speech, so the reading baseline is an approximation.

## True/False dev scripts (development validation)

- Ship a **TRUE narrative** (concrete, verifiable, internally consistent) and a **FALSE narrative**
  (vague, unverifiable, internally contradictory) as dev assets.
- Dev loop: read each; confirm the content engine scores the FALSE one worse on
  consistency/verifiability/richness.
- **Honest caveat (critical):** *reading* a false script is not the cognitive act of *spontaneous
  lying* — so this validates the **content analyzer's mechanics**, not real-world deception accuracy,
  and the **behavioral cues are not meaningful while reading** (no fabrication load). The scripts are
  a development tool, not an accuracy claim.

## Testing (TDD)

- **ContentJudge against a deterministic stub adapter** — tests never depend on a live LLM. The stub
  returns canned verdicts so we can test turn segmentation, the time-aligned cue pull, fusion
  (content-primary), and graceful degradation deterministically.
- **OllamaContentJudge**: unit-test the prompt assembly + JSON parsing (incl. malformed-output
  fallback) against recorded fixtures, not a live server.
- **True/False discrimination**: an integration test with a stub judge proving the fusion ranks a
  contradictory/unverifiable answer above a concrete one.
- `python3 -m pytest -q` + `python3 -m ruff check .` stay green.

## Roadmap / decomposition

- **Phase 1 (this spec):** Content engine (Q&A turns + Ollama `ContentJudge` seam + time-aligned
  content-primary fusion) + calibration reading phase + True/False dev scripts + graceful degradation.
- **Phase 2:** **Body family** (MediaPipe Pose/Holistic — torso/hands/neck) added to the cue engine
  so the "rhythm" includes upper body, per the operator's intent. Low science weight (gross-body cues
  are weak), high-precision emblem slips only.
- **Later:** `ClaudeContentJudge` adapter; free-monologue auto-segmentation; cross-modal coherence.

## Out of scope

Factual truth-checking; cloud LLMs in Phase 1; thermal/hardware; the deferred deep-audio
(Parselmouth) family.
