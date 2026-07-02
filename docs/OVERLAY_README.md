# Live Consensus Overlay — Quickstart

Real-time "AI-vision" deception **overlay** that runs entirely on your machine. The browser
captures your webcam + mic and runs MediaPipe, rPPG sampling, audio feature extraction, and
(optionally) live transcription; a local Python engine detects **40 cues across 4 voting
families** (29 visual · 3 audio · 7 linguistic · 1 physio), fuses them by family with a
two-gate consensus, and draws a telestrator, consensus panel, deforming **enneagram** (family
view), and radial **Cue Polygon** (per-cue synchrony view). **Raw video never leaves your
device** — only tiny feature vectors reach the engine over a localhost WebSocket. GitHub
stores the code and runs CI; it never runs the live app.

On top of the cue engine sits an optional **content engine**: a local Ollama LLM judges each
Q&A answer for content-pattern markers (consistency, richness, verifiability, evasion) and
fuses that with the cue activity from the same time window — content-primary, cue-confirm.
Enable with `BLITZ_OVERLAY_CONTENT=ollama` (requires `ollama` + a small model like
`llama3.2:3b`); without it the overlay degrades gracefully to the cue engine alone.

> **Audio caveat (Linguistic family):** the only path by which audio leaves the device is the
> optional **Linguistic transcriber** — when enabled it uses Chrome's Web Speech API, which streams
> mic audio to Google for transcription (no install/login). Video still never leaves the browser,
> and a fully-local Whisper adapter (no external call) is the documented upgrade behind the same
> `Transcriber` seam. A clear in-page notice + disable toggle appear while it is active.

> **Micro-expression caveat:** the `microexpression_burst` cue is an onset-velocity *proxy*,
> not true micro-expression recognition. Micro-expressions have low base rates and modest
> real-world effect sizes in the literature — the cue is deliberately weighted low (tier 4)
> and can never drive a FLAG on its own (the two-gate still requires a second family).

Honest framing: statuses are **CALIBRATING → CLEAR → WATCH → FLAG**, never a binary "LIE".
A red pulse fires *only* on a two-gate FLAG (≥2 independent families agree AND combined risk ≥ 0.65).

## One command

```bash
pip install -e .          # first time only (Python 3.10+)
blitz-overlay             # starts engine + browser host, opens http://127.0.0.1:8000
```

(or `python -m blitz_overlay`). Allow camera access when prompted.

- First ~90s = **CALIBRATING** (hard-gated: won't complete until every producing cue has
  enough baseline samples; the calibration card shows per-channel progress and guidance).
- Then **CLEAR/WATCH**; a **FLAG** needs any **2 independent families** to agree (Visual,
  Audio, Linguistic, Physio are all live voters).
- The **sensitivity slider** moves only the bell/burst operating point (K, lit-z, risk
  floor) — the science weights never move.

## Config (optional)

Copy `.env.example` to `.env` to change the port, gate threshold, or baseline length:

```bash
cp .env.example .env
```

No API keys are needed — all fusion is local math.

## Tests

```bash
pip install -e ".[dev]"
python -m ruff check .
python -m pytest -q
```

Includes a deterministic replay test that drives the engine to a two-gate FLAG from a
synthetic feature stream — no camera required.

## Privacy

The browser samples only blendshape coefficients, head pose, a few landmark-derived scalars,
and mean ROI colors for rPPG. The engine logs only derived decisions (status, posterior,
per-family contributions, cue z-scores) to `logs/` — never raw biometric. `logs/` and `.env`
are gitignored.

## Tuning the demo

The full science-default baseline is 90s. To reach a FLAG faster while testing, set a shorter
baseline in `.env` (e.g. `BLITZ_OVERLAY_BASELINE_SECONDS=30`), then deliberately produce a
sustained gaze aversion plus elevated heart rate so both families agree under the two-gate rule.
