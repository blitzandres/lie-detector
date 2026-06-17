# Live Consensus Overlay — Quickstart (Stage 1, webcam-only)

Real-time "AI-vision" deception **overlay** that runs entirely on your machine. The browser
captures your webcam and runs MediaPipe + rPPG; a local Python engine detects visual cues +
heart rate, fuses them by family with a two-gate consensus, and draws a telestrator + consensus
panel. **Raw video never leaves your device** — only tiny feature vectors reach the engine over
a localhost WebSocket. GitHub stores the code and runs CI; it never runs the live app.

> **Audio caveat (Linguistic family):** the only path by which audio leaves the device is the
> optional **Linguistic transcriber** — when enabled it uses Chrome's Web Speech API, which streams
> mic audio to Google for transcription (no install/login). Video still never leaves the browser,
> and a fully-local Whisper adapter (no external call) is the documented upgrade behind the same
> `Transcriber` seam. A clear in-page notice + disable toggle appear while it is active.

Honest framing: statuses are **CALIBRATING → CLEAR → WATCH → FLAG**, never a binary "LIE".
A red pulse fires *only* on a two-gate FLAG (≥2 independent families agree AND combined risk ≥ 0.65).

## One command

```bash
pip install -e .          # first time only (Python 3.10+)
blitz-overlay             # starts engine + browser host, opens http://127.0.0.1:8000
```

(or `python -m blitz_overlay`). Allow camera access when prompted.

- First ~90s = **CALIBRATING** (builds your rolling baseline; no flags permitted).
- Then **CLEAR/WATCH**; a **FLAG** needs both the Visual family and the Physio (heart-rate)
  family to agree — that's why rPPG is required in Stage 1.
- Audio and Linguistic voters show "—" (not wired in Stage 1).

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
