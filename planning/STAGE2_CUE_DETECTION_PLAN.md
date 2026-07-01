# Stage 2 — Cue Detection Upgrade (detection first, NOT visualization)

> 2026-06-23 · Deepen the DETECTION layer before any STAGE2_3D_OVERLAY work.
> Gate check: `python3 planning/stage2_cue_detection.py --check` · Reference: `stage2_cue_detection_reference/`

## Rule

**Do NOT start `STAGE2_3D_OVERLAY` until the 2A-3 gate passes** (≥ 12 cues across ≥ 3 independent
families). The 3D overlay is a *visualization* — it adds zero detection power on its own (see
`docs/research/2026-06-23-3d-reconstruction-vision.md`). Detection comes first.

## Gate status (run the check, don't trust this number — re-run it)

As of 2026-06-23 the gate **PASSES**: **32 cues across 4 families** (visual 21 · linguistic 7 ·
audio 3 · physio 1). So the 3D overlay is technically *unblocked*.

**Honest caveat — the gate is a shallow metric.** 21 of 32 cues are *visual* (the same face,
heavily correlated) and physio is a single finicky rPPG cue. "Robust detection" is about
**independent evidence**, not raw count. So Stage 2's real target is a **new independent channel**,
not a 22nd face cue.

## Stage-2 work (priority order)

### 2A — Body family (the real upgrade: a 5th INDEPENDENT family) ⭐
MediaPipe Pose/Holistic (33 body keypoints), browser-native, **low weight** (gross-body cues are
weak in the literature — keep them honest). Detail in `stage2_cue_detection_reference/body_family_cues.md`.
- **2A-1 — browser Pose extraction:** add MediaPipe `PoseLandmarker` (Lite) → 33 keypoints in the
  `FeatureFrame` (new `pose` block). Independent try/catch like the mic — a failed Pose model must
  never kill the face overlay. Honest memory note: Pose adds browser CPU/RAM on the 8 GB M1 —
  measure; if heavy, throttle Pose to ~10 fps.
- **2A-2 — Python body cue detectors + weights + TDD:** `blitz_overlay/cues/body.py` —
  `body.postural_shift` (#19), `body.self_adaptor` (#15/#59 hand-to-face/neck), `body.shoulder_shrug`
  (#57, rare/high-precision), `body.reduced_gestures` (#18, wrist-velocity DROP). Add a `body` family
  to `core/fusion` `_FAMILY_OF` + a `Modality.BODY`; new voter row + polygon arc `B1…Bn` + enneagram
  point. Science-weighted with catalog citations; never learned.
- **2A-3 — GATE:** re-run `--check`. Target after Body = **5 independent families** (V·A·L·P·Body) —
  genuinely robust, not just count-padded. This is the gate that unblocks `STAGE2_3D_OVERLAY`.

### 2B — Tier-1.5 cues from the 3D mesh we already have (cheap, no new model)
Use MediaPipe's existing 3D output as 3D: per-landmark **depth (z) motion**, full **3D head-pose
dynamics** (we only use yaw/pitch/roll for one cue), **3D facial asymmetry**, depth micro-tremor.
More visual cues, no RAM. (See the 3D chapter.)

### 2C — Cue quality / decorrelation pass (honesty work)
Audit the 21 visual cues for redundancy; confirm the fusion's within-family decorrelation handles
them; quality-gate weak cues. Goal: independent, validated evidence — not noise padding.

## Deferred to "after more RAM" (Tier-3+)
EMOCA/SPARK 3D expression · OpenGraphAU 41 AUs · Parselmouth audio (jitter/shimmer/HNR/VOT) ·
MMPose · RF-DETR object-manipulation cue. All heavy Python/GPU models that fight the 8 GB budget.

## Locked constraints (unchanged)
Science-driven weights (never learned) · honest framing (deception-pattern risk, never "LIE") ·
browser-does-signal / feature vectors over WS / privacy-by-design · each cue subclasses
`CueDetector` + uses the rolling baseline · body cues stay LOW weight.

## How to verify
- `python3 planning/stage2_cue_detection.py --check` → gate PASS/FAIL + per-family breakdown.
- `python3 -m pytest -q` and `python3 -m ruff check .` stay green.
