# Reference — Body Family cues (Stage 2A)

> The 5th independent family. MediaPipe Pose (33 keypoints), browser-native, **LOW weight**.
> Source evidence: `docs/CUE_CATALOG.md` (#14–20, #57–59). Gross-body cues are weak — honest.

## Why Body is the right Stage-2 upgrade
It's a **new independent channel** (separate from the face), so it genuinely strengthens the
two-gate / convergence (a real 5th family) — unlike adding a 22nd correlated face cue. But the
literature is clear that gross-body cues are weak, so every Body cue stays **low weight** and one
(shoulder shrug) is rare/high-precision.

## MediaPipe Pose keypoints we need (33-landmark model)
nose(0) · shoulders L/R(11,12) · elbows(13,14) · wrists(15,16) · hips L/R(23,24). Normalized x,y(,z)
per frame. Browser: `PoseLandmarker` (Lite) in `apps/overlay-web/` → a `pose` block in the FeatureFrame.

## Cue specs (catalog-grounded, low weight)

| cue_id | catalog | signal | how (Pose) | dir | d / tier |
|---|---|---|---|---|---|
| `body.postural_shift` | #19 | discomfort, wanting to exit | shoulder/hip-midpoint movement over ~3 s window | +1 | 0.40 / 3 |
| `body.self_adaptor` | #15/#59 | self-soothing under stress | wrist→face/neck proximity *duration* (hand near nose/shoulder region) | +1 | 0.50 / 2 |
| `body.shoulder_shrug` | #57 | suppressed "I don't know" emblem slip | bilateral shoulder-Y raise vs baseline (rare, ~85% precision when it fires) | +1 | 0.55 / 2 |
| `body.reduced_gestures` | #18 | liars gesture LESS | wrist-velocity over ~3 s window — **low** value is suspicious | -1 | 0.45 / 3 |

All person-relative via the rolling baseline. Each subclasses `CueDetector`. `body.reduced_gestures`
uses `direction = -1` (a *drop* below the person's baseline gesture rate is the suspicious side).

## Engine wiring checklist (2A-2)
- `core/schemas/cue_event.py`: add `Modality.BODY`.
- `core/fusion/bayesian_fusion.py` `_FAMILY_OF`: map `Modality.BODY → "body"`.
- `blitz_overlay/cues/base.py` `_MODALITY`: add `"body": Modality.BODY`.
- `blitz_overlay/weights.py`: 4 `body.*` weights + catalog citations.
- `blitz_overlay/cues/body.py`: 4 detectors + `BODY_DETECTORS`.
- `blitz_overlay/pipeline.py`: append `BODY_DETECTORS`; read `frame.pose`.
- `blitz_overlay/schemas.py`: add `pose` block to `FeatureFrame`.
- `consensus.py` `PANEL_FAMILIES` / `WIRED_FAMILIES`: add `"body"`.
- Browser: `pose-extractor.js` (PoseLandmarker) → attach `frame.pose`; polygon shows `B1…B4`.
- TDD throughout; `pytest` + `ruff` green. Re-run `--check` → expect **5 families**.

## Honest notes
- Adds the Pose model → more browser CPU/RAM on 8 GB; throttle Pose to ~10 fps if heavy.
- Body cues NEVER dominate — low weight by design.
- Shoulder shrug is rare; treat a fire as high-precision, not high-recall.
