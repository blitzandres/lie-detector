"""Deterministic synthetic feature stream for the replay test (spec §10).

Phases over ~65s at 30fps:
  - 40s calm baseline (centered gaze, ~72 bpm)        -> CALIBRATING then CLEAR
  - 25s stress onset (sustained gaze aversion + ~104 bpm) -> WATCH then FLAG (two families)
No randomness: every value is a deterministic function of frame index.
"""
from __future__ import annotations

import math

FPS = 30
DT_MS = int(1000 / FPS)


def _rgb(ts_ms: int, bpm: float) -> list[float]:
    f = bpm / 60.0
    t = ts_ms / 1000.0
    green = 120 + 8 * math.sin(2 * math.pi * f * t)
    return [180.0, green, 110.0]


def _frame(idx: int, *, gaze: float, bpm: float) -> dict:
    ts = idx * DT_MS
    jitter = 0.01 * math.sin(idx * 0.7)  # deterministic micro-jitter so baseline MAD != 0
    return {
        "ts": ts,
        "face_present": True,
        "confidence": 0.92,
        "blendshapes": {
            "eyeBlinkLeft": 0.05, "eyeBlinkRight": 0.05,
            "browInnerUp": 0.05 + abs(jitter), "browDownLeft": 0.04, "browDownRight": 0.04,
            "mouthPressLeft": 0.05, "mouthPressRight": 0.05, "mouthPucker": 0.03,
        },
        "geometry": {"gaze_x": gaze + jitter, "gaze_y": 0.0, "jaw_width_ratio": 0.80 + jitter},
        "head_pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
        "rppg": {"forehead_rgb": _rgb(ts, bpm), "cheek_rgb": _rgb(ts, bpm)},
    }


def replay_frames() -> list[dict]:
    frames: list[dict] = []
    idx = 0
    for _ in range(FPS * 40):                    # Phase A: 40s calm
        frames.append(_frame(idx, gaze=0.02, bpm=72))
        idx += 1
    for _ in range(FPS * 25):                    # Phase B: 25s sustained stress
        frames.append(_frame(idx, gaze=0.75, bpm=104))
        idx += 1
    return frames
