"""Science-driven cue weights — fixed from the literature, NEVER learned (READINESS #8).

Each weight is a published effect size (Cohen's d / Hedges' g) with a traceable citation.
Changing this set requires bumping WEIGHT_SET_VERSION; the version is stamped into every
prediction-log line for auditability (READINESS #18). Corpora validate accuracy only —
they must not move these weights.
"""

from __future__ import annotations

WEIGHT_SET_VERSION = "stage1-2026-06-16"

# cue_id -> {effect_size_d, reliability_tier(1 strong..4 anchor), family, region, citation}
CUE_WEIGHTS: dict[str, dict] = {
    "visual.gaze_aversion": {
        "effect_size_d": 0.70,
        "reliability_tier": 2,
        "family": "visual",
        "region": "eyes",
        "citation": (
            "Catalog cue 58 — sustained gaze aversion duration, d~0.6-0.8 high-stakes (CUE_CATALOG.md)."
        ),
    },
    "visual.blink_rate": {
        "effect_size_d": 0.40,
        "reliability_tier": 2,
        "family": "visual",
        "region": "eyes",
        "citation": (
            "Catalog cue 60 — blink suppression→rebound; deviation-from-baseline, pattern>static rate."
        ),
    },
    "visual.brow_flash": {
        "effect_size_d": 0.30,
        "reliability_tier": 3,
        "family": "visual",
        "region": "brow",
        "citation": (
            "Catalog cue 9 — AU1/2/4 brow movement; weak-moderate single-cue diagnosticity."
        ),
    },
    "visual.lip_press": {
        "effect_size_d": 0.30,
        "reliability_tier": 3,
        "family": "visual",
        "region": "mouth",
        "citation": (
            "Catalog cue 3 — lip compression (AU23/24), withholding; weak-moderate."
        ),
    },
    "visual.jaw_tension": {
        "effect_size_d": 0.28,
        "reliability_tier": 3,
        "family": "visual",
        "region": "jaw",
        "citation": (
            "Catalog cue 8 — jaw tension (AU28) via MediaPipe landmark-distance proxy; resolves Blocker 2."
        ),
    },
    "physio.heart_rate": {
        "effect_size_d": 0.50,
        "reliability_tier": 2,
        "family": "physio",
        "region": "forehead",
        "citation": (
            "Catalog cue 38 — rPPG heart-rate elevation, autonomic arousal proxy."
        ),
    },
    "audio.tremor": {
        "effect_size_d": 0.40,
        "reliability_tier": 2,
        "family": "audio",
        "region": "head",
        "citation": (
            "Catalog cue 22 — vocal tremor/jitter; strongest single audio cue for"
            " stress/cognitive load; d~0.4 moderate effect (CUE_CATALOG.md)."
        ),
    },
    "audio.pitch_f0": {
        "effect_size_d": 0.30,
        "reliability_tier": 3,
        "family": "audio",
        "region": "head",
        "citation": (
            "Catalog cue 21 — F0 elevation under stress/cognitive load;"
            " weak-moderate single-cue diagnosticity, d~0.3."
        ),
    },
    "audio.pause_ratio": {
        "effect_size_d": 0.30,
        "reliability_tier": 3,
        "family": "audio",
        "region": "head",
        "citation": (
            "Catalog cue 26 — pause duration/ratio; increased pausing under"
            " cognitive load or withholding; weak-moderate, d~0.3."
        ),
    },
}


def weight_for(cue_id: str) -> dict:
    return CUE_WEIGHTS[cue_id]
