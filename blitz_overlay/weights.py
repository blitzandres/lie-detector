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
    "linguistic.sensory_detail_poverty": {
        "effect_size_d": 0.29, "reliability_tier": 2,
        "family": "linguistic", "region": "mouth",
        "citation": (
            "Catalog cue — Reality Monitoring sensory-detail poverty; truthful accounts carry"
            " more perceptual detail (RESEARCH.md RM/CBCA); person-relative, d~0.29."
        ),
    },
    "linguistic.pronoun_avoidance": {
        "effect_size_d": 0.27, "reliability_tier": 2,
        "family": "linguistic", "region": "mouth",
        "citation": (
            "Catalog cue — reduced first-person pronoun use / self-distancing under deception"
            " (Newman/Pennebaker LIWC; RESEARCH.md), d~0.27."
        ),
    },
    "linguistic.distancing_language": {
        "effect_size_d": 0.24, "reliability_tier": 2,
        "family": "linguistic", "region": "mouth",
        "citation": (
            "Catalog cue — distancing / third-person referents replacing direct reference"
            " (RESEARCH.md distancing language), d~0.24."
        ),
    },
    "linguistic.filler_ratio": {
        "effect_size_d": 0.23, "reliability_tier": 3,
        "family": "linguistic", "region": "mouth",
        "citation": (
            "Catalog cue — filler / hesitation marker rate as cognitive-load proxy"
            " (RESEARCH.md), weak-moderate d~0.23."
        ),
    },
    "linguistic.qualifier_overload": {
        "effect_size_d": 0.21, "reliability_tier": 3,
        "family": "linguistic", "region": "mouth",
        "citation": (
            "Catalog cue — qualifier / hedge overload signalling uncommitted speech"
            " (RESEARCH.md), weak d~0.21."
        ),
    },
    "linguistic.negative_emotion_density": {
        "effect_size_d": 0.18, "reliability_tier": 3,
        "family": "linguistic", "region": "mouth",
        "citation": (
            "Catalog cue — elevated negative-emotion word density (LIWC; RESEARCH.md),"
            " weak d~0.18."
        ),
    },
    "linguistic.lexical_diversity_drop": {
        "effect_size_d": 0.16, "reliability_tier": 3,
        "family": "linguistic", "region": "mouth",
        "citation": (
            "Catalog cue — reduced lexical diversity (type-token ratio) under cognitive load"
            " (RESEARCH.md), weak d~0.16 (anchor-tier)."
        ),
    },
}


def weight_for(cue_id: str) -> dict:
    return CUE_WEIGHTS[cue_id]
