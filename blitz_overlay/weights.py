"""Science-driven cue weights — fixed from the literature, NEVER learned (READINESS #8).

Each weight is a published effect size (Cohen's d / Hedges' g) with a traceable citation.
Changing this set requires bumping WEIGHT_SET_VERSION; the version is stamped into every
prediction-log line for auditability (READINESS #18). Corpora validate accuracy only —
they must not move these weights.
"""

from __future__ import annotations

WEIGHT_SET_VERSION = "step2-2026-07-02"

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
    "visual.gaze_fixation": {
        "effect_size_d": 0.50, "reliability_tier": 2,
        "family": "visual", "region": "eyes",
        "citation": (
            "Catalog cue 56 — gaze fixation pattern (count/duration); fabrication = more"
            " frequent, shorter fixations vs recall; 70-80% with ML (CUE_CATALOG.md)."
        ),
    },
    "visual.pupil_dilation": {
        "effect_size_d": 0.40, "reliability_tier": 2,
        "family": "visual", "region": "eyes",
        "citation": (
            "Catalog cue 7/55 — pupil/iris dilation, cognitive-load spike; 65-75% alone."
            " Reliable only at 720p+, so quality is scaled down at low webcam resolution."
        ),
    },
    "visual.asymmetric_smile": {
        "effect_size_d": 0.35, "reliability_tier": 2,
        "family": "visual", "region": "mouth",
        "citation": (
            "Catalog cue 5 — smile asymmetry (Duchenne vs fake), AU6/AU12 left-right"
            " asymmetry (CUE_CATALOG.md)."
        ),
    },
    "visual.nose_wrinkle": {
        "effect_size_d": 0.28, "reliability_tier": 3,
        "family": "visual", "region": "mouth",
        "citation": (
            "Catalog cue 4 — nose wrinkle (AU9), disgust/discomfort; weak-moderate."
        ),
    },
    "visual.eye_blocking": {
        "effect_size_d": 0.28, "reliability_tier": 3,
        "family": "visual", "region": "eyes",
        "citation": (
            "Catalog cue 13 — eye blocking (prolonged eye closure while speaking),"
            " blink-duration classifier; weak-moderate."
        ),
    },
    "visual.eye_widen": {
        "effect_size_d": 0.25, "reliability_tier": 3,
        "family": "visual", "region": "eyes",
        "citation": (
            "Catalog cue 9-adjacent — eye widen (AU5, eyeWide), surprise/fear leakage;"
            " weak single-cue diagnosticity."
        ),
    },
    "visual.head_movement": {
        "effect_size_d": 0.30, "reliability_tier": 3, "family": "visual", "region": "head",
        "citation": "Catalog cue 14 — head-movement increase (restlessness/discomfort) via head-pose variance.",
    },
    "visual.eye_squint": {
        "effect_size_d": 0.25, "reliability_tier": 3, "family": "visual", "region": "eyes",
        "citation": "AU7 eye squint — tension/contempt micro-cue; weak single-cue diagnosticity.",
    },
    "visual.mouth_stretch": {
        "effect_size_d": 0.28, "reliability_tier": 3, "family": "visual", "region": "mouth",
        "citation": "AU20 lip stretch — fear/tension grimace; weak-moderate.",
    },
    "visual.mouth_frown": {
        "effect_size_d": 0.25, "reliability_tier": 3, "family": "visual", "region": "mouth",
        "citation": "AU15 lip-corner depressor — negative-affect leakage; weak.",
    },
    "visual.mouth_shrug": {
        "effect_size_d": 0.24, "reliability_tier": 3, "family": "visual", "region": "mouth",
        "citation": "AU17 chin raise / mouth shrug — doubt / uncertainty emblem; weak.",
    },
    "visual.jaw_shift": {
        "effect_size_d": 0.22, "reliability_tier": 3, "family": "visual", "region": "jaw",
        "citation": "Lateral/forward jaw displacement — jaw tension proxy; weak.",
    },
    "visual.jaw_drop": {
        "effect_size_d": 0.22, "reliability_tier": 3, "family": "visual", "region": "jaw",
        "citation": "AU26 jaw drop / mouth opening — surprise/affect; weak.",
    },
    "visual.lip_roll": {
        "effect_size_d": 0.26, "reliability_tier": 3, "family": "visual", "region": "mouth",
        "citation": "Lip suck/roll (AU28-adjacent) — withholding marker; weak-moderate.",
    },
    "visual.brow_outer_raise": {
        "effect_size_d": 0.25, "reliability_tier": 3, "family": "visual", "region": "brow",
        "citation": "AU2 outer-brow raise — surprise / overemphasis; weak.",
    },
    "visual.contempt_asymmetry": {
        "effect_size_d": 0.30, "reliability_tier": 2, "family": "visual", "region": "mouth",
        "citation": "AU14 unilateral contempt — left-right mouth-dimple asymmetry; moderate micro-expression marker.",
    },
    # Step 2b — temporal/dynamic cue expansion (spec 2026-07-02-step2-visual-deepening)
    "visual.duchenne_absence": {
        "effect_size_d": 0.35, "reliability_tier": 3, "family": "visual", "region": "mouth",
        "citation": "Ekman & Friesen — Duchenne marker: AU12 without AU6 = social/masked smile; moderate diagnosticity for masked affect (catalog cue 5 family).",
    },
    "visual.stress_brow": {
        "effect_size_d": 0.30, "reliability_tier": 3, "family": "visual", "region": "brow",
        "citation": "FACS — AU1+AU2+AU4 combination: fear/stress brow; combo more specific than single-AU brow movement (catalog cue 9 family).",
    },
    "visual.face_asymmetry": {
        "effect_size_d": 0.30, "reliability_tier": 3, "family": "visual", "region": "mouth",
        "citation": "Facial asymmetry under load — unilateral action-intensity differences; weak-moderate, person-relative baseline required (catalog cue 5/12 family).",
    },
    "visual.head_velocity": {
        "effect_size_d": 0.25, "reliability_tier": 3, "family": "visual", "region": "head",
        "citation": "Catalog cue 14 family — head-movement dynamics; velocity component, weak single-cue diagnosticity, person-relative.",
    },
    "visual.head_acceleration": {
        "effect_size_d": 0.25, "reliability_tier": 3, "family": "visual", "region": "head",
        "citation": "Catalog cue 14 family — sudden movement onsets (jerk) distinct from sustained restlessness; weak, person-relative.",
    },
    "visual.blink_duration": {
        "effect_size_d": 0.30, "reliability_tier": 3, "family": "visual", "region": "eyes",
        "citation": "Catalog cue 60 family — blink duration/rebound timing complements rate; long closures under load.",
    },
}


def weight_for(cue_id: str) -> dict:
    return CUE_WEIGHTS[cue_id]
