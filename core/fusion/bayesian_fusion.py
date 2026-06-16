"""Bayesian log-odds fusion for normalized cue streams."""

import math

from core.schemas.cue_event import CueEvent

# Default prior: 30% base rate of deceptive responses (conservative)
DEFAULT_PRIOR = 0.30

# Phase multipliers for temporal weighting
PHASE_MULTIPLIERS = {
    "anticipation": 1.2,
    "response": 1.5,
    "recovery": 1.0
}

# Reliability tier weights
TIER_WEIGHTS = {1: 1.5, 2: 1.0, 3: 0.6, 4: 0.3}

# Correlated cue families (prevent double-counting)
CORRELATION_GROUPS = {
    "laryngeal_stress": ["audio.pitch_f0", "audio.jitter", "audio.shimmer",
                         "audio.hnr_drop", "audio.vot_shortening", "audio.formant_dispersion"],
    "autonomic_arousal": ["physiological.heart_rate", "physiological.eda_proxy",
                          "physiological.multi_roi_divergence"],
    "cognitive_load":    ["audio.speech_rate", "audio.pause_duration", "audio.fillers",
                          "linguistic.syntactic_depth"],
    "social_distance":   ["linguistic.pronoun_avoidance", "linguistic.distancing_language",
                          "visual.gaze_aversion"],
}


def logit(p: float) -> float:
    """Convert probability to log-odds."""
    p = max(1e-6, min(1 - 1e-6, p))
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    """Convert log-odds to probability."""
    return 1 / (1 + math.exp(-x))


def compute_llr(cue: CueEvent) -> float:
    """
    Compute log-likelihood ratio for a single cue.
    LLR_i ≈ d_i × z_i - (d_i² / 2)
    """
    d = cue.effect_size_d
    z = cue.z_score
    return d * z - (d ** 2 / 2)


def cue_weight(cue: CueEvent) -> float:
    """Combine extraction quality, temporal phase, and literature tier."""
    phase_weight = PHASE_MULTIPLIERS.get(cue.phase.value, 1.0)
    tier_weight = TIER_WEIGHTS.get(cue.reliability_tier, 0.5)
    return max(0.0, cue.quality) * phase_weight * tier_weight


def group_penalty(cue: CueEvent) -> float:
    """Down-weight correlated cues that belong to the same evidence family."""
    for members in CORRELATION_GROUPS.values():
        if cue.cue_id in members:
            return 0.7
    return 1.0


def estimate_uncertainty(cues: list[CueEvent]) -> float:
    """Return a simple 90% CI half-width proxy from cue count and quality."""
    if not cues:
        return 0.45

    quality_sum = sum(max(cue.quality, 0.05) for cue in cues)
    active = max(1.0, quality_sum)
    return min(0.45, max(0.08, 0.32 / math.sqrt(active)))


def fuse(cues: list[CueEvent], prior: float = DEFAULT_PRIOR) -> dict:
    """Fuse cues into a posterior probability and explanation payload."""
    posterior_log_odds = logit(prior)
    channel_contributions: dict[str, float] = {}
    scored_cues: list[CueEvent] = []

    for cue in cues:
        cue.llr = compute_llr(cue)
        contribution = cue.llr * cue_weight(cue) * group_penalty(cue)
        posterior_log_odds += contribution
        channel_contributions.setdefault(cue.modality.value, 0.0)
        channel_contributions[cue.modality.value] += contribution
        scored_cues.append(cue)

    posterior = sigmoid(posterior_log_odds)
    uncertainty = estimate_uncertainty(scored_cues)
    confidence_interval = (
        max(0.0, posterior - uncertainty),
        min(1.0, posterior + uncertainty),
    )

    return {
        "posterior": posterior,
        "posterior_log_odds": posterior_log_odds,
        "channel_contributions": channel_contributions,
        "top_cues": sorted(scored_cues, key=lambda cue: abs(cue.llr), reverse=True)[:5],
        "uncertainty": uncertainty,
        "confidence_interval": confidence_interval,
    }


def convergence_gate_passed(cues: list[CueEvent], threshold: float = 0.65,
                             posterior: float = 0.0) -> bool:
    """
    Two-gate convergence check:
    Gate 1: Posterior probability > threshold
    Gate 2: At least 2 independent modality families active
    Both must pass.
    """
    if posterior < threshold:
        return False

    active_modalities = {cue.modality for cue in cues if cue.z_score > 1.0}
    return len(active_modalities) >= 2
