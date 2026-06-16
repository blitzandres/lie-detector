"""
Blitz Engine — CueEvent Schema
The canonical internal object. All modality plugins produce CueEvents.
"""
from dataclasses import dataclass, field
from enum import Enum


class Modality(str, Enum):
    VISUAL = "visual"
    AUDIO = "audio"
    LINGUISTIC = "linguistic"
    PHYSIOLOGICAL = "physiological"
    CBCA = "cbca"


class Phase(str, Enum):
    ANTICIPATION = "anticipation"  # -2s to speech onset
    RESPONSE = "response"          # speech onset to end
    RECOVERY = "recovery"          # 0s to +4s after speech end


@dataclass
class CueEvent:
    """
    A single timestamped behavioral signal from a modality plugin.

    All values are normalized to personal baseline before entering fusion.
    """
    cue_id: str                          # e.g. "audio.vot_shortening"
    modality: Modality
    timestamp_ms: int                    # when the cue fired
    phase: Phase
    raw_value: float                     # raw extracted measurement
    z_score: float                       # robust Z vs personal baseline
    llr: float                           # log-likelihood ratio contribution
    quality: float                       # extraction confidence [0.0, 1.0]
    question_id: str                     # which question this fired on
    region: str = ""                     # telestrator anchor region (spec §5)
    on_word: str | None = None           # specific word that triggered cue
    effect_size_d: float = 0.25          # literature effect size for this cue
    reliability_tier: int = 3            # 1=strong, 2=moderate, 3=weak, 4=anchor


@dataclass
class BlitzOutput:
    """
    The final output from the fusion layer for a single analyzed clip.
    """
    session_id: str
    question_id: str
    timestamp_ms: int

    risk_score: float                    # [0.0, 1.0] posterior P(deception)
    uncertainty: float                   # half-width of 90% CI
    confidence_interval: tuple           # (low, high)

    channel_contributions: dict          # per-modality risk scores
    top_cues: list                       # top CueEvents that drove the score

    quality_flags: dict                  # video quality, baseline duration, etc.
    narrative: str                       # AI-generated explanation

    compliance: dict = field(default_factory=lambda: {
        "not_for_sole_decision": True,   # hardcoded, cannot be disabled
        "use_case": None,
        "consent_declared": False,
        "jurisdiction": None
    })
