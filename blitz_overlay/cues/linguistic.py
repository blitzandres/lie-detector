"""Linguistic cue detectors reading a browser-supplied transcript window.

Each detector reads frame.transcript {"text", "seq"} and computes one lexicon feature by
reusing modalities/linguistic/analyzer.py (single source of truth for the word-lists and
feature math). When frame.transcript is absent (mic/recognizer off) or the window is too
short, measure() returns None and the linguistic family is simply absent from consensus.

Person-relative via the rolling baseline (catalog: linguistic cues are baseline-relative).
Honest framing: Web Speech transcription is a browser approximation; WhisperX (fully local)
is the documented upgrade behind the browser Transcriber seam.
"""
from __future__ import annotations

from blitz_overlay.cues.base import CueDetector
from blitz_overlay.schemas import FeatureFrame
from modalities.linguistic.analyzer import LinguisticAnalyzer

_ANALYZER = LinguisticAnalyzer()
MIN_WORDS = 5  # below this a window is too short to score reliably


class LinguisticCue(CueDetector):
    """Base for transcript-window lexicon cues. Subclasses set cue_id only.

    direction +1: the lexicon feature is defined so that the suspicious pattern is the
    HIGH direction (more fillers / more distancing / less first-person / less sensory detail).
    """

    direction = 1

    def measure(self, frame: FeatureFrame) -> float | None:
        t = frame.transcript
        if not t:
            return None
        text = t.get("text") or ""
        if len(_ANALYZER.tokenize(text)) < MIN_WORDS:
            return None
        return float(_ANALYZER.extract_features(text)[self.cue_id])

    def quality(self, frame: FeatureFrame) -> float:
        """Confidence scales with window length — short windows are noisy."""
        t = frame.transcript or {}
        n = len(_ANALYZER.tokenize(t.get("text") or ""))
        return max(0.0, min(1.0, n / 15.0))


class SensoryDetailPoverty(LinguisticCue):
    cue_id = "linguistic.sensory_detail_poverty"


class PronounAvoidance(LinguisticCue):
    cue_id = "linguistic.pronoun_avoidance"


class DistancingLanguage(LinguisticCue):
    cue_id = "linguistic.distancing_language"


class FillerRatio(LinguisticCue):
    cue_id = "linguistic.filler_ratio"


class QualifierOverload(LinguisticCue):
    cue_id = "linguistic.qualifier_overload"


class NegativeEmotionDensity(LinguisticCue):
    cue_id = "linguistic.negative_emotion_density"


class LexicalDiversityDrop(LinguisticCue):
    cue_id = "linguistic.lexical_diversity_drop"


LINGUISTIC_DETECTORS = [
    SensoryDetailPoverty, PronounAvoidance, DistancingLanguage, FillerRatio,
    QualifierOverload, NegativeEmotionDensity, LexicalDiversityDrop,
]
