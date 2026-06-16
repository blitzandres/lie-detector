"""First-pass linguistic cue extractor for the Blitz Engine MVP."""

from __future__ import annotations

import re
from collections import Counter

from core.schemas.cue_event import CueEvent, Modality, Phase

TOKEN_RE = re.compile(r"[A-Za-z']+")

FILLERS = {
    "uh", "um", "like", "you", "know", "actually", "basically", "literally",
}
FIRST_PERSON = {"i", "me", "my", "mine", "myself"}
DISTANCING = {"that", "someone", "somebody", "person", "individual", "they", "them"}
QUALIFIERS = {"honestly", "really", "basically", "literally", "to", "be", "fair", "frankly"}
NEGATIVE_WORDS = {"no", "not", "never", "nothing", "nobody", "can't", "won't", "fear", "worried"}
SENSORY_WORDS = {"saw", "heard", "felt", "blue", "red", "loud", "quiet", "cold", "warm", "bright"}


class LinguisticAnalyzer:
    """Extracts a small set of text-only cues without external model dependencies."""

    cue_specs = {
        "linguistic.filler_ratio": {"effect_size_d": 0.23, "reliability_tier": 3},
        "linguistic.pronoun_avoidance": {"effect_size_d": 0.27, "reliability_tier": 2},
        "linguistic.distancing_language": {"effect_size_d": 0.24, "reliability_tier": 2},
        "linguistic.qualifier_overload": {"effect_size_d": 0.21, "reliability_tier": 3},
        "linguistic.negative_emotion_density": {"effect_size_d": 0.18, "reliability_tier": 3},
        "linguistic.sensory_detail_poverty": {"effect_size_d": 0.29, "reliability_tier": 2},
        "linguistic.lexical_diversity_drop": {"effect_size_d": 0.16, "reliability_tier": 3},
        "linguistic.response_delay_ms": {"effect_size_d": 0.26, "reliability_tier": 2},
    }

    def tokenize(self, text: str) -> list[str]:
        return TOKEN_RE.findall(text.lower())

    def extract_features(self, text: str, response_latency_ms: int | None = None) -> dict[str, float]:
        tokens = self.tokenize(text)
        counts = Counter(tokens)
        token_count = max(1, len(tokens))
        unique_ratio = len(set(tokens)) / token_count

        filler_count = sum(counts[word] for word in FILLERS)
        first_person_count = sum(counts[word] for word in FIRST_PERSON)
        distancing_count = sum(counts[word] for word in DISTANCING)
        qualifier_count = sum(counts[word] for word in QUALIFIERS)
        negative_count = sum(counts[word] for word in NEGATIVE_WORDS)
        sensory_count = sum(counts[word] for word in SENSORY_WORDS)

        return {
            "linguistic.filler_ratio": filler_count / token_count,
            "linguistic.pronoun_avoidance": 1.0 - min(1.0, first_person_count / max(3, token_count * 0.08)),
            "linguistic.distancing_language": distancing_count / token_count,
            "linguistic.qualifier_overload": qualifier_count / token_count,
            "linguistic.negative_emotion_density": negative_count / token_count,
            "linguistic.sensory_detail_poverty": 1.0 - min(1.0, sensory_count / max(2, token_count * 0.05)),
            "linguistic.lexical_diversity_drop": 1.0 - unique_ratio,
            "linguistic.response_delay_ms": float(response_latency_ms or 0),
        }

    def build_baseline_observations(
        self,
        baseline_texts: list[str],
        baseline_latencies_ms: list[int] | None = None,
    ) -> dict[str, list[float]]:
        observations = {cue_id: [] for cue_id in self.cue_specs}
        latencies = baseline_latencies_ms or [0] * len(baseline_texts)

        for text, latency in zip(baseline_texts, latencies, strict=False):
            for cue_id, value in self.extract_features(text, response_latency_ms=latency).items():
                observations[cue_id].append(value)

        return observations

    def analyze(
        self,
        text: str,
        question_id: str,
        baseline,
        response_latency_ms: int | None = None,
        timestamp_ms: int = 0,
    ) -> list[CueEvent]:
        features = self.extract_features(text, response_latency_ms=response_latency_ms)
        cues: list[CueEvent] = []

        for cue_id, raw_value in features.items():
            spec = self.cue_specs[cue_id]
            z_score = baseline.normalize(cue_id, raw_value)
            quality = 0.35 if len(text.strip()) < 20 else 0.8
            if cue_id == "linguistic.response_delay_ms" and response_latency_ms is None:
                quality = 0.0

            cues.append(
                CueEvent(
                    cue_id=cue_id,
                    modality=Modality.LINGUISTIC,
                    timestamp_ms=timestamp_ms,
                    phase=Phase.RESPONSE,
                    raw_value=raw_value,
                    z_score=z_score,
                    llr=0.0,
                    quality=quality,
                    question_id=question_id,
                    effect_size_d=spec["effect_size_d"],
                    reliability_tier=spec["reliability_tier"],
                )
            )

        return cues
