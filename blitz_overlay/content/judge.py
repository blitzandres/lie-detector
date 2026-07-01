"""Content-analysis seam: the primary (content-first) deception layer.

A ContentJudge scores the *content pattern* of an answer — consistency, Reality-Monitoring
richness, verifiability, relevance — NEVER factual truth. StubContentJudge is a deterministic
heuristic so the rest of the system (fusion, turns, tests) never depends on a live LLM.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

_TOKEN = re.compile(r"[A-Za-z']+")
# Words that signal vagueness / hedging (content-poverty markers).
_VAGUE = {
    "someone", "something", "somewhere", "somehow", "stuff", "things", "people",
    "around", "maybe", "guess", "think", "probably", "kind", "sort", "whatever",
    "anyway", "just", "like", "you", "know",
}


@dataclass
class ContentVerdict:
    """One content judgment of an answer. risk/scores in [0,1]. available=False => layer offline."""

    risk: float
    scores: dict[str, float]
    flagged_phrases: list[dict] = field(default_factory=list)
    rationale: str = ""
    available: bool = True

    def to_dict(self) -> dict:
        return {
            "risk": round(self.risk, 4),
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "flagged_phrases": self.flagged_phrases,
            "rationale": self.rationale,
            "available": self.available,
        }

    @classmethod
    def offline(cls, reason: str) -> ContentVerdict:
        return cls(risk=0.0, scores={}, flagged_phrases=[], rationale=reason, available=False)


class ContentJudge(ABC):
    @abstractmethod
    def judge(self, question: str, answer: str, history: list[dict], baseline) -> ContentVerdict:
        """Score the content of `answer` (given `question` for context)."""


class StubContentJudge(ContentJudge):
    """Deterministic content heuristic — no network. Vagueness/hedging up, concrete detail down."""

    def judge(self, question: str, answer: str, history: list[dict], baseline) -> ContentVerdict:
        tokens = _TOKEN.findall(answer.lower())
        n = max(1, len(tokens))
        vague = sum(1 for t in tokens if t in _VAGUE)
        # concrete = numbers + capitalized proper nouns (verifiable detail)
        concrete = len(re.findall(r"\d", answer)) + sum(1 for w in _TOKEN.findall(answer) if w[:1].isupper())

        vague_ratio = vague / n
        verifiability = max(0.0, min(1.0, concrete / max(4, n * 0.15)))
        richness_rm = verifiability  # proxy: concrete detail ~ RM richness for the stub
        relevance = 1.0 if tokens else 0.0
        consistency = 1.0  # the stub has no cross-answer memory

        # risk: high when vague & low verifiability. [0,1].
        risk = max(0.0, min(1.0, 0.6 * vague_ratio + 0.5 * (1.0 - verifiability)))
        flagged = []
        for w in _TOKEN.findall(answer):
            if w.lower() in _VAGUE and len(flagged) < 5:
                flagged.append({"text": w, "reason": "vague / hedging marker"})
        return ContentVerdict(
            risk=risk,
            scores={"consistency": consistency, "richness_rm": richness_rm,
                    "verifiability": verifiability, "relevance": relevance},
            flagged_phrases=flagged,
            rationale="stub heuristic (vagueness vs concrete detail)",
            available=True,
        )
