"""OllamaContentJudge — the local-LLM content adapter (localhost:11434).

The HTTP call is injected (`call`) so tests never need a live server. On any connection
error the verdict is `available=False` (the layer is honestly offline). The model is asked
for STRICT JSON; parsing is robust to prose wrappers and falls back to a low-confidence
available verdict on garbage.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable

from blitz_overlay.content.judge import ContentJudge, ContentVerdict

OLLAMA_URL = "http://localhost:11434/api/generate"

_RUBRIC = """You are a deception-PATTERN analyst. You do NOT decide if a statement is factually
true. You score only content patterns associated with deception, on a 0..1 scale each:
- consistency: 1.0 = no internal contradiction, 0.0 = contradicts itself / the question.
- richness_rm: 1.0 = rich sensory/contextual/peripheral detail (Reality Monitoring), 0.0 = thin.
- verifiability: 1.0 = many checkable names/places/times, 0.0 = vague/unverifiable.
- relevance: 1.0 = directly answers the question, 0.0 = evasive/off-topic.
Then risk: 0..1 overall deception-pattern risk (high = thin, inconsistent, unverifiable, evasive).
Reply with STRICT JSON only:
{"risk":0.0,"scores":{"consistency":0.0,"richness_rm":0.0,"verifiability":0.0,"relevance":0.0},
"flagged_phrases":[{"text":"...","reason":"..."}],"rationale":"one sentence"}"""


def build_prompt(question: str, answer: str, history: list[dict]) -> str:
    hist = "\n".join(f"Q: {h.get('question','')}\nA: {h.get('answer','')}" for h in (history or []))
    hist_block = f"\nEarlier in this interview:\n{hist}\n" if hist else ""
    return f"{_RUBRIC}\n{hist_block}\nQUESTION: {question}\nANSWER: {answer}\n\nJSON:"


def parse_verdict_json(raw: str) -> ContentVerdict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            scores = {k: float(v) for k, v in (data.get("scores") or {}).items()}
            return ContentVerdict(
                risk=float(data.get("risk", 0.0)),
                scores=scores,
                flagged_phrases=list(data.get("flagged_phrases") or []),
                rationale=str(data.get("rationale", "")),
                available=True,
            )
        except (ValueError, TypeError):
            pass
    return ContentVerdict(risk=0.0, scores={}, flagged_phrases=[],
                          rationale="unparseable model output", available=True)


def _http_call(model: str, prompt: str, timeout: float) -> str:
    import httpx
    resp = httpx.post(OLLAMA_URL,
                      json={"model": model, "prompt": prompt, "stream": False,
                            "options": {"temperature": 0.0}},
                      timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("response", "")


class OllamaContentJudge(ContentJudge):
    def __init__(self, model: str = "llama3.2:3b", timeout: float = 30.0,
                 call: Callable[[str], str] | None = None):
        self.model = model
        self.timeout = timeout
        self._call = call or (lambda prompt: _http_call(self.model, prompt, self.timeout))

    def judge(self, question: str, answer: str, history: list[dict], baseline) -> ContentVerdict:
        prompt = build_prompt(question, answer, history)
        try:
            raw = self._call(prompt)
        except Exception as err:  # noqa: BLE001 — any transport failure => honest offline
            return ContentVerdict.offline(f"content layer offline (ollama): {err}")
        return parse_verdict_json(raw)
