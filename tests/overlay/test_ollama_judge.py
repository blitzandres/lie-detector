"""OllamaContentJudge: prompt assembly, JSON parsing, and offline fallback (no live server)."""
from blitz_overlay.content.judge import ContentVerdict
from blitz_overlay.content.ollama_judge import OllamaContentJudge, parse_verdict_json


def test_parse_well_formed_json():
    raw = ('{"risk":0.72,"scores":{"consistency":0.5,"richness_rm":0.3,'
           '"verifiability":0.2,"relevance":0.8},'
           '"flagged_phrases":[{"text":"some guy","reason":"vague"}],"rationale":"thin"}')
    v = parse_verdict_json(raw)
    assert isinstance(v, ContentVerdict)
    assert v.risk == 0.72
    assert v.scores["consistency"] == 0.5
    assert v.flagged_phrases[0]["text"] == "some guy"
    assert v.available is True


def test_parse_json_embedded_in_prose():
    raw = "Sure! Here is the analysis:\n{\"risk\":0.4,\"scores\":{},\"rationale\":\"ok\"}\nHope that helps."
    v = parse_verdict_json(raw)
    assert v.risk == 0.4


def test_parse_garbage_returns_low_confidence_available():
    v = parse_verdict_json("the model rambled with no json at all")
    assert v.available is True
    assert v.risk == 0.0
    assert "unparseable" in v.rationale.lower()


def test_judge_uses_injected_caller():
    captured = {}

    def fake_call(prompt: str) -> str:
        captured["prompt"] = prompt
        return '{"risk":0.9,"scores":{"consistency":0.1},"rationale":"contradictory"}'

    judge = OllamaContentJudge(model="llama3.2:3b", call=fake_call)
    v = judge.judge("Where were you?", "I was home. I was at work.", history=[], baseline=None)
    assert v.risk == 0.9
    assert "Where were you?" in captured["prompt"]
    assert "I was home" in captured["prompt"]


def test_judge_offline_when_caller_raises():
    def boom(prompt: str) -> str:
        raise ConnectionError("ollama not running")

    judge = OllamaContentJudge(model="llama3.2:3b", call=boom)
    v = judge.judge("Q?", "A.", history=[], baseline=None)
    assert v.available is False
    assert "offline" in v.rationale.lower() or "ollama" in v.rationale.lower()
