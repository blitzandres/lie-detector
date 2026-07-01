"""ContentVerdict shape + the deterministic StubContentJudge used everywhere in tests."""
from blitz_overlay.content.judge import ContentJudge, ContentVerdict, StubContentJudge


def test_verdict_to_dict_round_trips():
    v = ContentVerdict(
        risk=0.7,
        scores={"consistency": 0.4, "richness_rm": 0.3, "verifiability": 0.2, "relevance": 0.6},
        flagged_phrases=[{"text": "someone took it", "reason": "vague, unverifiable"}],
        rationale="thin account",
        available=True,
    )
    d = v.to_dict()
    assert d["risk"] == 0.7
    assert d["scores"]["consistency"] == 0.4
    assert d["flagged_phrases"][0]["text"] == "someone took it"
    assert d["available"] is True


def test_stub_is_a_content_judge():
    assert isinstance(StubContentJudge(), ContentJudge)


def test_stub_scores_vague_answer_riskier_than_concrete():
    """Deterministic heuristic: more vague/hedge words + fewer concrete tokens => higher risk.

    This lets us test fusion + the True/False discrimination without a live LLM.
    """
    stub = StubContentJudge()
    concrete = stub.judge("Where were you at 9pm?",
                          "I was at Mario's Pizza on 5th Street with my sister Anna until 10.",
                          history=[], baseline=None)
    vague = stub.judge("Where were you at 9pm?",
                       "I was just somewhere around, you know, with some people I think.",
                       history=[], baseline=None)
    assert vague.risk > concrete.risk
    assert vague.available is True


def test_stub_empty_answer_low_confidence():
    v = StubContentJudge().judge("Q?", "", history=[], baseline=None)
    assert v.available is True
    assert 0.0 <= v.risk <= 1.0
