"""session.judge_turn fuses a content verdict with the cue activity of the answer window."""
from blitz_overlay.content.judge import StubContentJudge
from blitz_overlay.pipeline import OverlaySession


def test_judge_turn_returns_fused_result_with_stub():
    sess = OverlaySession(baseline_seconds=0, content_judge=StubContentJudge())
    res = sess.judge_turn(question="Where were you at 9pm?",
                          answer="I was just around somewhere with some people I guess.",
                          t0=0, t1=5000)
    assert res["content_available"] is True
    assert "combined" in res and 0.0 <= res["combined"] <= 1.0
    assert "label" in res
    assert res["content"]["risk"] > 0.3  # vague answer -> elevated content risk


def test_judge_turn_pulls_cue_window(monkeypatch):
    """A synchronous 2-family cue burst inside the window should mark convergence."""
    sess = OverlaySession(baseline_seconds=0, content_judge=StubContentJudge())
    # Inject lit cue activity into the timeline at t≈1000 across 2 families.
    sess.timeline.add(1000, [("visual.gaze_aversion", "visual", 4.0),
                             ("audio.tremor", "audio", 3.0)])
    res = sess.judge_turn(question="Q?",
                          answer="just some stuff, you know, somewhere with people",
                          t0=900, t1=1100)
    assert res["cue_synchronous_families"] >= 2
