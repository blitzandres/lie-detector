"""Pipeline integration for the linguistic family + per-utterance seq de-dup."""
from blitz_overlay.pipeline import OverlaySession


def _frame(ts, text, seq):
    return {
        "ts": ts, "face_present": True, "confidence": 0.9,
        "transcript": {"text": text, "seq": seq},
    }


def test_linguistic_detectors_registered():
    s = OverlaySession(baseline_seconds=0)
    ids = {d.cue_id for d in s.detectors}
    assert "linguistic.filler_ratio" in ids
    assert "linguistic.pronoun_avoidance" in ids


def test_duplicate_seq_not_refed_to_baseline():
    """A repeated transcript seq must not add another baseline observation."""
    s = OverlaySession(baseline_seconds=0)
    s.process(_frame(0, "um i was like at the store yesterday", seq=1))
    n_after_first = s.baseline.observation_count("linguistic.filler_ratio")
    assert n_after_first >= 1
    # Same seq again on the next frame -> treated as absent -> no new observation
    s.process(_frame(33, "um i was like at the store yesterday", seq=1))
    assert s.baseline.observation_count("linguistic.filler_ratio") == n_after_first
    # New seq -> a new observation is added
    s.process(_frame(66, "the person took that thing away from someone", seq=2))
    assert s.baseline.observation_count("linguistic.filler_ratio") == n_after_first + 1
