"""Content-primary fusion: content drives the verdict; the cue window confirms."""
from blitz_overlay.content.fusion import fuse_turn
from blitz_overlay.content.judge import ContentVerdict


def _cue_window(families, peak_z=3.0, sync=2):
    return {"n_frames": 10, "cue_ids": [], "families": list(families),
            "peak_z": peak_z, "max_families_synchronous": sync}


def test_offline_content_falls_back_to_cue_only():
    res = fuse_turn(ContentVerdict.offline("ollama down"), _cue_window(["visual", "physio"]))
    assert res["content_available"] is False
    assert res["label"].startswith("cue-only")


def test_content_and_cue_agree_is_high_confidence():
    v = ContentVerdict(risk=0.8, scores={}, flagged_phrases=[], rationale="thin")
    res = fuse_turn(v, _cue_window(["visual", "audio"], sync=2))
    assert res["content_available"] is True
    assert res["combined"] >= 0.7
    assert res["convergence"] is True
    assert "converge" in res["label"].lower()


def test_content_high_but_cues_quiet_is_content_only_flag():
    v = ContentVerdict(risk=0.8, scores={}, flagged_phrases=[], rationale="thin")
    res = fuse_turn(v, _cue_window([], sync=0))
    assert res["convergence"] is False
    assert "content" in res["label"].lower()


def test_low_content_low_cue_is_clear():
    v = ContentVerdict(risk=0.1, scores={}, flagged_phrases=[], rationale="rich")
    res = fuse_turn(v, _cue_window([], sync=0))
    assert res["combined"] < 0.45
    assert res["label"].lower().startswith("clear")
