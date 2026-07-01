"""End-to-end: the existing replay (visual gaze + physio HR) rings the bell at max
sensitivity (k=2) but stays silent at the conservative default (k=3, only 2 cues lit)."""
from blitz_overlay.pipeline import OverlaySession
from tests.overlay.fixtures.replay_session import replay_frames


def _run(sensitivity):
    s = OverlaySession(baseline_seconds=40)  # replay calibrates for 40s then stresses
    bell_rang = False
    rows_seen = 0
    for raw in replay_frames():
        raw = {**raw, "config": {"sensitivity": sensitivity}}
        out = s.process(raw)
        rows_seen = max(rows_seen, len(out.cue_rows))
        if out.bell.get("just_rang"):
            bell_rang = True
    return bell_rang, rows_seen


def test_cue_rows_cover_all_registered_detectors():
    s = OverlaySession(baseline_seconds=0)
    out = s.process(next(iter(replay_frames())))
    ids = {r.cue_id for r in out.cue_rows}
    # every registered detector appears as a row (visual + audio + linguistic + physio)
    assert {d.cue_id for d in s.detectors}.issubset(ids)


def test_bell_rings_at_max_sensitivity():
    rang, rows = _run(sensitivity=1.0)
    assert rang is True
    assert rows >= 8  # all detectors present as rows


def test_bell_silent_at_conservative_default():
    rang, _ = _run(sensitivity=0.0)
    # only gaze + heart_rate lit (2 cues) -> below k=3 -> no burst -> no bell
    assert rang is False
