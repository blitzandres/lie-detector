from blitz_overlay.pipeline import OverlaySession
from tests.overlay.fixtures.replay_session import replay_frames


def test_replay_is_deterministic(tmp_path):
    def run():
        sess = OverlaySession(gate_threshold=0.65, baseline_seconds=30, log_dir=tmp_path)
        return [sess.process(f).status for f in replay_frames()]
    assert run() == run()  # identical output across runs


def test_replay_progresses_calibrating_then_reaches_flag(tmp_path):
    sess = OverlaySession(gate_threshold=0.65, baseline_seconds=30, log_dir=tmp_path)
    statuses = [sess.process(f).status for f in replay_frames()]
    assert statuses[0] == "CALIBRATING"
    assert "FLAG" in statuses
    assert statuses.index("CALIBRATING") < statuses.index("FLAG")


def test_replay_flag_requires_two_families(tmp_path):
    sess = OverlaySession(gate_threshold=0.65, baseline_seconds=30, log_dir=tmp_path)
    flag_consensus = None
    for f in replay_frames():
        c = sess.process(f)
        if c.flag:
            flag_consensus = c
            break
    assert flag_consensus is not None
    assert flag_consensus.n_agree >= 2
    fresh = [fam.name for fam in flag_consensus.families if fam.fresh]
    assert "visual" in fresh and "physio" in fresh
