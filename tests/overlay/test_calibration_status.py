"""Tests for the hard-gated active-calibration status (option 2)."""
from blitz_overlay.calibration_status import compute_calibration

DET = [
    ("visual.blink_rate", "visual", "blink_rate"),
    ("visual.gaze_aversion", "visual", "gaze_aversion"),
    ("audio.tremor", "audio", "tremor"),
    ("linguistic.filler_ratio", "linguistic", "filler_ratio"),
    ("physio.heart_rate", "physio", "heart_rate"),
]


def test_calibrating_until_time_elapses():
    cal, d = compute_calibration(
        DET, {"visual.blink_rate": 10, "visual.gaze_aversion": 10}, elapsed_s=2, target_s=5)
    assert cal is True
    assert d["active"] is True
    assert d["progress"] == 0.4


def test_waiting_cue_blocks_completion_past_time():
    # time elapsed but a producing cue has only 3 obs (1..7) -> waiting -> still calibrating
    cal, d = compute_calibration(
        DET, {"visual.blink_rate": 20, "visual.gaze_aversion": 20, "linguistic.filler_ratio": 3},
        elapsed_s=10, target_s=5)
    assert cal is True
    assert "linguistic.filler_ratio" in d["blocking"]
    assert "linguistic" in d["needs"]


def test_completes_when_time_done_and_no_waiting():
    cal, d = compute_calibration(
        DET, {"visual.blink_rate": 20, "visual.gaze_aversion": 20, "physio.heart_rate": 12},
        elapsed_s=10, target_s=5)
    assert cal is False
    assert d["active"] is False
    assert d["blocking"] == []


def test_no_signal_cue_does_not_block():
    # audio/linguistic/physio have zero obs -> no-signal -> excluded from the gate
    cal, _ = compute_calibration(
        DET, {"visual.blink_rate": 20, "visual.gaze_aversion": 20}, elapsed_s=10, target_s=5)
    assert cal is False


def test_timeout_forces_completion_despite_waiting():
    cal, d = compute_calibration(
        DET, {"visual.blink_rate": 20, "linguistic.filler_ratio": 2},
        elapsed_s=70, target_s=5, timeout_s=65)
    assert cal is False
    assert d["timed_out"] is True


def test_already_calibrated_latches():
    cal, _ = compute_calibration(DET, {}, elapsed_s=1, target_s=5, already_calibrated=True)
    assert cal is False


def test_family_status_and_counts():
    cal, d = compute_calibration(
        DET, {"visual.blink_rate": 20, "visual.gaze_aversion": 3}, elapsed_s=10, target_s=5)
    fam = d["families"]
    assert fam["visual"]["ready"] == 1
    assert fam["visual"]["waiting"] == 1
    assert fam["visual"]["status"] == "waiting"
    assert fam["audio"]["status"] == "idle"
    assert cal is True  # a waiting cue keeps it calibrating
