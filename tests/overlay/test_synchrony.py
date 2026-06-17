"""Tests for SynchronyDetector — temporal co-firing burst over the existing per-cue z-scores."""
from blitz_overlay.synchrony import SynchronyDetector


def test_burst_requires_k_cues_and_two_families():
    d = SynchronyDetector(window_ms=1000, lit_z=2.0, k=3)
    # 3 lit cues but only ONE family -> not a burst (correlated, not convergent)
    snap = d.update(0, [("visual.gaze_aversion", "visual", 3.0),
                        ("visual.blink_rate", "visual", 2.5),
                        ("visual.lip_press", "visual", 2.2)])
    assert snap["n_lit"] == 3
    assert snap["n_families"] == 1
    assert snap["burst"] is False


def test_burst_true_with_three_cues_across_two_families():
    d = SynchronyDetector(window_ms=1000, lit_z=2.0, k=3)
    snap = d.update(0, [("visual.gaze_aversion", "visual", 3.0),
                        ("visual.blink_rate", "visual", 2.5),
                        ("physio.heart_rate", "physio", 2.4)])
    assert snap["n_lit"] == 3
    assert snap["n_families"] == 2
    assert snap["burst"] is True
    assert set(snap["families_lit"]) == {"visual", "physio"}


def test_below_lit_z_is_not_lit():
    d = SynchronyDetector(window_ms=1000, lit_z=2.0, k=3)
    snap = d.update(0, [("visual.gaze_aversion", "visual", 1.9),
                        ("physio.heart_rate", "physio", 5.0)])
    assert snap["n_lit"] == 1
    assert snap["burst"] is False


def test_rolling_window_expires_stale_lit_cues():
    d = SynchronyDetector(window_ms=1000, lit_z=2.0, k=3)
    d.update(0, [("visual.gaze_aversion", "visual", 3.0),
                 ("visual.blink_rate", "visual", 3.0)])
    # 1500ms later only physio lights; the two visual cues are now stale (>1000ms old)
    snap = d.update(1500, [("physio.heart_rate", "physio", 3.0)])
    assert snap["n_lit"] == 1
    assert snap["n_families"] == 1
    assert snap["burst"] is False


def test_recent_lit_cues_within_window_aggregate():
    d = SynchronyDetector(window_ms=1000, lit_z=2.0, k=3)
    d.update(0, [("visual.gaze_aversion", "visual", 3.0)])
    d.update(300, [("audio.tremor", "audio", 3.0)])
    snap = d.update(600, [("physio.heart_rate", "physio", 3.0)])
    # all three lit within the 1000ms window -> burst across 3 families
    assert snap["n_lit"] == 3
    assert snap["n_families"] == 3
    assert snap["burst"] is True


def test_set_params_updates_thresholds():
    d = SynchronyDetector(window_ms=1000, lit_z=2.0, k=3)
    d.set_params(lit_z=1.5, k=2)
    snap = d.update(0, [("visual.gaze_aversion", "visual", 1.6),
                        ("physio.heart_rate", "physio", 1.6)])
    assert snap["n_lit"] == 2
    assert snap["burst"] is True
