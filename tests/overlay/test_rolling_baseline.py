from core.calibration import RollingBaseline


def test_calibrating_until_window_filled_then_ready():
    rb = RollingBaseline(baseline_seconds=10)
    rb.update({"visual.blink_rate": 12.0}, ts_ms=0)
    assert rb.mode == "rolling"
    assert rb.is_calibrating is True
    for t in range(1, 12):
        rb.update({"visual.blink_rate": 12.0 + (t % 3) * 0.1}, ts_ms=t * 1000)
    assert rb.is_calibrating is False
    assert rb.ready is True


def test_normalize_returns_zero_while_calibrating():
    rb = RollingBaseline(baseline_seconds=10)
    rb.update({"visual.blink_rate": 12.0}, ts_ms=0)
    assert rb.normalize("visual.blink_rate", 40.0) == 0.0


def test_robust_z_after_calibration_flags_deviation():
    rb2 = RollingBaseline(baseline_seconds=5, window_seconds=60)
    for t, v in enumerate([11.0, 12.0, 13.0, 12.0, 11.0, 13.0, 12.0]):
        rb2.update({"visual.blink_rate": v}, ts_ms=t * 1000)
    z = rb2.normalize("visual.blink_rate", 30.0)
    assert z > 3.0


def test_unknown_cue_normalizes_to_zero():
    rb = RollingBaseline(baseline_seconds=1)
    rb.update({"visual.blink_rate": 12.0}, ts_ms=0)
    rb.update({"visual.blink_rate": 12.0}, ts_ms=2000)
    assert rb.normalize("visual.never_seen", 5.0) == 0.0


def test_window_evicts_old_observations():
    rb = RollingBaseline(baseline_seconds=1, window_seconds=5)
    for t in range(0, 10):
        rb.update({"c": float(t)}, ts_ms=t * 1000)
    assert rb.observation_count("c") <= 6
