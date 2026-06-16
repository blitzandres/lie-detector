import math

from blitz_overlay.cues.physio import RppgHeartRate
from blitz_overlay.schemas import FeatureFrame
from core.calibration import RollingBaseline


def _frame(ts, bpm):
    f = bpm / 60.0
    t = ts / 1000.0
    g = 120 + 8 * math.sin(2 * math.pi * f * t)
    return FeatureFrame.from_dict({
        "ts": ts, "face_present": True, "confidence": 0.9,
        "rppg": {"forehead_rgb": [180.0, g, 110.0], "cheek_rgb": [185.0, g, 112.0]},
    })


def test_measure_none_until_buffer_fills():
    d = RppgHeartRate(fps=30)
    assert d.measure(_frame(0, 72)) is None  # buffer not full


def test_measure_returns_bpm_once_buffer_fills():
    d = RppgHeartRate(fps=30)
    bpm = None
    for i in range(300):  # 10s at 30fps
        bpm = d.measure(_frame(int(i * 1000 / 30), 72))
    assert bpm is not None and 60 <= bpm <= 84


def test_emits_event_when_hr_elevated_vs_baseline():
    d = RppgHeartRate(fps=30)
    rb = RollingBaseline(baseline_seconds=0, window_seconds=600)
    i = 0
    # baseline ~72 bpm
    for _ in range(450):
        ts = int(i * 1000 / 30)
        i += 1
        v = d.measure(_frame(ts, 72))
        if v is not None:
            rb.update({"physio.heart_rate": v}, ts_ms=ts)
    # elevate to ~105 bpm
    event = None
    for _ in range(450):
        ts = int(i * 1000 / 30)
        i += 1
        v = d.measure(_frame(ts, 105))
        if v is not None:
            rb.update({"physio.heart_rate": v}, ts_ms=ts)
            event = d.update(_frame(ts, 105), rb)
    assert event is not None
    assert event.modality.value == "physiological"
    assert event.region == "forehead"
