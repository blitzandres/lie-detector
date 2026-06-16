from blitz_overlay.cues.base import CueDetector
from blitz_overlay.schemas import FeatureFrame
from core.calibration import RollingBaseline


class _Dummy(CueDetector):
    cue_id = "visual.blink_rate"

    def measure(self, frame):
        return frame.blendshapes.get("eyeBlinkLeft", None)


def test_detector_exposes_metadata_from_weights():
    d = _Dummy()
    assert d.family == "visual"
    assert d.region == "eyes"
    assert d.effect_size_d > 0
    assert d.reliability_tier in (1, 2, 3, 4)


def test_update_returns_none_when_measure_is_none():
    d = _Dummy()
    rb = RollingBaseline(baseline_seconds=0)
    rb.update({"visual.blink_rate": 0.1}, ts_ms=0)
    rb.update({"visual.blink_rate": 0.1}, ts_ms=1000)
    frame = FeatureFrame.from_dict({"ts": 2000, "face_present": True, "confidence": 0.9})
    assert d.update(frame, rb) is None  # eyeBlinkLeft absent -> measure None


def test_update_emits_cue_event_on_deviation():
    d = _Dummy()
    rb = RollingBaseline(baseline_seconds=0, window_seconds=60)
    for t, v in enumerate([0.1, 0.12, 0.09, 0.11, 0.1, 0.13]):
        rb.update({"visual.blink_rate": v}, ts_ms=t * 1000)
    frame = FeatureFrame.from_dict({
        "ts": 7000, "face_present": True, "confidence": 0.9,
        "blendshapes": {"eyeBlinkLeft": 0.9},
    })
    event = d.update(frame, rb)
    assert event is not None
    assert event.cue_id == "visual.blink_rate"
    assert event.modality.value == "visual"
    assert event.z_score > 2.0
    assert event.quality > 0


def test_low_confidence_widens_uncertainty_via_quality():
    d = _Dummy()
    rb = RollingBaseline(baseline_seconds=0, window_seconds=60)
    for t, v in enumerate([0.1, 0.12, 0.09, 0.11, 0.1, 0.13]):
        rb.update({"visual.blink_rate": v}, ts_ms=t * 1000)
    low = FeatureFrame.from_dict({"ts": 7000, "face_present": True, "confidence": 0.2,
                                  "blendshapes": {"eyeBlinkLeft": 0.9}})
    high = FeatureFrame.from_dict({"ts": 7000, "face_present": True, "confidence": 0.95,
                                   "blendshapes": {"eyeBlinkLeft": 0.9}})
    assert d.update(low, rb).quality < d.update(high, rb).quality
