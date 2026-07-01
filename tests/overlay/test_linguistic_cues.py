"""Tests for the linguistic cue detectors (reused analyzer.py lexicons).

Mirrors test_audio_cues.py:
  - a frame WITH a transcript window yields a float measurement
  - a frame WITHOUT a transcript block yields None
  - a too-short window yields None
  - LINGUISTIC_DETECTORS contains exactly the seven expected detectors
"""
from blitz_overlay.cues.linguistic import (
    LINGUISTIC_DETECTORS,
    FillerRatio,
    PronounAvoidance,
)
from blitz_overlay.schemas import FeatureFrame
from core.calibration import RollingBaseline


def _frame(ts, text, seq=1):
    return FeatureFrame.from_dict({
        "ts": ts, "face_present": True, "confidence": 0.9,
        "transcript": {"text": text, "seq": seq},
    })


def _no_transcript_frame(ts):
    return FeatureFrame.from_dict({"ts": ts, "face_present": True, "confidence": 0.9})


def test_registry_has_seven_linguistic_detectors():
    assert len(LINGUISTIC_DETECTORS) == 7
    ids = {d().cue_id for d in LINGUISTIC_DETECTORS}
    assert ids == {
        "linguistic.sensory_detail_poverty", "linguistic.pronoun_avoidance",
        "linguistic.distancing_language", "linguistic.filler_ratio",
        "linguistic.qualifier_overload", "linguistic.negative_emotion_density",
        "linguistic.lexical_diversity_drop",
    }


def test_filler_ratio_measures_window():
    d = FillerRatio()
    # 8 tokens, 2 fillers ("um", "like") -> ratio 0.25
    m = d.measure(_frame(0, "um i was like at the store yesterday"))
    assert abs(m - 0.25) < 1e-9


def test_returns_none_without_transcript():
    assert FillerRatio().measure(_no_transcript_frame(0)) is None


def test_returns_none_when_window_too_short():
    # < MIN_WORDS (5) tokens
    assert PronounAvoidance().measure(_frame(0, "i was home")) is None


def test_linguistic_cue_emits_event_after_calibration():
    """FillerRatio fires once a person-relative deviation exceeds z_threshold.

    Linguistic features are bounded ~[0,1], so the flat-baseline fallback (z = value - median,
    capped near 1.0) cannot reach z>=2. We therefore calibrate on speech with a small but
    NONZERO filler-ratio spread (alternating 0.0 and ~0.11), giving a real MAD, then inject a
    filler-saturated window whose ratio sits many MADs out -> z >> 2.
    """
    d = FillerRatio()
    rb = RollingBaseline(baseline_seconds=0, window_seconds=600)

    # Two calm windows: ratio 0.0 and ~0.111 -> median ~0.055, MAD ~0.055 (nonzero spread).
    calm = [
        "i went to the store and bought some food there",   # 10 tokens, 0 fillers -> 0.0
        "um i went to the store and bought food",            # 9 tokens, 1 filler  -> ~0.111
    ]
    seq = 0
    for i, t in enumerate(range(0, 6000, 200)):
        seq += 1
        v = d.measure(_frame(t, calm[i % 2], seq=seq))
        rb.update({"linguistic.filler_ratio": v if v is not None else 0.0}, ts_ms=t)

    # Filler-saturated window: 8 fillers / 11 tokens -> ~0.727, many MADs above baseline.
    hot = "um uh like you know basically i was actually literally there"
    ts, event = 6000, None
    for _ in range(20):
        seq += 1
        frame = _frame(ts, hot, seq=seq)
        v = d.measure(frame)
        rb.update({"linguistic.filler_ratio": v if v is not None else 0.0}, ts_ms=ts)
        event = d.update(frame, rb, value=v)
        ts += 200

    assert event is not None
    assert event.cue_id == "linguistic.filler_ratio"
    assert event.region == "mouth"
    assert event.z_score >= d.z_threshold
