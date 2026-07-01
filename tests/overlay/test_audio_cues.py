"""Tests for the three audio cue detectors (PitchF0, PauseRatio, VoiceTremor).

Pattern mirrors test_visual_cues.py:
  - a frame WITH audio block yields a float measurement
  - a frame WITHOUT audio block yields None
  - AUDIO_DETECTORS registry contains exactly the three expected detectors
"""
from blitz_overlay.cues.audio import (
    AUDIO_DETECTORS,
    PauseRatio,
    PitchF0,
    VoiceTremor,
)
from blitz_overlay.schemas import FeatureFrame
from core.calibration import RollingBaseline


def _audio_frame(ts, **audio_kw):
    """Build a FeatureFrame carrying an audio block with given kwargs."""
    return FeatureFrame.from_dict({
        "ts": ts,
        "face_present": True,
        "confidence": 0.9,
        "blendshapes": {},
        "geometry": {},
        "head_pose": {},
        "audio": {
            "f0": audio_kw.get("f0", 0.0),
            "energy": audio_kw.get("energy", 0.0),
            "pause_ratio": audio_kw.get("pause_ratio", 0.0),
            "tremor": audio_kw.get("tremor", 0.0),
        },
    })


def _no_audio_frame(ts):
    """Build a FeatureFrame with no audio block."""
    return FeatureFrame.from_dict({
        "ts": ts,
        "face_present": True,
        "confidence": 0.9,
        "blendshapes": {},
        "geometry": {},
        "head_pose": {},
    })


# ── Registry ──────────────────────────────────────────────────────────────────

def test_registry_has_three_audio_detectors():
    assert len(AUDIO_DETECTORS) == 3
    ids = {d().cue_id for d in AUDIO_DETECTORS}
    assert ids == {"audio.pitch_f0", "audio.pause_ratio", "audio.tremor"}


# ── PitchF0 ───────────────────────────────────────────────────────────────────

def test_pitch_f0_returns_hz_when_voiced():
    d = PitchF0()
    m = d.measure(_audio_frame(0, f0=220.0))
    assert m == 220.0


def test_pitch_f0_returns_none_when_unvoiced():
    d = PitchF0()
    assert d.measure(_audio_frame(0, f0=0.0)) is None


def test_pitch_f0_returns_none_when_no_audio():
    d = PitchF0()
    assert d.measure(_no_audio_frame(0)) is None


# ── PauseRatio ────────────────────────────────────────────────────────────────

def test_pause_ratio_returns_value():
    d = PauseRatio()
    m = d.measure(_audio_frame(0, pause_ratio=0.4))
    assert abs(m - 0.4) < 1e-9


def test_pause_ratio_returns_zero_when_fully_voiced():
    d = PauseRatio()
    assert d.measure(_audio_frame(0, pause_ratio=0.0)) == 0.0


def test_pause_ratio_returns_none_when_no_audio():
    d = PauseRatio()
    assert d.measure(_no_audio_frame(0)) is None


# ── VoiceTremor ───────────────────────────────────────────────────────────────

def test_voice_tremor_returns_cv():
    d = VoiceTremor()
    m = d.measure(_audio_frame(0, tremor=0.05))
    assert abs(m - 0.05) < 1e-9


def test_voice_tremor_returns_zero_when_stable():
    d = VoiceTremor()
    assert d.measure(_audio_frame(0, tremor=0.0)) == 0.0


def test_voice_tremor_returns_none_when_no_audio():
    d = VoiceTremor()
    assert d.measure(_no_audio_frame(0)) is None


# ── CueEvent emission after calibration ───────────────────────────────────────

def test_audio_cue_emits_event_after_calibration():
    """VoiceTremor should emit a CueEvent once z >= threshold (mirrors visual test).

    Calibrate on near-zero tremor (0.01, constant → MAD=0, flat_fallback unit spread).
    Then inject a high-tremor value (3.0) that exceeds z_threshold=2.0 by at least 1 unit.
    With flat_fallback: z = value − median = 3.0 − 0.01 = 2.99 ≥ 2.0.
    """
    d = VoiceTremor()
    rb = RollingBaseline(baseline_seconds=0, window_seconds=120)

    # Calibrate on stable, near-zero tremor frames (30 frames over 6 s)
    for t in range(0, 6000, 200):
        v = d.measure(_audio_frame(t, tremor=0.01))
        rb.update({"audio.tremor": v if v is not None else 0.0}, ts_ms=t)

    # Feed high-tremor frames — z = 3.0 - 0.01 = 2.99 ≥ 2.0 threshold
    ts = 6000
    event = None
    for _ in range(20):
        frame = _audio_frame(ts, tremor=3.0)
        v = d.measure(frame)
        rb.update({"audio.tremor": v if v is not None else 0.0}, ts_ms=ts)
        event = d.update(frame, rb)
        ts += 200

    assert event is not None, (
        "VoiceTremor should emit a CueEvent when tremor >> baseline (z ~= 2.99 ≥ 2.0)"
    )
    assert event.cue_id == "audio.tremor"
    assert event.region == "head"
    assert event.z_score >= d.z_threshold
