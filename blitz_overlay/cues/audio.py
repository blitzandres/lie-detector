"""Three audio cue detectors reading browser Web Audio DSP features (spec §3, stage-1 audio).

Each detector reads frame.audio (a dict of lightweight scalars computed by AudioCapture in
the browser). When frame.audio is absent (mic denied or not wired), measure() returns None
and the audio family is simply absent from consensus — no error, no flag.

Honest framing: these are real-time browser approximations, not clinical-grade measurements.
Parselmouth / WhisperX will supersede them in a later stage.
"""
from __future__ import annotations

from blitz_overlay.cues.base import CueDetector
from blitz_overlay.schemas import FeatureFrame


class PitchF0(CueDetector):
    """Fundamental pitch (F0) elevation — stress / cognitive-load proxy (catalog cue 21).

    Returns Hz when voice is present (f0 > 0), None when silent/unvoiced or no audio block.
    Direction +1: elevated F0 is the suspicious direction (excited / stressed speech).
    """

    cue_id = "audio.pitch_f0"
    direction = 1

    def measure(self, frame: FeatureFrame) -> float | None:
        if not frame.audio:
            return None
        f0 = frame.audio.get("f0", 0)
        if not f0 or f0 <= 0:
            return None  # unvoiced / silent — no measurement
        return float(f0)


class PauseRatio(CueDetector):
    """Pause ratio — fraction of recent ~2 s frames below silence threshold (catalog cue 26).

    Returns [0, 1]: 0 = fully voiced, 1 = full silence.
    Direction +1: elevated pause ratio is the suspicious direction (withholding / hesitation).
    """

    cue_id = "audio.pause_ratio"
    direction = 1

    def measure(self, frame: FeatureFrame) -> float | None:
        if not frame.audio:
            return None
        val = frame.audio.get("pause_ratio")
        if val is None:
            return None
        return float(val)


class VoiceTremor(CueDetector):
    """Voice tremor — short-term F0 variability (normalized stddev / CV) (catalog cue 22).

    Returns coefficient of variation of recent voiced F0 values; higher = more unstable voice.
    Strongest single audio cue per the cue catalog (d~0.4, tier 2).
    Direction +1: elevated tremor is the suspicious direction (vocal stress / instability).
    """

    cue_id = "audio.tremor"
    direction = 1

    def measure(self, frame: FeatureFrame) -> float | None:
        if not frame.audio:
            return None
        val = frame.audio.get("tremor")
        if val is None:
            return None
        return float(val)


AUDIO_DETECTORS = [PitchF0, PauseRatio, VoiceTremor]
