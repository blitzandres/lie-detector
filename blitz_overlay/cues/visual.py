"""Five visual cue detectors mapped to MediaPipe blendshapes / landmark geometry (spec §5)."""
from __future__ import annotations

from collections import deque

from blitz_overlay.cues.base import CueDetector
from blitz_overlay.schemas import FeatureFrame

BLINK_CLOSED_THRESHOLD = 0.5     # eyeBlink coefficient above this = eye closed
BLINK_WINDOW_MS = 30_000         # rolling window for blink-rate estimate
GAZE_OFFSET_THRESHOLD = 0.30     # combined gaze magnitude considered "averted"


class BlinkRate(CueDetector):
    """Blinks/min from eyeBlink blendshapes vs baseline (catalog cue 1/60)."""

    cue_id = "visual.blink_rate"
    direction = 1  # both directions matter, but elevated rate is the flag signal here

    def __init__(self) -> None:
        super().__init__()
        self._closed = False
        self._blink_ts: deque[int] = deque()

    def measure(self, frame: FeatureFrame) -> float | None:
        bs = frame.blendshapes
        if "eyeBlinkLeft" not in bs and "eyeBlinkRight" not in bs:
            return None
        closed_amt = max(bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0))
        now = frame.ts
        if closed_amt >= BLINK_CLOSED_THRESHOLD and not self._closed:
            self._closed = True
            self._blink_ts.append(now)          # rising edge = one blink
        elif closed_amt < BLINK_CLOSED_THRESHOLD:
            self._closed = False
        while self._blink_ts and self._blink_ts[0] < now - BLINK_WINDOW_MS:
            self._blink_ts.popleft()
        span_ms = max(1000, now - (self._blink_ts[0] if self._blink_ts else now))
        return len(self._blink_ts) * 60_000.0 / span_ms


class GazeAversion(CueDetector):
    """Sustained gaze-aversion *duration* in seconds (catalog cue 58)."""

    cue_id = "visual.gaze_aversion"
    direction = 1

    def __init__(self) -> None:
        super().__init__()
        self._averted_since: int | None = None

    def measure(self, frame: FeatureFrame) -> float | None:
        g = frame.geometry
        gx, gy = g.get("gaze_x"), g.get("gaze_y")
        if gx is None and gy is None:
            return None
        magnitude = (float(gx or 0.0) ** 2 + float(gy or 0.0) ** 2) ** 0.5
        now = frame.ts
        if magnitude >= GAZE_OFFSET_THRESHOLD:
            if self._averted_since is None:
                self._averted_since = now
            duration_s = (now - self._averted_since) / 1000.0
        else:
            self._averted_since = None
            duration_s = 0.0
        return duration_s


class BrowFlash(CueDetector):
    """Brow movement AU1/2 (browInnerUp) and AU4 (browDown) spikes (catalog cue 9)."""

    cue_id = "visual.brow_flash"
    direction = 1

    def measure(self, frame: FeatureFrame) -> float | None:
        bs = frame.blendshapes
        keys = ("browInnerUp", "browDownLeft", "browDownRight")
        if not any(k in bs for k in keys):
            return None
        inner = bs.get("browInnerUp", 0.0)
        down = (bs.get("browDownLeft", 0.0) + bs.get("browDownRight", 0.0)) / 2.0
        return max(inner, down)


class LipPress(CueDetector):
    """Lip compression / pucker (catalog cue 3)."""

    cue_id = "visual.lip_press"
    direction = 1

    def measure(self, frame: FeatureFrame) -> float | None:
        bs = frame.blendshapes
        keys = ("mouthPressLeft", "mouthPressRight", "mouthPucker")
        if not any(k in bs for k in keys):
            return None
        press = (bs.get("mouthPressLeft", 0.0) + bs.get("mouthPressRight", 0.0)) / 2.0
        return max(press, bs.get("mouthPucker", 0.0))


class JawTension(CueDetector):
    """Jaw-tension proxy from landmark-distance ratio (catalog cue 8, resolves Blocker 2/AU28)."""

    cue_id = "visual.jaw_tension"
    direction = 1

    def measure(self, frame: FeatureFrame) -> float | None:
        ratio = frame.geometry.get("jaw_width_ratio")
        return None if ratio is None else float(ratio)


VISUAL_DETECTORS = [BlinkRate, GazeAversion, BrowFlash, LipPress, JawTension]
