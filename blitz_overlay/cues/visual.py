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


EYE_CLOSED_THRESHOLD = 0.5       # eyeBlink coefficient above this = eye held closed
GAZE_FIX_WINDOW_MS = 1500        # window for gaze darting velocity


class GazeFixation(CueDetector):
    """Gaze darting velocity — fabrication = more frequent, shorter fixations (catalog cue 56).

    Mean per-sample gaze movement over a ~1.5 s window. High = darting (suspicious); low =
    steady fixation. Distinct from gaze_aversion, which measures sustained off-centre duration.
    """

    cue_id = "visual.gaze_fixation"
    direction = 1

    def __init__(self) -> None:
        super().__init__()
        self._hist: deque[tuple[int, float, float]] = deque()

    def measure(self, frame: FeatureFrame) -> float | None:
        g = frame.geometry
        gx, gy = g.get("gaze_x"), g.get("gaze_y")
        if gx is None and gy is None:
            return None
        now = frame.ts
        self._hist.append((now, float(gx or 0.0), float(gy or 0.0)))
        while self._hist and self._hist[0][0] < now - GAZE_FIX_WINDOW_MS:
            self._hist.popleft()
        if len(self._hist) < 2:
            return 0.0
        steps = 0.0
        for (_, x0, y0), (_, x1, y1) in zip(self._hist, list(self._hist)[1:], strict=False):
            steps += ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        return steps / (len(self._hist) - 1)


class PupilDilation(CueDetector):
    """Pupil/iris dilation proxy — cognitive-load spike (catalog cue 7/55).

    Reads geometry.iris_ratio (iris diameter ÷ eye width). Quality is scaled down because
    the catalog notes this is only reliable at 720p+.
    """

    cue_id = "visual.pupil_dilation"
    direction = 1

    def measure(self, frame: FeatureFrame) -> float | None:
        val = frame.geometry.get("iris_ratio")
        return None if val is None else float(val)

    def quality(self, frame: FeatureFrame) -> float:
        return 0.5 * max(0.0, min(1.0, frame.confidence))


class EyeBlocking(CueDetector):
    """Eye blocking — prolonged eye closure *duration* while speaking (catalog cue 13)."""

    cue_id = "visual.eye_blocking"
    direction = 1

    def __init__(self) -> None:
        super().__init__()
        self._closed_since: int | None = None

    def measure(self, frame: FeatureFrame) -> float | None:
        bs = frame.blendshapes
        if "eyeBlinkLeft" not in bs and "eyeBlinkRight" not in bs:
            return None
        closed = max(bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0)) >= EYE_CLOSED_THRESHOLD
        now = frame.ts
        if closed:
            if self._closed_since is None:
                self._closed_since = now
            return (now - self._closed_since) / 1000.0
        self._closed_since = None
        return 0.0


class EyeWiden(CueDetector):
    """Eye widen (AU5, eyeWide) — surprise/fear leakage (catalog cue 9-adjacent)."""

    cue_id = "visual.eye_widen"
    direction = 1

    def measure(self, frame: FeatureFrame) -> float | None:
        bs = frame.blendshapes
        if "eyeWideLeft" not in bs and "eyeWideRight" not in bs:
            return None
        return max(bs.get("eyeWideLeft", 0.0), bs.get("eyeWideRight", 0.0))


class NoseWrinkle(CueDetector):
    """Nose wrinkle (AU9, noseSneer) — disgust/discomfort (catalog cue 4)."""

    cue_id = "visual.nose_wrinkle"
    direction = 1

    def measure(self, frame: FeatureFrame) -> float | None:
        bs = frame.blendshapes
        if "noseSneerLeft" not in bs and "noseSneerRight" not in bs:
            return None
        return max(bs.get("noseSneerLeft", 0.0), bs.get("noseSneerRight", 0.0))


class AsymmetricSmile(CueDetector):
    """Smile asymmetry (AU6/AU12 left-right) — fake vs Duchenne (catalog cue 5)."""

    cue_id = "visual.asymmetric_smile"
    direction = 1

    def measure(self, frame: FeatureFrame) -> float | None:
        bs = frame.blendshapes
        if "mouthSmileLeft" not in bs and "mouthSmileRight" not in bs:
            return None
        return abs(bs.get("mouthSmileLeft", 0.0) - bs.get("mouthSmileRight", 0.0))


HEAD_MOVE_WINDOW_MS = 2000


class _MaxBlendshapeCue(CueDetector):
    """Base for cues that are the max of a set of blendshape coefficients."""

    direction = 1
    keys: tuple[str, ...] = ()

    def measure(self, frame: FeatureFrame) -> float | None:
        bs = frame.blendshapes
        if not any(k in bs for k in self.keys):
            return None
        return max(bs.get(k, 0.0) for k in self.keys)


class EyeSquint(_MaxBlendshapeCue):
    cue_id = "visual.eye_squint"
    keys = ("eyeSquintLeft", "eyeSquintRight")


class MouthStretch(_MaxBlendshapeCue):
    cue_id = "visual.mouth_stretch"
    keys = ("mouthStretchLeft", "mouthStretchRight")


class MouthFrown(_MaxBlendshapeCue):
    cue_id = "visual.mouth_frown"
    keys = ("mouthFrownLeft", "mouthFrownRight")


class MouthShrug(_MaxBlendshapeCue):
    cue_id = "visual.mouth_shrug"
    keys = ("mouthShrugUpper", "mouthShrugLower")


class JawShift(_MaxBlendshapeCue):
    cue_id = "visual.jaw_shift"
    keys = ("jawLeft", "jawRight", "jawForward")


class JawDrop(_MaxBlendshapeCue):
    cue_id = "visual.jaw_drop"
    keys = ("jawOpen",)


class LipRoll(_MaxBlendshapeCue):
    cue_id = "visual.lip_roll"
    keys = ("mouthRollUpper", "mouthRollLower")


class BrowOuterRaise(_MaxBlendshapeCue):
    cue_id = "visual.brow_outer_raise"
    keys = ("browOuterUpLeft", "browOuterUpRight")


class ContemptAsymmetry(CueDetector):
    """Unilateral contempt (AU14) — left-right mouth-dimple asymmetry."""

    cue_id = "visual.contempt_asymmetry"
    direction = 1

    def measure(self, frame: FeatureFrame) -> float | None:
        bs = frame.blendshapes
        if "mouthDimpleLeft" not in bs and "mouthDimpleRight" not in bs:
            return None
        return abs(bs.get("mouthDimpleLeft", 0.0) - bs.get("mouthDimpleRight", 0.0))


class HeadMovement(CueDetector):
    """Head-movement magnitude over a ~2 s window — restlessness/discomfort (catalog cue 14)."""

    cue_id = "visual.head_movement"
    direction = 1

    def __init__(self) -> None:
        super().__init__()
        self._hist: deque[tuple[int, float, float, float]] = deque()

    def measure(self, frame: FeatureFrame) -> float | None:
        hp = frame.head_pose
        if not hp or not any(k in hp for k in ("yaw", "pitch", "roll")):
            return None
        now = frame.ts
        self._hist.append((now, float(hp.get("yaw", 0.0)),
                           float(hp.get("pitch", 0.0)), float(hp.get("roll", 0.0))))
        while self._hist and self._hist[0][0] < now - HEAD_MOVE_WINDOW_MS:
            self._hist.popleft()
        if len(self._hist) < 2:
            return 0.0
        steps = 0.0
        hist = list(self._hist)
        for (_, y0, p0, r0), (_, y1, p1, r1) in zip(hist, hist[1:], strict=False):
            steps += ((y1 - y0) ** 2 + (p1 - p0) ** 2 + (r1 - r0) ** 2) ** 0.5
        return steps / (len(hist) - 1)


SMILE_FLOOR = 0.3   # below this there is no smile to authenticate


class DuchenneAbsence(CueDetector):
    """Smile without eye involvement (AU12 high, AU6 low) — social/masked smile.

    Ekman's Duchenne marker: genuine enjoyment recruits orbicularis oculi (cheekSquint).
    Signal only exists while smiling; no smile → 0 (not suspicious).
    """

    cue_id = "visual.duchenne_absence"
    direction = 1

    def measure(self, frame: FeatureFrame) -> float | None:
        bs = frame.blendshapes
        keys = ("mouthSmileLeft", "mouthSmileRight", "cheekSquintLeft", "cheekSquintRight")
        if not any(k in bs for k in keys):
            return None
        smile = (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0)) / 2.0
        if smile < SMILE_FLOOR:
            return 0.0
        cheek = (bs.get("cheekSquintLeft", 0.0) + bs.get("cheekSquintRight", 0.0)) / 2.0
        return smile * max(0.0, smile - cheek)


class StressBrow(CueDetector):
    """AU1+AU2+AU4 co-occurrence — the fear/stress brow. All three must be present."""

    cue_id = "visual.stress_brow"
    direction = 1

    def measure(self, frame: FeatureFrame) -> float | None:
        bs = frame.blendshapes
        keys = ("browInnerUp", "browOuterUpLeft", "browOuterUpRight",
                "browDownLeft", "browDownRight")
        if not any(k in bs for k in keys):
            return None
        inner = bs.get("browInnerUp", 0.0)
        outer = (bs.get("browOuterUpLeft", 0.0) + bs.get("browOuterUpRight", 0.0)) / 2.0
        down = (bs.get("browDownLeft", 0.0) + bs.get("browDownRight", 0.0)) / 2.0
        return min(inner, outer, down)  # co-occurrence: the weakest component gates the combo


# L/R blendshape pairs for the multi-region asymmetry index (smile/dimple asymmetry
# already has dedicated cues — excluded to avoid double counting).
ASYMMETRY_PAIRS = (
    ("eyeBlinkLeft", "eyeBlinkRight"),
    ("eyeSquintLeft", "eyeSquintRight"),
    ("browDownLeft", "browDownRight"),
    ("mouthStretchLeft", "mouthStretchRight"),
    ("mouthFrownLeft", "mouthFrownRight"),
    ("mouthPressLeft", "mouthPressRight"),
)


class FaceAsymmetry(CueDetector):
    """Multi-region left/right deviation (eye + brow + mouth) beyond the smile cues."""

    cue_id = "visual.face_asymmetry"
    direction = 1

    def measure(self, frame: FeatureFrame) -> float | None:
        bs = frame.blendshapes
        deltas = [abs(bs[left] - bs[right])
                  for left, right in ASYMMETRY_PAIRS if left in bs and right in bs]
        if not deltas:
            return None
        return sum(deltas) / len(deltas)


HEAD_KIN_WINDOW_MS = 800         # smoothing window for velocity/acceleration
BLINK_MEMORY_MS = 5_000          # a completed blink stays reportable this long


class _HeadKinematics(CueDetector):
    """Shared pose-history buffer for velocity/acceleration cues."""

    direction = 1

    def __init__(self) -> None:
        super().__init__()
        self._hist: deque[tuple[int, float, float, float]] = deque()

    def _push(self, frame: FeatureFrame) -> list[tuple[int, float, float, float]] | None:
        hp = frame.head_pose
        if not hp or not any(k in hp for k in ("yaw", "pitch", "roll")):
            return None
        now = frame.ts
        self._hist.append((now, float(hp.get("yaw", 0.0)),
                           float(hp.get("pitch", 0.0)), float(hp.get("roll", 0.0))))
        while self._hist and self._hist[0][0] < now - HEAD_KIN_WINDOW_MS:
            self._hist.popleft()
        return list(self._hist)

    @staticmethod
    def _velocities(hist: list[tuple[int, float, float, float]]) -> list[float]:
        out = []
        for (t0, y0, p0, r0), (t1, y1, p1, r1) in zip(hist, hist[1:], strict=False):
            dt = max(1, t1 - t0) / 1000.0
            dist = ((y1 - y0) ** 2 + (p1 - p0) ** 2 + (r1 - r0) ** 2) ** 0.5
            out.append(dist / dt)
        return out


class HeadVelocity(_HeadKinematics):
    """Head rotation speed (°/s) — nods/shakes/tilts as tension or emphasis."""

    cue_id = "visual.head_velocity"

    def measure(self, frame: FeatureFrame) -> float | None:
        hist = self._push(frame)
        if hist is None:
            return None
        v = self._velocities(hist)
        return sum(v) / len(v) if v else 0.0


class HeadAcceleration(_HeadKinematics):
    """Sudden head-movement onsets (°/s per step) — jerky motion vs steady rotation."""

    cue_id = "visual.head_acceleration"

    def measure(self, frame: FeatureFrame) -> float | None:
        hist = self._push(frame)
        if hist is None:
            return None
        v = self._velocities(hist)
        if len(v) < 2:
            return 0.0
        return max(abs(v1 - v0) for v0, v1 in zip(v, v[1:], strict=False))


class BlinkDuration(CueDetector):
    """Duration of the most recent completed blink (s) — long blinks and slow rebound."""

    cue_id = "visual.blink_duration"
    direction = 1

    def __init__(self) -> None:
        super().__init__()
        self._closed_since: int | None = None
        self._last_blink: tuple[int, float] | None = None   # (ended_ts, duration_s)

    def measure(self, frame: FeatureFrame) -> float | None:
        bs = frame.blendshapes
        if "eyeBlinkLeft" not in bs and "eyeBlinkRight" not in bs:
            return None
        closed = max(bs.get("eyeBlinkLeft", 0.0),
                     bs.get("eyeBlinkRight", 0.0)) >= BLINK_CLOSED_THRESHOLD
        now = frame.ts
        if closed and self._closed_since is None:
            self._closed_since = now
        elif not closed and self._closed_since is not None:
            self._last_blink = (now, (now - self._closed_since) / 1000.0)
            self._closed_since = None
        if self._last_blink and now - self._last_blink[0] <= BLINK_MEMORY_MS:
            return self._last_blink[1]
        return 0.0


VISUAL_DETECTORS = [
    BlinkRate, GazeAversion, BrowFlash, LipPress, JawTension,
    GazeFixation, PupilDilation, EyeBlocking, EyeWiden, NoseWrinkle, AsymmetricSmile,
    HeadMovement, EyeSquint, MouthStretch, MouthFrown, MouthShrug,
    JawShift, JawDrop, LipRoll, BrowOuterRaise, ContemptAsymmetry,
    DuchenneAbsence, StressBrow, FaceAsymmetry,
    HeadVelocity, HeadAcceleration, BlinkDuration,
]
