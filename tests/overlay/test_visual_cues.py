from blitz_overlay.cues.visual import (
    VISUAL_DETECTORS,
    BlinkRate,
    BrowFlash,
    GazeAversion,
    JawTension,
    LipPress,
)
from blitz_overlay.schemas import FeatureFrame
from core.calibration import RollingBaseline


def _frame(ts, **bs_geo):
    bs = {k: v for k, v in bs_geo.items() if not k.startswith("g_")}
    geo = {k[2:]: v for k, v in bs_geo.items() if k.startswith("g_")}
    return FeatureFrame.from_dict({
        "ts": ts, "face_present": True, "confidence": 0.9,
        "blendshapes": bs, "geometry": geo,
        "head_pose": {"yaw": geo.get("yaw", 0.0), "pitch": geo.get("pitch", 0.0)},
    })


def test_registry_has_visual_detectors():
    # originally 5; grows to 11 after facial-cue-empowerment (Task 2)
    ids = {d().cue_id for d in VISUAL_DETECTORS}
    assert {"visual.blink_rate", "visual.gaze_aversion", "visual.brow_flash",
            "visual.lip_press", "visual.jaw_tension"}.issubset(ids)


def test_brow_flash_measure_combines_inner_up_and_brow_down():
    d = BrowFlash()
    m = d.measure(_frame(0, browInnerUp=0.6, browDownLeft=0.2, browDownRight=0.4))
    assert m == 0.6  # max of browInnerUp and mean browDown


def test_lip_press_measure_combines_press_and_pucker():
    d = LipPress()
    m = d.measure(_frame(0, mouthPressLeft=0.3, mouthPressRight=0.5, mouthPucker=0.2))
    assert abs(m - 0.4) < 1e-9  # mean of press L/R = 0.4 dominates pucker 0.2


def test_jaw_tension_reads_geometry_ratio():
    d = JawTension()
    assert d.measure(_frame(0, g_jaw_width_ratio=0.81)) == 0.81
    assert d.measure(_frame(0)) is None  # missing geometry -> unavailable


def test_blink_rate_counts_blinks_per_minute():
    d = BlinkRate()
    ts = 0
    rate = None
    for _i in range(3):
        rate = d.measure(_frame(ts, eyeBlinkLeft=0.05, eyeBlinkRight=0.05))
        ts += 1000  # open
        d.measure(_frame(ts, eyeBlinkLeft=0.8, eyeBlinkRight=0.8))
        ts += 200  # closed (blink)
        rate = d.measure(_frame(ts, eyeBlinkLeft=0.05, eyeBlinkRight=0.05))
        ts += 800  # open
    assert rate is not None and rate > 0


def test_gaze_aversion_requires_sustained_offset():
    d = GazeAversion()
    assert d.measure(_frame(0, g_gaze_x=0.5, g_gaze_y=0.0)) == 0.0  # not yet sustained
    m = None
    for t in range(0, 3000, 200):
        m = d.measure(_frame(t, g_gaze_x=0.6, g_gaze_y=0.1))
    assert m is not None and m > 2.0  # sustained aversion duration in seconds


def test_visual_cue_emits_event_after_calibration():
    d = GazeAversion()
    rb = RollingBaseline(baseline_seconds=0, window_seconds=120)
    for t in range(0, 6000, 200):
        v = d.measure(_frame(t, g_gaze_x=0.02, g_gaze_y=0.0))
        rb.update({"visual.gaze_aversion": v}, ts_ms=t)
    ts = 6000
    event = None
    for _ in range(20):
        v = d.measure(_frame(ts, g_gaze_x=0.7, g_gaze_y=0.2))
        rb.update({"visual.gaze_aversion": v}, ts_ms=ts)
        event = d.update(_frame(ts, g_gaze_x=0.7, g_gaze_y=0.2), rb)
        ts += 200
    assert event is not None and event.region == "eyes"


def _bs_frame(ts, **bs):
    from blitz_overlay.schemas import FeatureFrame
    return FeatureFrame.from_dict({"ts": ts, "face_present": True, "confidence": 0.9,
                                   "blendshapes": bs, "geometry": {}})


def _geo_frame(ts, **geo):
    from blitz_overlay.schemas import FeatureFrame
    return FeatureFrame.from_dict({"ts": ts, "face_present": True, "confidence": 0.9,
                                   "blendshapes": {}, "geometry": geo})


def test_visual_registry_has_eleven_detectors():
    from blitz_overlay.cues.visual import VISUAL_DETECTORS
    assert len(VISUAL_DETECTORS) == 11
    ids = {d().cue_id for d in VISUAL_DETECTORS}
    assert {"visual.gaze_fixation", "visual.pupil_dilation", "visual.eye_blocking",
            "visual.eye_widen", "visual.nose_wrinkle", "visual.asymmetric_smile"}.issubset(ids)


def test_pupil_dilation_reads_iris_ratio():
    from blitz_overlay.cues.visual import PupilDilation
    d = PupilDilation()
    assert abs(d.measure(_geo_frame(0, iris_ratio=0.42)) - 0.42) < 1e-9
    assert d.measure(_geo_frame(0)) is None                 # no iris_ratio -> abstain
    assert d.measure(_geo_frame(0, iris_ratio=None)) is None


def test_pupil_dilation_quality_scaled_for_low_res():
    from blitz_overlay.cues.visual import PupilDilation
    d = PupilDilation()
    # quality is scaled below raw confidence (catalog: needs 720p+)
    assert d.quality(_geo_frame(0, iris_ratio=0.4)) < 0.9


def test_eye_widen_takes_max_side():
    from blitz_overlay.cues.visual import EyeWiden
    d = EyeWiden()
    assert d.measure(_bs_frame(0, eyeWideLeft=0.2, eyeWideRight=0.7)) == 0.7
    assert d.measure(_bs_frame(0)) is None


def test_nose_wrinkle_takes_max_side():
    from blitz_overlay.cues.visual import NoseWrinkle
    d = NoseWrinkle()
    assert d.measure(_bs_frame(0, noseSneerLeft=0.3, noseSneerRight=0.1)) == 0.3
    assert d.measure(_bs_frame(0)) is None


def test_asymmetric_smile_is_absolute_difference():
    from blitz_overlay.cues.visual import AsymmetricSmile
    d = AsymmetricSmile()
    m = d.measure(_bs_frame(0, mouthSmileLeft=0.7, mouthSmileRight=0.2))
    assert abs(m - 0.5) < 1e-9
    assert d.measure(_bs_frame(0)) is None


def test_eye_blocking_accumulates_closed_duration():
    from blitz_overlay.cues.visual import EyeBlocking
    d = EyeBlocking()
    assert d.measure(_bs_frame(0, eyeBlinkLeft=0.9, eyeBlinkRight=0.9)) == 0.0   # just closed
    held = d.measure(_bs_frame(1500, eyeBlinkLeft=0.9, eyeBlinkRight=0.9))       # 1.5s closed
    assert abs(held - 1.5) < 1e-6
    assert d.measure(_bs_frame(2000, eyeBlinkLeft=0.0, eyeBlinkRight=0.0)) == 0.0  # eyes open -> reset


def test_gaze_fixation_measures_darting_velocity():
    from blitz_overlay.cues.visual import GazeFixation
    d = GazeFixation()
    # steady gaze -> ~0 velocity
    for t in range(0, 1600, 100):
        v = d.measure(_geo_frame(t, gaze_x=0.1, gaze_y=0.0))
    assert v < 0.05
    # darting gaze -> higher velocity
    d2 = GazeFixation()
    last = 0.0
    for i, t in enumerate(range(0, 1600, 100)):
        gx = 0.4 if i % 2 else -0.4
        last = d2.measure(_geo_frame(t, gaze_x=gx, gaze_y=0.0))
    assert last > 0.2
