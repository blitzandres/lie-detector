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


def test_visual_registry_has_twentyseven_detectors():
    from blitz_overlay.cues.visual import VISUAL_DETECTORS
    assert len(VISUAL_DETECTORS) == 27
    ids = {d().cue_id for d in VISUAL_DETECTORS}
    assert {"visual.head_movement", "visual.eye_squint", "visual.mouth_stretch",
            "visual.mouth_frown", "visual.mouth_shrug", "visual.jaw_shift", "visual.jaw_drop",
            "visual.lip_roll", "visual.brow_outer_raise", "visual.contempt_asymmetry"}.issubset(ids)


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


def _hp_frame(ts, yaw=0.0, pitch=0.0, roll=0.0):
    from blitz_overlay.schemas import FeatureFrame
    return FeatureFrame.from_dict({"ts": ts, "face_present": True, "confidence": 0.9,
                                   "blendshapes": {}, "geometry": {},
                                   "head_pose": {"yaw": yaw, "pitch": pitch, "roll": roll}})


def test_visual_registry_has_twentyone_detectors_subset():
    from blitz_overlay.cues.visual import VISUAL_DETECTORS
    ids = {d().cue_id for d in VISUAL_DETECTORS}
    assert {"visual.gaze_fixation", "visual.pupil_dilation", "visual.eye_blocking",
            "visual.eye_widen", "visual.nose_wrinkle", "visual.asymmetric_smile"}.issubset(ids)


def test_max_blendshape_cues_take_max_and_abstain():
    from blitz_overlay.cues.visual import EyeSquint, JawDrop, JawShift
    assert EyeSquint().measure(_bs_frame(0, eyeSquintLeft=0.2, eyeSquintRight=0.6)) == 0.6
    assert EyeSquint().measure(_bs_frame(0)) is None
    assert JawShift().measure(_bs_frame(0, jawLeft=0.1, jawRight=0.4, jawForward=0.2)) == 0.4
    assert JawDrop().measure(_bs_frame(0, jawOpen=0.55)) == 0.55


def test_contempt_asymmetry_is_absolute_difference():
    from blitz_overlay.cues.visual import ContemptAsymmetry
    d = ContemptAsymmetry()
    assert abs(d.measure(_bs_frame(0, mouthDimpleLeft=0.6, mouthDimpleRight=0.1)) - 0.5) < 1e-9
    assert d.measure(_bs_frame(0)) is None


def test_head_movement_accumulates_over_window():
    from blitz_overlay.cues.visual import HeadMovement
    d = HeadMovement()
    # steady head -> ~0 movement
    last = 0.0
    for t in range(0, 1600, 100):
        last = d.measure(_hp_frame(t, yaw=5.0, pitch=2.0, roll=1.0))
    assert last < 0.5
    # jerky head -> larger movement
    d2 = HeadMovement()
    last2 = 0.0
    for i, t in enumerate(range(0, 1600, 100)):
        last2 = d2.measure(_hp_frame(t, yaw=20.0 if i % 2 else -20.0, pitch=0.0, roll=0.0))
    assert last2 > 5.0


def test_head_movement_abstains_without_head_pose():
    from blitz_overlay.cues.visual import HeadMovement
    from blitz_overlay.schemas import FeatureFrame
    f = FeatureFrame.from_dict({"ts": 0, "face_present": True, "confidence": 0.9})
    assert HeadMovement().measure(f) is None


def test_duchenne_absence_high_when_smile_without_cheeks():
    from blitz_overlay.cues.visual import DuchenneAbsence
    d = DuchenneAbsence()
    masked = d.measure(_frame(0, mouthSmileLeft=0.8, mouthSmileRight=0.8,
                              cheekSquintLeft=0.05, cheekSquintRight=0.05))
    genuine = d.measure(_frame(0, mouthSmileLeft=0.8, mouthSmileRight=0.8,
                               cheekSquintLeft=0.7, cheekSquintRight=0.7))
    assert masked > genuine
    assert d.measure(_frame(0, mouthSmileLeft=0.1, mouthSmileRight=0.1,
                            cheekSquintLeft=0.0, cheekSquintRight=0.0)) == 0.0  # no smile -> no signal
    assert d.measure(_frame(0, browInnerUp=0.5)) is None  # inputs absent -> abstain


def test_stress_brow_requires_co_occurrence():
    from blitz_overlay.cues.visual import StressBrow
    d = StressBrow()
    all_up = d.measure(_frame(0, browInnerUp=0.6, browOuterUpLeft=0.5, browOuterUpRight=0.5,
                              browDownLeft=0.4, browDownRight=0.4))
    inner_only = d.measure(_frame(0, browInnerUp=0.6, browOuterUpLeft=0.0, browOuterUpRight=0.0,
                                  browDownLeft=0.0, browDownRight=0.0))
    assert all_up > inner_only
    assert inner_only == 0.0  # AU1 alone is not the AU1+2+4 combo
    assert d.measure(_frame(0, jawOpen=0.5)) is None


def test_face_asymmetry_averages_lr_pairs():
    from blitz_overlay.cues.visual import FaceAsymmetry
    d = FaceAsymmetry()
    sym = d.measure(_frame(0, eyeSquintLeft=0.4, eyeSquintRight=0.4,
                           browDownLeft=0.3, browDownRight=0.3))
    asym = d.measure(_frame(0, eyeSquintLeft=0.8, eyeSquintRight=0.1,
                            browDownLeft=0.6, browDownRight=0.1))
    assert asym > sym
    assert sym == 0.0
    assert d.measure(_frame(0, browInnerUp=0.5)) is None  # no paired keys -> abstain


def test_head_velocity_measures_rotation_speed():
    from blitz_overlay.cues.visual import HeadVelocity
    d = HeadVelocity()
    d.measure(_frame(0, g_yaw=0.0))
    still = d.measure(_frame(100, g_yaw=0.0))
    d2 = HeadVelocity()
    d2.measure(_frame(0, g_yaw=0.0))
    moving = d2.measure(_frame(100, g_yaw=8.0))   # 8 deg in 100 ms = 80 deg/s
    assert moving > still
    assert d.measure(FeatureFrame.from_dict({"ts": 200, "face_present": True})) is None


def test_head_acceleration_spikes_on_sudden_onset():
    from blitz_overlay.cues.visual import HeadAcceleration
    d = HeadAcceleration()
    for i, yaw in enumerate([0.0, 2.0, 4.0, 6.0]):     # steady rotation
        steady = d.measure(_frame(i * 100, g_yaw=yaw))
    d2 = HeadAcceleration()
    for i, yaw in enumerate([0.0, 0.0, 0.0, 9.0]):     # sudden jerk
        sudden = d2.measure(_frame(i * 100, g_yaw=yaw))
    assert sudden > steady


def test_blink_duration_reports_last_completed_blink():
    from blitz_overlay.cues.visual import BlinkDuration
    d = BlinkDuration()
    d.measure(_frame(0, eyeBlinkLeft=0.1))          # open
    d.measure(_frame(100, eyeBlinkLeft=0.9))        # closes
    d.measure(_frame(500, eyeBlinkLeft=0.9))        # held closed
    val = d.measure(_frame(600, eyeBlinkLeft=0.1))  # reopens -> blink took ~500 ms
    assert abs(val - 0.5) < 0.05
    assert d.measure(_frame(700, eyeBlinkLeft=0.1)) > 0.0      # remembered within window
    assert d.measure(_frame(20_000, eyeBlinkLeft=0.1)) == 0.0  # decayed after window
