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


def test_registry_has_five_visual_detectors():
    assert len(VISUAL_DETECTORS) == 5
    ids = {d().cue_id for d in VISUAL_DETECTORS}
    assert ids == {
        "visual.blink_rate", "visual.gaze_aversion", "visual.brow_flash",
        "visual.lip_press", "visual.jaw_tension",
    }


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
