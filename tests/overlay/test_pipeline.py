import math

from blitz_overlay.pipeline import OverlaySession


def _feature(ts, *, gaze=0.02, bpm=72, blink=0.05, jaw=0.80, confidence=0.9, face=True):
    f = bpm / 60.0
    t = ts / 1000.0
    g = 120 + 8 * math.sin(2 * math.pi * f * t)
    return {
        "ts": ts, "face_present": face, "confidence": confidence,
        "blendshapes": {"eyeBlinkLeft": blink, "eyeBlinkRight": blink,
                        "browInnerUp": 0.05, "mouthPressLeft": 0.05, "mouthPressRight": 0.05},
        "geometry": {"gaze_x": gaze, "gaze_y": 0.0, "jaw_width_ratio": jaw},
        "head_pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
        "rppg": {"forehead_rgb": [180.0, g, 110.0], "cheek_rgb": [185.0, g, 112.0]},
    }


def test_starts_calibrating(tmp_path):
    sess = OverlaySession(gate_threshold=0.65, baseline_seconds=5, log_dir=tmp_path)
    c = sess.process(_feature(0))
    assert c.status == "CALIBRATING"


def test_no_face_pauses_cues(tmp_path):
    sess = OverlaySession(gate_threshold=0.65, baseline_seconds=0, log_dir=tmp_path)
    c = sess.process(_feature(0, face=False))
    assert c.active_cues == []
    assert "no subject" in c.message.lower()


def test_calm_session_stays_clear_or_watch_never_flags(tmp_path):
    sess = OverlaySession(gate_threshold=0.65, baseline_seconds=3, log_dir=tmp_path)
    last = None
    for i in range(400):
        last = sess.process(_feature(int(i * 1000 / 30), gaze=0.02, bpm=72))
    assert last.flag is False
    assert last.status in ("CLEAR", "WATCH")


def test_writes_prediction_log(tmp_path):
    sess = OverlaySession(gate_threshold=0.65, baseline_seconds=0, log_dir=tmp_path)
    sess.process(_feature(0))
    sess.process(_feature(100))
    logs = list(tmp_path.glob("*.jsonl"))
    assert logs and logs[0].read_text().strip()
