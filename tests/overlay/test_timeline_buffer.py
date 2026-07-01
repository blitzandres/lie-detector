"""TimelineBuffer keeps per-frame cue activity so a turn can pull its [t0,t1] window."""
from blitz_overlay.content.timeline import TimelineBuffer


def test_window_summarizes_lit_cues_in_range():
    tl = TimelineBuffer(retain_ms=60000)
    tl.add(1000, [("visual.gaze_aversion", "visual", 3.0)])
    tl.add(1200, [("visual.gaze_aversion", "visual", 3.5), ("audio.tremor", "audio", 2.2)])
    tl.add(5000, [("visual.blink_rate", "visual", 4.0)])  # outside the window below

    w = tl.window(900, 1500)
    assert w["n_frames"] == 2
    assert set(w["families"]) == {"visual", "audio"}
    assert "visual.gaze_aversion" in w["cue_ids"]
    assert w["peak_z"] == 3.5
    assert w["max_families_synchronous"] >= 2  # both families lit within the window


def test_window_empty_when_no_frames_in_range():
    tl = TimelineBuffer(retain_ms=60000)
    tl.add(1000, [("visual.gaze_aversion", "visual", 3.0)])
    w = tl.window(8000, 9000)
    assert w["n_frames"] == 0
    assert w["families"] == []
    assert w["peak_z"] == 0.0


def test_retention_drops_old_frames():
    tl = TimelineBuffer(retain_ms=1000)
    tl.add(0, [("visual.gaze_aversion", "visual", 3.0)])
    tl.add(2000, [("audio.tremor", "audio", 3.0)])  # 2s later -> first frame evicted
    assert all(ts >= 1000 for ts, _ in tl._frames)
