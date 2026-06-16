from blitz_overlay.weights import CUE_WEIGHTS, WEIGHT_SET_VERSION, weight_for

WIRED = {
    "visual.blink_rate", "visual.gaze_aversion", "visual.brow_flash",
    "visual.lip_press", "visual.jaw_tension", "physio.heart_rate",
}


def test_all_wired_cues_have_annotated_weights():
    for cue_id in WIRED:
        spec = CUE_WEIGHTS[cue_id]
        assert spec["effect_size_d"] > 0
        assert spec["reliability_tier"] in (1, 2, 3, 4)
        assert spec["family"] in ("visual", "physio", "audio", "linguistic")
        assert spec["region"]
        assert spec["citation"], f"{cue_id} needs a citation (science-driven, not learned)"


def test_weight_set_is_versioned():
    assert isinstance(WEIGHT_SET_VERSION, str) and WEIGHT_SET_VERSION


def test_weight_for_helper_returns_spec():
    assert weight_for("visual.gaze_aversion")["effect_size_d"] == CUE_WEIGHTS["visual.gaze_aversion"]["effect_size_d"]


def test_gaze_is_strongest_visual_cue():
    # cue 58 (gaze aversion duration) d~0.6-0.8 should outrank brow/lip/jaw proxies
    assert CUE_WEIGHTS["visual.gaze_aversion"]["effect_size_d"] >= CUE_WEIGHTS["visual.brow_flash"]["effect_size_d"]
