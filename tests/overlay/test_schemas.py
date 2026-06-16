from blitz_overlay.schemas import (
    SCHEMA_VERSION,
    ActiveCue,
    Consensus,
    FamilyVote,
    FeatureFrame,
    Region,
)


def test_feature_frame_roundtrip():
    raw = {
        "schema_version": SCHEMA_VERSION,
        "ts": 1000,
        "face_present": True,
        "confidence": 0.9,
        "blendshapes": {"eyeBlinkLeft": 0.1, "browInnerUp": 0.2},
        "head_pose": {"yaw": 1.0, "pitch": -2.0, "roll": 0.5},
        "geometry": {"jaw_width_ratio": 0.83, "gaze_x": 0.1, "gaze_y": -0.05},
        "rppg": {"forehead_rgb": [180.0, 120.0, 110.0], "cheek_rgb": [190.0, 130.0, 120.0]},
    }
    frame = FeatureFrame.from_dict(raw)
    assert frame.ts == 1000
    assert frame.face_present is True
    assert frame.blendshapes["browInnerUp"] == 0.2
    assert frame.geometry["jaw_width_ratio"] == 0.83
    assert frame.rppg["forehead_rgb"][0] == 180.0


def test_feature_frame_tolerates_missing_optionals():
    frame = FeatureFrame.from_dict({"ts": 5, "face_present": False})
    assert frame.face_present is False
    assert frame.blendshapes == {}
    assert frame.rppg is None


def test_region_enum_values():
    assert Region.EYES.value == "eyes"
    assert {r.value for r in Region} >= {"eyes", "brow", "mouth", "jaw", "forehead"}


def test_consensus_to_dict_is_json_serializable():
    import json
    consensus = Consensus(
        schema_version=SCHEMA_VERSION,
        ts=2000,
        status="WATCH",
        risk=0.42,
        flag=False,
        n_agree=1,
        n_required=2,
        families=[
            FamilyVote(name="visual", wired=True, fresh=True, vote=False, contribution=0.3),
            FamilyVote(name="physio", wired=True, fresh=False, vote=False, contribution=0.0),
            FamilyVote(name="audio", wired=False, fresh=False, vote=False, contribution=0.0),
        ],
        active_cues=[ActiveCue(cue_id="visual.gaze_aversion", region="eyes", z=2.1, confidence=0.8)],
    )
    payload = json.dumps(consensus.to_dict())
    assert '"status": "WATCH"' in payload
    assert '"region": "eyes"' in payload
