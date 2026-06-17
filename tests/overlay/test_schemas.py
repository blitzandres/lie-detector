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
    assert frame.audio is None


def test_feature_frame_audio_roundtrip():
    """Audio block is preserved through from_dict when present."""
    raw = {
        "ts": 2000,
        "face_present": True,
        "confidence": 0.9,
        "blendshapes": {},
        "geometry": {},
        "head_pose": {},
        "audio": {"f0": 120.5, "energy": 0.03, "pause_ratio": 0.2, "tremor": 0.05},
    }
    frame = FeatureFrame.from_dict(raw)
    assert frame.audio is not None
    assert abs(frame.audio["f0"] - 120.5) < 1e-9
    assert abs(frame.audio["pause_ratio"] - 0.2) < 1e-9
    assert abs(frame.audio["tremor"] - 0.05) < 1e-9


def test_feature_frame_audio_absent_when_key_missing():
    """audio=None when the key is not present in the raw dict."""
    frame = FeatureFrame.from_dict({"ts": 3000, "face_present": True, "confidence": 0.5})
    assert frame.audio is None


def test_region_enum_values():
    assert Region.EYES.value == "eyes"
    assert {r.value for r in Region} >= {"eyes", "brow", "mouth", "jaw", "forehead"}


def test_feature_frame_carries_transcript_block():
    from blitz_overlay.schemas import FeatureFrame
    frame = FeatureFrame.from_dict({
        "ts": 10, "face_present": True, "confidence": 0.9,
        "transcript": {"text": "i was at home all night", "seq": 3},
    })
    assert frame.transcript == {"text": "i was at home all night", "seq": 3}


def test_feature_frame_transcript_absent_is_none():
    from blitz_overlay.schemas import FeatureFrame
    frame = FeatureFrame.from_dict({"ts": 10, "face_present": True})
    assert frame.transcript is None


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
