"""VisualAnalyzer tests — stub backends only; never touches real models."""
import pytest

from core.calibration.baseline import PersonalBaseline
from modalities.visual.analyzer import MIN_FRAMES, VisualAnalyzer
from modalities.visual.backends import StubBackend, VisualFrame
from modalities.visual.flow import FlowSample, StubFlowSource


def _clip(n=40, au01=0.1, au12=0.0, au06=0.0, yaw=0.0, fear=0.05, wobble=0.0):
    frames = []
    for i in range(n):
        w = wobble * (i % 2)
        frames.append(VisualFrame(
            ts_ms=i * 100, face_present=True, quality=0.9,
            aus={"AU01": au01 + w, "AU02": au01 + w, "AU04": au01 + w,
                 "AU06": au06, "AU12": au12, "AU14": 0.05, "AU24": 0.1},
            emotions={"fear": fear, "anger": 0.02, "disgust": 0.02,
                      "sadness": 0.02, "happiness": 0.5},
            head_pose={"yaw": yaw * i, "pitch": 0.0, "roll": 0.0},
        ))
    return frames


def test_extract_features_produces_all_stub_reachable_cues():
    an = VisualAnalyzer(backend=StubBackend(_clip()))
    feats = an.extract_features("clip.mp4")
    for cue in ("visual.au_stress_brow", "visual.au_lip_press", "visual.au_contempt",
                "visual.duchenne_deficit", "visual.emotion_leakage",
                "visual.head_dynamics", "visual.expressivity_rigidity",
                "visual.au_micro_burst"):
        assert cue in feats
    assert "visual.flow_agitation" not in feats   # no FlowSource wired


def test_flow_agitation_requires_flow_source():
    flow = StubFlowSource([FlowSample(0, 0.2, 1.4), FlowSample(100, 0.3, 2.0)])
    an = VisualAnalyzer(backend=StubBackend(_clip()), flow=flow)
    assert abs(an.extract_features("clip.mp4")["visual.flow_agitation"] - 2.0) < 1e-9


def test_stress_brow_rises_with_combo():
    calm = VisualAnalyzer(backend=StubBackend(_clip(au01=0.05)))
    stressed = VisualAnalyzer(backend=StubBackend(_clip(au01=0.6)))
    assert (stressed.extract_features("c.mp4")["visual.au_stress_brow"]
            > calm.extract_features("c.mp4")["visual.au_stress_brow"])


def test_duchenne_deficit_zero_without_smile_and_high_when_masked():
    no_smile = VisualAnalyzer(backend=StubBackend(_clip(au12=0.1)))
    masked = VisualAnalyzer(backend=StubBackend(_clip(au12=0.8, au06=0.05)))
    genuine = VisualAnalyzer(backend=StubBackend(_clip(au12=0.8, au06=0.7)))
    assert no_smile.extract_features("c.mp4")["visual.duchenne_deficit"] == 0.0
    assert (masked.extract_features("c.mp4")["visual.duchenne_deficit"]
            > genuine.extract_features("c.mp4")["visual.duchenne_deficit"])


def test_rigidity_is_negated_variance():
    frozen = VisualAnalyzer(backend=StubBackend(_clip(wobble=0.0)))
    lively = VisualAnalyzer(backend=StubBackend(_clip(wobble=0.4)))
    # negated std: frozen (no variance) must score HIGHER than lively
    assert (frozen.extract_features("c.mp4")["visual.expressivity_rigidity"]
            > lively.extract_features("c.mp4")["visual.expressivity_rigidity"])


def test_abstains_on_too_few_face_frames():
    an = VisualAnalyzer(backend=StubBackend(_clip(n=MIN_FRAMES - 1)))
    with pytest.raises(ValueError, match="input_quality_insufficient"):
        an.extract_features("c.mp4")


def test_analyze_emits_cue_events_with_specs():
    an = VisualAnalyzer(backend=StubBackend(_clip()))
    baseline = PersonalBaseline()
    # PersonalBaseline needs >=3 observations per cue -> at least 3 baseline clips
    baseline.record_baseline(an.build_baseline_observations(["b1.mp4", "b2.mp4", "b3.mp4"]),
                             duration_s=120)
    events = an.analyze("c.mp4", question_id="q1", baseline=baseline, timestamp_ms=5)
    assert events and all(e.cue_id.startswith("visual.") for e in events)
    assert all(e.modality.value == "visual" for e in events)
    assert all(e.effect_size_d > 0 for e in events)
