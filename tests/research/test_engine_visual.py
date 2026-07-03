from blitz_engine.engine import BlitzEngine
from modalities.visual.analyzer import VisualAnalyzer
from modalities.visual.backends import StubBackend
from tests.research.test_visual_analyzer import _clip


def test_engine_runs_visual_modality_end_to_end():
    analyzer = VisualAnalyzer(backend=StubBackend(_clip()))
    engine = BlitzEngine(modalities=["visual"], visual_analyzer=analyzer)
    session = engine.new_session(
        baseline_video_files=["b1.mp4", "b2.mp4", "b3.mp4"],
        consent=True, use_case="research", jurisdiction="CA-US",
    )
    result = session.analyze(video_path="r.mp4", question="Where were you?")
    assert 0.0 <= result.risk_score <= 1.0
    assert result.quality_flags["input_mode"] == "video"
    assert any(c["cue_id"].startswith("visual.") for c in result.top_cues)
