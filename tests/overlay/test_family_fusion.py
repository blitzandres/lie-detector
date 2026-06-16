from core.fusion.bayesian_fusion import FAMILY_THRESHOLD, family_of, fuse_by_family, two_gate
from core.schemas.cue_event import CueEvent, Modality, Phase


def _cue(cue_id, modality, z, d=0.5, tier=2, quality=0.9):
    return CueEvent(cue_id=cue_id, modality=modality, timestamp_ms=0, phase=Phase.RESPONSE,
                    raw_value=0.0, z_score=z, llr=0.0, quality=quality, question_id="live",
                    effect_size_d=d, reliability_tier=tier)


def test_family_of_maps_modalities():
    assert family_of(Modality.VISUAL) == "visual"
    assert family_of(Modality.PHYSIOLOGICAL) == "physio"


def test_within_family_correlated_cues_do_not_stack_like_independent():
    five = [_cue(f"visual.c{i}", Modality.VISUAL, z=3.0) for i in range(5)]
    one = [_cue("visual.c0", Modality.VISUAL, z=3.0)]
    res5 = fuse_by_family(five)
    res1 = fuse_by_family(one)
    assert res5["families"]["visual"] < 5 * res1["families"]["visual"]
    assert res5["families"]["visual"] >= res1["families"]["visual"]


def test_two_gate_requires_two_independent_families():
    visual_only = [_cue("visual.a", Modality.VISUAL, z=4.0),
                   _cue("visual.b", Modality.VISUAL, z=4.0)]
    res = fuse_by_family(visual_only)
    gate = two_gate(res, threshold=0.65)
    assert gate["flag"] is False
    assert gate["n_agree"] <= 1


def test_two_gate_flags_with_two_families_and_high_risk():
    cues = [_cue("visual.gaze", Modality.VISUAL, z=4.0, d=0.7),
            _cue("physio.hr", Modality.PHYSIOLOGICAL, z=4.0, d=0.5)]
    res = fuse_by_family(cues)
    gate = two_gate(res, threshold=0.65)
    assert gate["n_agree"] == 2
    assert res["posterior"] >= 0.65
    assert gate["flag"] is True


def test_family_vote_threshold_constant_exposed():
    assert 0.0 < FAMILY_THRESHOLD < 1.0
