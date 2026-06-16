from blitz_overlay.consensus import ConsensusBuilder
from blitz_overlay.schemas import Consensus
from core.schemas.cue_event import CueEvent, Modality, Phase


def _cue(cue_id, modality, z, region, d=0.6):
    return CueEvent(cue_id=cue_id, modality=modality, timestamp_ms=0, phase=Phase.RESPONSE,
                    raw_value=0.0, z_score=z, llr=0.0, quality=0.9, question_id="live",
                    effect_size_d=d, reliability_tier=2)


def test_calibrating_blocks_flags():
    cb = ConsensusBuilder()
    out = cb.build(cues=[_cue("visual.gaze_aversion", Modality.VISUAL, 5.0, "eyes")],
                   calibrating=True, ts=1000, regions={"visual.gaze_aversion": "eyes"})
    assert isinstance(out, Consensus)
    assert out.status == "CALIBRATING"
    assert out.flag is False


def test_clear_when_no_cues():
    cb = ConsensusBuilder()
    out = cb.build(cues=[], calibrating=False, ts=1000, regions={})
    assert out.status == "CLEAR"
    assert out.risk < 0.65


def test_watch_when_single_family_elevated():
    cb = ConsensusBuilder()
    cues = [_cue("visual.gaze_aversion", Modality.VISUAL, 6.0, "eyes"),
            _cue("visual.brow_flash", Modality.VISUAL, 6.0, "brow")]
    out = cb.build(cues=cues, calibrating=False, ts=1000,
                   regions={"visual.gaze_aversion": "eyes", "visual.brow_flash": "brow"})
    assert out.status == "WATCH"
    assert out.flag is False
    assert out.n_required == 2


def test_flag_only_under_two_gate():
    cb = ConsensusBuilder()
    cues = [_cue("visual.gaze_aversion", Modality.VISUAL, 7.0, "eyes", d=0.7),
            _cue("physio.heart_rate", Modality.PHYSIOLOGICAL, 7.0, "forehead", d=0.5)]
    out = cb.build(cues=cues, calibrating=False, ts=1000,
                   regions={"visual.gaze_aversion": "eyes", "physio.heart_rate": "forehead"})
    assert out.status == "FLAG"
    assert out.flag is True
    assert out.n_agree == 2


def test_unwired_families_shown_not_fresh():
    cb = ConsensusBuilder()
    out = cb.build(cues=[], calibrating=False, ts=1000, regions={})
    names = {f.name: f for f in out.families}
    assert names["audio"].wired is False and names["audio"].fresh is False
    assert names["linguistic"].wired is False
    assert names["visual"].wired is True


def test_active_cues_carry_region_for_telestrator():
    cb = ConsensusBuilder()
    cues = [_cue("visual.lip_press", Modality.VISUAL, 4.0, "mouth")]
    out = cb.build(cues=cues, calibrating=False, ts=1000,
                   regions={"visual.lip_press": "mouth"})
    assert out.active_cues[0].region == "mouth"
    assert out.active_cues[0].cue_id == "visual.lip_press"


def test_online_and_activity_fields_propagate():
    """online_families and family_activity are surfaced on FamilyVote; unwired stays False."""
    cb = ConsensusBuilder()
    out = cb.build(
        cues=[], calibrating=False, ts=1000, regions={},
        online_families={"visual"}, family_activity={"visual": 0.5},
    )
    names = {f.name: f for f in out.families}
    assert names["visual"].online is True
    assert names["visual"].activity == 0.5
    # physio: wired but not in online_families → offline
    assert names["physio"].online is False
    # audio: unwired → always offline regardless
    assert names["audio"].online is False
    assert names["audio"].activity == 0.0
