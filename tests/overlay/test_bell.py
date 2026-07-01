"""Tests for BellController (earned, debounced, sustained) + map_sensitivity."""
from blitz_overlay.bell import BellController, map_sensitivity


def _burst(ok=True, families=("visual", "physio"), cues=("a", "b", "c")):
    return {"burst": ok, "lit_cue_ids": list(cues), "families_lit": list(families)}


def test_silent_until_sustained():
    b = BellController(hold_ms=1500, risk_floor=0.65)
    s = b.update(0, _burst(), posterior=0.8)
    assert s["ringing"] is False and s["just_rang"] is False
    s = b.update(1000, _burst(), posterior=0.8)   # only 1.0s held
    assert s["just_rang"] is False
    s = b.update(1600, _burst(), posterior=0.8)   # 1.6s held -> ring
    assert s["just_rang"] is True
    assert s["ringing"] is True
    assert s["label"] == "strong deception-pattern convergence"


def test_just_rang_fires_once_per_episode():
    b = BellController(hold_ms=1500, risk_floor=0.65)
    b.update(0, _burst(), posterior=0.8)
    b.update(1600, _burst(), posterior=0.8)       # rings
    s = b.update(1700, _burst(), posterior=0.8)   # still held -> no second edge
    assert s["just_rang"] is False
    assert s["ringing"] is True


def test_re_arms_after_condition_drops():
    b = BellController(hold_ms=1500, risk_floor=0.65)
    b.update(0, _burst(), posterior=0.8)
    b.update(1600, _burst(), posterior=0.8)       # rings
    s = b.update(1700, _burst(ok=False), posterior=0.8)  # burst drops -> reset
    assert s["ringing"] is False and s["just_rang"] is False
    b.update(1800, _burst(), posterior=0.8)
    s = b.update(3400, _burst(), posterior=0.8)   # held 1.6s again -> rings again
    assert s["just_rang"] is True


def test_risk_floor_blocks_bell():
    b = BellController(hold_ms=1500, risk_floor=0.65)
    b.update(0, _burst(), posterior=0.5)          # below risk_floor
    s = b.update(2000, _burst(), posterior=0.5)
    assert s["just_rang"] is False and s["ringing"] is False


def test_record_on_ring():
    b = BellController(hold_ms=1500, risk_floor=0.65)
    b.update(0, _burst(cues=("x", "y", "z"), families=("visual", "audio")), posterior=0.9)
    s = b.update(1600, _burst(cues=("x", "y", "z"), families=("visual", "audio")), posterior=0.9)
    assert s["just_rang"] is True
    rec = s["record"]
    assert rec["cue_ids"] == ["x", "y", "z"]
    assert rec["families"] == ["visual", "audio"]
    assert rec["risk"] == 0.9
    assert rec["ts"] == 1600


def test_map_sensitivity_endpoints():
    lo = map_sensitivity(0.0)
    assert lo == {"k": 3, "lit_z": 2.0, "risk_floor": 0.65}
    hi = map_sensitivity(1.0)
    assert hi == {"k": 2, "lit_z": 1.5, "risk_floor": 0.45}


def test_map_sensitivity_clamps():
    assert map_sensitivity(-5.0) == map_sensitivity(0.0)
    assert map_sensitivity(5.0) == map_sensitivity(1.0)
