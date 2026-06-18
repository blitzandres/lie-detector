"""Consensus builder: turns scored cues into the honest status payload (spec §6–§8)."""
from __future__ import annotations

from blitz_overlay.schemas import SCHEMA_VERSION, ActiveCue, Consensus, FamilyVote
from core.fusion.bayesian_fusion import family_of, fuse_by_family, two_gate

# Families displayed as voters, in panel order. Stage 1 wires visual + physio + audio.
WIRED_FAMILIES = {"visual", "physio", "audio", "linguistic"}
PANEL_FAMILIES = ["visual", "physio", "audio", "linguistic"]

WATCH_RISK = 0.45  # risk above this (but not a FLAG) shows WATCH


class ConsensusBuilder:
    def __init__(self, gate_threshold: float = 0.65):
        self.gate_threshold = gate_threshold

    def build(self, cues, calibrating: bool, ts: int, regions: dict[str, str],
              message: str = "",
              online_families: set[str] | None = None,
              family_activity: dict[str, float] | None = None,
              cue_rows=None, convergence=None, bell=None, calibration=None) -> Consensus:
        fused = fuse_by_family(cues)
        gate = two_gate(fused, threshold=self.gate_threshold)
        risk = fused["posterior"]
        votes = fused.get("family_votes", {})
        contrib = fused.get("families", {})
        fresh_families = {family_of(c.modality) for c in cues}

        families = []
        for name in PANEL_FAMILIES:
            wired = name in WIRED_FAMILIES
            families.append(FamilyVote(
                name=name,
                wired=wired,
                fresh=wired and name in fresh_families,
                vote=bool(votes.get(name, False)) and wired,
                contribution=float(contrib.get(name, 0.0)),
                online=wired and name in (online_families or set()),
                activity=float((family_activity or {}).get(name, 0.0)),
            ))

        active = [
            ActiveCue(cue_id=c.cue_id, region=regions.get(c.cue_id, "head"),
                      z=c.z_score, confidence=c.quality)
            for c in sorted(cues, key=lambda c: abs(c.z_score), reverse=True)
        ]

        if calibrating:
            status, flag = "CALIBRATING", False
            risk = 0.0
            message = message or "Calibrating personal baseline — no flags permitted yet."
        elif gate["flag"]:
            status, flag = "FLAG", True
        elif risk >= WATCH_RISK or any(f.vote for f in families):
            status, flag = "WATCH", False
        else:
            status, flag = "CLEAR", False

        # Honest cap (spec §8): if fewer than 2 wired families are fresh, FLAG is unreachable.
        if status == "WATCH" and len([f for f in families if f.fresh]) < 2 and not message:
            message = "Only one family active — capped at WATCH until a second family agrees."

        return Consensus(
            schema_version=SCHEMA_VERSION,
            ts=ts,
            status=status,
            risk=risk,
            flag=flag,
            n_agree=gate["n_agree"] if not calibrating else 0,
            n_required=gate["n_required"],
            families=families,
            active_cues=active,
            cue_rows=cue_rows or [],
            convergence=convergence or {},
            bell=bell or {},
            calibration=calibration or {},
            message=message,
        )
