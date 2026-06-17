"""Append-only prediction logger (READINESS #8). Logging != learning.

Stores only derived decision data — status, posterior, per-family contributions, active
cue z-scores, baseline mode, weight-set version, schema version. NEVER raw biometric
(no landmarks/blendshapes/RGB), honoring the privacy posture (READINESS #15).
"""
from __future__ import annotations

import json
from pathlib import Path

from blitz_overlay.schemas import Consensus
from blitz_overlay.weights import WEIGHT_SET_VERSION


class PredictionLogger:
    def __init__(self, session_id: str, log_dir: str | Path = "logs"):
        self.session_id = session_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / f"predictions-{session_id}.jsonl"

    def log(self, consensus: Consensus, baseline_mode: str) -> None:
        record = {
            "session_id": self.session_id,
            "ts": consensus.ts,
            "status": consensus.status,
            "posterior": consensus.risk,
            "flag": consensus.flag,
            "bell_record": (consensus.bell or {}).get("record") if (consensus.bell or {}).get("just_rang") else None,
            "n_agree": consensus.n_agree,
            "n_required": consensus.n_required,
            "families": [f.to_dict() for f in consensus.families],
            "active_cues": [c.to_dict() for c in consensus.active_cues],
            "baseline_mode": baseline_mode,
            "weight_set_version": WEIGHT_SET_VERSION,
            "schema_version": consensus.schema_version,
            "ground_truth": None,  # empty audit slot; never auto-filled (no learning loop)
        }
        with self.path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
