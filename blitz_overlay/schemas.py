"""Versioned WebSocket contract between the browser and the Python engine.

Only derived feature vectors cross the wire — never raw video (spec §3, READINESS #15).
The browser mirrors SCHEMA_VERSION in apps/overlay-web/js/schema.js.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

SCHEMA_VERSION = "1.0"


class Region(str, Enum):
    """Telestrator anchor regions a cue can light up (spec §5)."""

    EYES = "eyes"
    BROW = "brow"
    MOUTH = "mouth"
    JAW = "jaw"
    FOREHEAD = "forehead"
    HEAD = "head"
    BODY = "body"


@dataclass
class FeatureFrame:
    """One browser → engine frame. Tiny: blendshapes + pose + a few derived scalars."""

    ts: int                                   # client timestamp (ms)
    face_present: bool = False
    confidence: float = 0.0                   # landmark confidence [0,1]
    blendshapes: dict = field(default_factory=dict)   # name -> coefficient [0,1]
    head_pose: dict = field(default_factory=dict)     # yaw/pitch/roll (degrees)
    geometry: dict = field(default_factory=dict)      # jaw_width_ratio, gaze_x, gaze_y, ear_*
    rppg: dict | None = None                  # {"forehead_rgb":[r,g,b], "cheek_rgb":[r,g,b]} or None
    audio: dict | None = None                 # {"f0":Hz, "energy":rms, "pause_ratio":0-1, "tremor":cv} or None
    transcript: dict | None = None            # {"text": str, "seq": int} or None
    config: dict | None = None                # {"sensitivity": 0..1} or None
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, d: dict) -> FeatureFrame:
        return cls(
            ts=int(d.get("ts", 0)),
            face_present=bool(d.get("face_present", False)),
            confidence=float(d.get("confidence", 0.0)),
            blendshapes=dict(d.get("blendshapes") or {}),
            head_pose=dict(d.get("head_pose") or {}),
            geometry=dict(d.get("geometry") or {}),
            rppg=(dict(d["rppg"]) if d.get("rppg") else None),
            audio=(dict(d["audio"]) if d.get("audio") else None),
            transcript=(dict(d["transcript"]) if d.get("transcript") else None),
            config=(dict(d["config"]) if d.get("config") else None),
            schema_version=str(d.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass
class FamilyVote:
    """One consensus voter (a modality family)."""

    name: str
    wired: bool          # is this family implemented in this stage?
    fresh: bool          # has it produced a recent, usable signal?
    vote: bool           # does it currently vote "flag"?
    contribution: float  # its log-odds contribution to combined risk
    online: bool = False    # wired family is receiving data this frame (face present + a measurement)
    activity: float = 0.0   # 0..1 live signal level for this family (max normalized |z| across its cues)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "wired": self.wired,
            "fresh": self.fresh,
            "vote": self.vote,
            "contribution": round(self.contribution, 4),
            "online": self.online,
            "activity": round(self.activity, 4),
        }


@dataclass
class ActiveCue:
    """A currently-firing cue, for telestrator anchoring."""

    cue_id: str
    region: str
    z: float
    confidence: float

    def to_dict(self) -> dict:
        return {
            "cue_id": self.cue_id,
            "region": self.region,
            "z": round(self.z, 3),
            "confidence": round(self.confidence, 3),
        }


@dataclass
class CueRow:
    """One row in the live Parallel Cue Verifier checklist."""

    cue_id: str
    family: str
    region: str
    label: str
    z: float
    lit: bool
    online: bool

    def to_dict(self) -> dict:
        return {
            "cue_id": self.cue_id,
            "family": self.family,
            "region": self.region,
            "label": self.label,
            "z": round(self.z, 3),
            "lit": self.lit,
            "online": self.online,
        }


@dataclass
class Consensus:
    """Engine → browser payload driving the overlay."""

    schema_version: str
    ts: int
    status: str            # CALIBRATING | CLEAR | WATCH | FLAG
    risk: float            # [0,1] combined posterior
    flag: bool             # earned two-gate FLAG (drives the red pulse)
    n_agree: int           # independent families currently voting flag
    n_required: int        # families required to agree (2)
    families: list[FamilyVote] = field(default_factory=list)
    active_cues: list[ActiveCue] = field(default_factory=list)
    cue_rows: list[CueRow] = field(default_factory=list)
    convergence: dict = field(default_factory=dict)
    bell: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    message: str = ""      # degradation/status message (spec §8)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "ts": self.ts,
            "status": self.status,
            "risk": round(self.risk, 4),
            "flag": self.flag,
            "n_agree": self.n_agree,
            "n_required": self.n_required,
            "families": [f.to_dict() for f in self.families],
            "active_cues": [c.to_dict() for c in self.active_cues],
            "cue_rows": [r.to_dict() for r in self.cue_rows],
            "convergence": self.convergence,
            "bell": self.bell,
            "calibration": self.calibration,
            "message": self.message,
        }
