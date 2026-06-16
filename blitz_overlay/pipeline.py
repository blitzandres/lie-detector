"""Per-connection live pipeline: feature frame -> consensus payload (spec §3)."""
from __future__ import annotations

import uuid
from pathlib import Path

from blitz_overlay.consensus import ConsensusBuilder
from blitz_overlay.cues.physio import RppgHeartRate
from blitz_overlay.cues.visual import VISUAL_DETECTORS
from blitz_overlay.logger import PredictionLogger
from blitz_overlay.schemas import Consensus, FeatureFrame
from core.calibration import RollingBaseline

EMIT_EVERY_MS = 100  # throttle consensus emission to ~10 Hz


class OverlaySession:
    def __init__(self, gate_threshold: float = 0.65, baseline_seconds: int = 90,
                 fps: float = 30.0, log_dir: str | Path = "logs"):
        self.session_id = uuid.uuid4().hex[:12]
        self.detectors = [cls() for cls in VISUAL_DETECTORS] + [RppgHeartRate(fps=fps)]
        self.baseline = RollingBaseline(baseline_seconds=baseline_seconds)
        self.consensus = ConsensusBuilder(gate_threshold=gate_threshold)
        self.logger = PredictionLogger(self.session_id, log_dir=log_dir)
        self.regions = {d.cue_id: d.region for d in self.detectors}
        self._last_emit_ts = -EMIT_EVERY_MS
        self._last_consensus: Consensus | None = None

    def process(self, raw: dict) -> Consensus:
        frame = FeatureFrame.from_dict(raw)

        if not frame.face_present:
            out = self.consensus.build(
                cues=[], calibrating=self.baseline.is_calibrating, ts=frame.ts,
                regions=self.regions, message="No subject detected — cues paused.")
            self._last_consensus = out
            self.logger.log(out, baseline_mode=self.baseline.mode)
            return out

        # 1) measure every cue, 2) feed the baseline once per frame, 3) score deviations
        measurements: dict[str, float] = {}
        for det in self.detectors:
            value = det.measure(frame)
            if value is not None:
                measurements[det.cue_id] = value
        self.baseline.update(measurements, ts_ms=frame.ts)

        # compute continuous family liveness (even below the z>=2 cue threshold)
        online_families: set[str] = set()
        family_activity: dict[str, float] = {}
        for det in self.detectors:
            if det.cue_id not in measurements:
                continue
            fam = det.family
            online_families.add(fam)
            z = abs(self.baseline.normalize(det.cue_id, measurements[det.cue_id]))
            level = max(0.0, min(1.0, z / 6.0))
            family_activity[fam] = max(family_activity.get(fam, 0.0), level)

        cues = []
        for det in self.detectors:
            if det.cue_id not in measurements:
                continue
            event = det.update(frame, self.baseline, value=measurements[det.cue_id])
            if event is not None:
                cues.append(event)

        out = self.consensus.build(
            cues=cues, calibrating=self.baseline.is_calibrating, ts=frame.ts,
            regions=self.regions,
            online_families=online_families, family_activity=family_activity)
        self._last_consensus = out
        self.logger.log(out, baseline_mode=self.baseline.mode)
        return out

    def should_emit(self, ts: int) -> bool:
        if ts - self._last_emit_ts >= EMIT_EVERY_MS:
            self._last_emit_ts = ts
            return True
        return False
