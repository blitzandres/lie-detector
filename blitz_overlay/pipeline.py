"""Per-connection live pipeline: feature frame -> consensus payload (spec §3)."""
from __future__ import annotations

import uuid
from pathlib import Path

from blitz_overlay.bell import BellController, map_sensitivity
from blitz_overlay.calibration_status import compute_calibration
from blitz_overlay.consensus import ConsensusBuilder
from blitz_overlay.content.fusion import fuse_turn
from blitz_overlay.content.judge import ContentJudge, StubContentJudge
from blitz_overlay.content.timeline import TimelineBuffer
from blitz_overlay.cues.audio import AUDIO_DETECTORS
from blitz_overlay.cues.linguistic import LINGUISTIC_DETECTORS
from blitz_overlay.cues.physio import RppgHeartRate
from blitz_overlay.cues.visual import VISUAL_DETECTORS
from blitz_overlay.logger import PredictionLogger
from blitz_overlay.schemas import Consensus, CueRow, FeatureFrame
from blitz_overlay.synchrony import SynchronyDetector
from core.calibration import RollingBaseline
from core.fusion.bayesian_fusion import fuse_by_family

EMIT_EVERY_MS = 100  # throttle consensus emission to ~10 Hz


class OverlaySession:
    def __init__(self, gate_threshold: float = 0.65, baseline_seconds: int = 90,
                 fps: float = 30.0, log_dir: str | Path = "logs",
                 content_judge: ContentJudge | None = None):
        self.session_id = uuid.uuid4().hex[:12]
        self.detectors = (
            [cls() for cls in VISUAL_DETECTORS]
            + [cls() for cls in AUDIO_DETECTORS]
            + [cls() for cls in LINGUISTIC_DETECTORS]
            + [RppgHeartRate(fps=fps)]
        )
        self.baseline = RollingBaseline(baseline_seconds=baseline_seconds)
        self.consensus = ConsensusBuilder(gate_threshold=gate_threshold)
        self.logger = PredictionLogger(self.session_id, log_dir=log_dir)
        self.regions = {d.cue_id: d.region for d in self.detectors}
        self._cue_family = {d.cue_id: d.family for d in self.detectors}
        self._last_emit_ts = -EMIT_EVERY_MS
        self._last_consensus: Consensus | None = None
        self._last_transcript_seq: int | None = None
        self.synchrony = SynchronyDetector()
        self.bell = BellController()
        self.timeline = TimelineBuffer()
        self.content_judge: ContentJudge = content_judge or StubContentJudge()
        self._turn_history: list[dict] = []
        # Active-calibration (hard gate): completes only when every producing cue has a base.
        self._det_meta = [(d.cue_id, d.family, d.cue_id.split(".")[-1]) for d in self.detectors]
        self._calib_target = baseline_seconds
        self._calib_timeout = baseline_seconds + 60   # safety escape so a dead channel can't strand us
        self._calibrated = False

    def _calibration(self) -> tuple[bool, dict]:
        obs = {cid: self.baseline.observation_count(cid) for cid, _, _ in self._det_meta}
        calibrating, calib = compute_calibration(
            self._det_meta, obs, self.baseline.elapsed_seconds, self._calib_target,
            timeout_s=self._calib_timeout, already_calibrated=self._calibrated)
        if not calibrating:
            self._calibrated = True
        return calibrating, calib

    def _apply_sensitivity(self, frame: FeatureFrame) -> None:
        if not frame.config:
            return
        s = frame.config.get("sensitivity")
        if s is None:
            return
        params = map_sensitivity(float(s))
        self.synchrony.set_params(lit_z=params["lit_z"], k=params["k"])
        self.bell.set_params(risk_floor=params["risk_floor"])

    def _build_cue_rows(self, directed_z: dict[str, float], measured: set[str]) -> list[CueRow]:
        rows = []
        for det in self.detectors:
            z = directed_z.get(det.cue_id, 0.0)
            rows.append(CueRow(
                cue_id=det.cue_id, family=det.family, region=det.region,
                label=det.cue_id.split(".")[-1],
                z=z, lit=z >= self.synchrony.lit_z, online=det.cue_id in measured,
            ))
        return rows

    def _family_of_cue(self, cue_id: str) -> str:
        return self._cue_family.get(cue_id, "visual")

    def process(self, raw: dict) -> Consensus:
        frame = FeatureFrame.from_dict(raw)

        self._apply_sensitivity(frame)

        # Per-utterance gate: a repeated transcript seq is treated as absent so the
        # rolling baseline samples linguistic cues per utterance, not per 30 Hz frame.
        if frame.transcript is not None:
            seq = frame.transcript.get("seq")
            if seq is not None and seq == self._last_transcript_seq:
                frame.transcript = None
            else:
                self._last_transcript_seq = seq

        if not frame.face_present:
            convergence = self.synchrony.update(frame.ts, [])
            bell = self.bell.update(frame.ts, convergence, 0.0)
            calibrating, calibration = self._calibration()
            if calibrating:
                calibration["guidance"] = "No face detected — look at the camera to calibrate."
            out = self.consensus.build(
                cues=[], calibrating=calibrating, ts=frame.ts,
                regions=self.regions, message="No subject detected — cues paused.",
                cue_rows=self._build_cue_rows({}, set()),
                convergence=convergence, bell=bell, calibration=calibration)
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
        directed_z: dict[str, float] = {}
        for det in self.detectors:
            if det.cue_id not in measurements:
                continue
            fam = det.family
            online_families.add(fam)
            raw_z = self.baseline.normalize(det.cue_id, measurements[det.cue_id])
            directed_z[det.cue_id] = raw_z * det.direction
            level = max(0.0, min(1.0, abs(raw_z) / 6.0))
            family_activity[fam] = max(family_activity.get(fam, 0.0), level)

        self.timeline.add(frame.ts, [(cid, self._family_of_cue(cid), z) for cid, z in directed_z.items()])

        cues = []
        for det in self.detectors:
            if det.cue_id not in measurements:
                continue
            event = det.update(frame, self.baseline, value=measurements[det.cue_id])
            if event is not None:
                cues.append(event)

        calibrating, calibration = self._calibration()

        cue_levels = [(cid, self._family_of_cue(cid), z) for cid, z in directed_z.items()]
        convergence = self.synchrony.update(frame.ts, cue_levels)
        posterior = 0.0 if calibrating else fuse_by_family(cues)["posterior"]
        bell = self.bell.update(frame.ts, convergence, posterior)
        cue_rows = self._build_cue_rows(directed_z, set(measurements.keys()))

        out = self.consensus.build(
            cues=cues, calibrating=calibrating, ts=frame.ts,
            regions=self.regions,
            online_families=online_families, family_activity=family_activity,
            cue_rows=cue_rows, convergence=convergence, bell=bell, calibration=calibration)
        self._last_consensus = out
        self.logger.log(out, baseline_mode=self.baseline.mode)
        return out

    def judge_turn(self, question: str, answer: str, t0: int, t1: int) -> dict:
        """Content-first verdict for one Q&A turn, fused with the cue activity in [t0, t1]."""
        verdict = self.content_judge.judge(question, answer, self._turn_history, self.baseline)
        cue_window = self.timeline.window(t0, t1)
        result = fuse_turn(verdict, cue_window)
        result["content"] = verdict.to_dict()
        result["cue_window"] = cue_window
        result["question"] = question
        if verdict.available:
            self._turn_history.append({"question": question, "answer": answer})
        return result

    def should_emit(self, ts: int) -> bool:
        if ts - self._last_emit_ts >= EMIT_EVERY_MS:
            self._last_emit_ts = ts
            return True
        return False
