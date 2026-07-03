"""Runnable text-first Blitz Engine MVP."""

from __future__ import annotations

import time
import uuid

from core.calibration.baseline import PersonalBaseline
from core.fusion.bayesian_fusion import DEFAULT_PRIOR, convergence_gate_passed, fuse
from core.schemas.cue_event import BlitzOutput
from modalities.audio import AudioAnalyzer
from modalities.linguistic import LinguisticAnalyzer

ALLOWED_MODALITIES = {"linguistic", "audio", "visual"}


class BlitzEngine:
    """Coordinates enabled modalities and creates analysis sessions."""

    def __init__(
        self,
        modalities: list[str] | None = None,
        prior: float = DEFAULT_PRIOR,
        convergence_threshold: float = 0.65,
        visual_analyzer=None,
    ):
        self.modalities = modalities or ["linguistic"]
        unsupported = [modality for modality in self.modalities if modality not in ALLOWED_MODALITIES]
        if unsupported:
            raise ValueError(
                f"Unsupported modalities for current MVP: {', '.join(unsupported)}. "
                "Implemented: linguistic, audio, visual (research tier)."
            )

        self.prior = prior
        self.convergence_threshold = convergence_threshold
        self.linguistic = LinguisticAnalyzer() if "linguistic" in self.modalities else None
        self.audio = AudioAnalyzer() if "audio" in self.modalities else None
        if "visual" in self.modalities:
            # Import here so the base engine never pulls the research-tier module chain.
            from modalities.visual.analyzer import VisualAnalyzer
            self.visual = visual_analyzer or VisualAnalyzer()
        else:
            self.visual = None

    def new_session(
        self,
        *,
        baseline_texts: list[str] | None = None,
        baseline_audio_files: list[str] | None = None,
        baseline_video_files: list[str] | None = None,
        consent: bool,
        use_case: str,
        jurisdiction: str,
        baseline_duration_s: float | None = None,
        baseline_latencies_ms: list[int] | None = None,
    ) -> BlitzSession:
        if not consent:
            raise ValueError("Consent must be declared to create a session.")

        if not baseline_texts and not baseline_audio_files and not baseline_video_files:
            raise ValueError("Provide baseline_texts, baseline_audio_files or baseline_video_files")

        baseline = PersonalBaseline()
        observations: dict[str, list[float]] = {}
        if self.linguistic:
            if not baseline_texts:
                raise ValueError("baseline_texts is required when linguistic modality is enabled")
            observations.update(
                self.linguistic.build_baseline_observations(
                    baseline_texts=baseline_texts,
                    baseline_latencies_ms=baseline_latencies_ms,
                )
            )
        if self.audio:
            if not baseline_audio_files:
                raise ValueError("baseline_audio_files is required when audio modality is enabled")
            observations.update(self.audio.build_baseline_observations(baseline_audio_files))
        if self.visual:
            if not baseline_video_files:
                raise ValueError("baseline_video_files is required when visual modality is enabled")
            if len(baseline_video_files) < 3:
                raise ValueError("visual baseline needs at least 3 clips (3+ observations per cue)")
            observations.update(self.visual.build_baseline_observations(baseline_video_files))

        if baseline_duration_s is None:
            text_duration = float(len(baseline_texts) * 30) if baseline_texts else 0.0
            audio_duration = float(len(baseline_audio_files) * 20) if baseline_audio_files else 0.0
            video_duration = float(len(baseline_video_files) * 30) if baseline_video_files else 0.0
            baseline_duration_s = max(90.0, text_duration + audio_duration + video_duration)

        baseline.record_baseline(observations, duration_s=baseline_duration_s)
        return BlitzSession(
            engine=self,
            baseline=baseline,
            use_case=use_case,
            jurisdiction=jurisdiction,
            consent=consent,
        )


class BlitzSession:
    """Single-subject calibrated analysis session."""

    def __init__(
        self,
        engine: BlitzEngine,
        baseline: PersonalBaseline,
        use_case: str,
        jurisdiction: str,
        consent: bool,
    ):
        self.engine = engine
        self.baseline = baseline
        self.use_case = use_case
        self.jurisdiction = jurisdiction
        self.consent = consent
        self.session_id = str(uuid.uuid4())
        self.question_counter = 0

    def analyze_text(
        self,
        response_text: str,
        question: str | None = None,
        question_id: str | None = None,
        response_latency_ms: int | None = None,
    ) -> BlitzOutput:
        return self.analyze(
            response_text=response_text,
            question=question,
            question_id=question_id,
            response_latency_ms=response_latency_ms,
        )

    def analyze_audio(
        self,
        audio_path: str,
        question: str | None = None,
        question_id: str | None = None,
    ) -> BlitzOutput:
        return self.analyze(
            audio_path=audio_path,
            question=question,
            question_id=question_id,
        )

    def analyze(
        self,
        response_text: str | None = None,
        audio_path: str | None = None,
        video_path: str | None = None,
        question: str | None = None,
        question_id: str | None = None,
        response_latency_ms: int | None = None,
    ) -> BlitzOutput:
        if not response_text and not audio_path and not video_path:
            raise ValueError("Provide response_text, audio_path or video_path")

        self.question_counter += 1
        resolved_question_id = question_id or f"q{self.question_counter}"
        timestamp_ms = int(time.time() * 1000)
        cues = []

        if response_text is not None:
            if not response_text.strip():
                raise ValueError("response_text cannot be empty")
            if self.engine.linguistic:
                cues.extend(
                    self.engine.linguistic.analyze(
                        text=response_text,
                        question_id=resolved_question_id,
                        baseline=self.baseline,
                        response_latency_ms=response_latency_ms,
                        timestamp_ms=timestamp_ms,
                    )
                )

        if audio_path is not None:
            if self.engine.audio is None:
                raise ValueError("Audio modality is not enabled for this session")
            cues.extend(
                self.engine.audio.analyze(
                    wav_path=audio_path,
                    question_id=resolved_question_id,
                    baseline=self.baseline,
                    timestamp_ms=timestamp_ms,
                )
            )

        if video_path is not None:
            if self.engine.visual is None:
                raise ValueError("Visual modality is not enabled for this session")
            cues.extend(
                self.engine.visual.analyze(
                    video_path,
                    question_id=resolved_question_id,
                    baseline=self.baseline,
                    timestamp_ms=timestamp_ms,
                )
            )

        fused = fuse(cues, prior=self.engine.prior)
        posterior = fused["posterior"]
        gate_passed = convergence_gate_passed(cues, threshold=self.engine.convergence_threshold, posterior=posterior)
        narrative = self._build_narrative(
            question=question,
            response_text=response_text,
            posterior=posterior,
            gate_passed=gate_passed,
            top_cues=fused["top_cues"],
        )

        quality_flags = {
            "baseline": self.baseline.quality_report(),
            "implemented_modalities": self.engine.modalities,
            "input_mode": (
                "multimodal" if sum(x is not None for x in (response_text, audio_path, video_path)) > 1
                else "text" if response_text else "audio" if audio_path else "video"
            ),
            "convergence_gate_passed": gate_passed,
        }

        return BlitzOutput(
            session_id=self.session_id,
            question_id=resolved_question_id,
            timestamp_ms=timestamp_ms,
            risk_score=posterior,
            uncertainty=fused["uncertainty"],
            confidence_interval=fused["confidence_interval"],
            channel_contributions=fused["channel_contributions"],
            top_cues=[self._serialize_cue(cue) for cue in fused["top_cues"]],
            quality_flags=quality_flags,
            narrative=narrative,
            compliance={
                "not_for_sole_decision": True,
                "use_case": self.use_case,
                "consent_declared": self.consent,
                "jurisdiction": self.jurisdiction,
            },
        )

    def _build_narrative(
        self,
        question: str | None,
        response_text: str,
        posterior: float,
        gate_passed: bool,
        top_cues: list,
    ) -> str:
        cue_names = ", ".join(cue.cue_id.rsplit(".", 1)[-1] for cue in top_cues[:3]) or "no strong cues"
        status = "elevated" if gate_passed else "below convergence threshold"
        question_text = f" Question: {question}." if question else ""
        return (
            f"Text-only MVP assessment is {status} with posterior {posterior:.2f}.{question_text} "
            f"Primary linguistic drivers: {cue_names}. "
            f"This output is a research aid and not a factual lie determination."
        )

    def _serialize_cue(self, cue) -> dict:
        return {
            "cue_id": cue.cue_id,
            "modality": cue.modality.value,
            "timestamp_ms": cue.timestamp_ms,
            "phase": cue.phase.value,
            "raw_value": cue.raw_value,
            "z_score": cue.z_score,
            "llr": cue.llr,
            "quality": cue.quality,
            "question_id": cue.question_id,
            "on_word": cue.on_word,
            "effect_size_d": cue.effect_size_d,
            "reliability_tier": cue.reliability_tier,
        }
