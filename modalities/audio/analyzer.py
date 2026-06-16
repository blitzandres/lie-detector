"""WAV-based audio cue extractor for the Blitz Engine MVP."""

from __future__ import annotations

import array
import math
import statistics
import wave
from collections.abc import Sequence
from pathlib import Path

from core.schemas.cue_event import CueEvent, Modality, Phase


class AudioAnalyzer:
    """Extracts lightweight audio cues from uncompressed WAV files."""

    cue_specs = {
        "audio.energy_variability": {"effect_size_d": 0.24, "reliability_tier": 3},
        "audio.silence_ratio": {"effect_size_d": 0.22, "reliability_tier": 2},
        "audio.zero_crossing_rate": {"effect_size_d": 0.18, "reliability_tier": 3},
        "audio.peak_to_rms": {"effect_size_d": 0.20, "reliability_tier": 3},
        "audio.soft_segment_ratio": {"effect_size_d": 0.21, "reliability_tier": 2},
    }

    def load_wav(self, wav_path: str) -> tuple[list[float], int]:
        path = Path(wav_path)
        if path.suffix.lower() != ".wav":
            raise ValueError(f"AudioAnalyzer expects a .wav file, got {path.name}")

        with wave.open(str(path), "rb") as handle:
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            frames = handle.readframes(handle.getnframes())

        if sample_width != 2:
            raise ValueError("Only 16-bit PCM WAV files are supported in the MVP")

        samples = array.array("h")
        samples.frombytes(frames)
        if channels > 1:
            samples = array.array("h", samples[::channels])

        normalized = [sample / 32768.0 for sample in samples]
        return normalized, sample_rate

    def _windowed(self, samples: Sequence[float], window_size: int) -> list[list[float]]:
        return [list(samples[i : i + window_size]) for i in range(0, len(samples), window_size) if samples[i : i + window_size]]

    def _rms(self, samples: Sequence[float]) -> float:
        if not samples:
            return 0.0
        return math.sqrt(sum(sample * sample for sample in samples) / len(samples))

    def _zero_crossing_rate(self, samples: Sequence[float]) -> float:
        if len(samples) < 2:
            return 0.0
        crossings = 0
        for left, right in zip(samples, samples[1:], strict=False):
            if (left >= 0 > right) or (left < 0 <= right):
                crossings += 1
        return crossings / (len(samples) - 1)

    def extract_features(self, wav_path: str) -> dict[str, float]:
        samples, sample_rate = self.load_wav(wav_path)
        if not samples:
            raise ValueError("WAV file contains no samples")

        window_size = max(1, sample_rate // 10)
        windows = self._windowed(samples, window_size)
        window_rms = [self._rms(window) for window in windows]
        overall_rms = self._rms(samples)
        peak = max(abs(sample) for sample in samples)
        silence_threshold = max(0.01, overall_rms * 0.35)
        silent_windows = [window for window in window_rms if window < silence_threshold]
        soft_windows = [window for window in window_rms if silence_threshold <= window < silence_threshold * 1.8]

        return {
            "audio.energy_variability": statistics.pstdev(window_rms) / (statistics.fmean(window_rms) or 1.0) if len(window_rms) > 1 else 0.0,
            "audio.silence_ratio": len(silent_windows) / max(1, len(window_rms)),
            "audio.zero_crossing_rate": self._zero_crossing_rate(samples),
            "audio.peak_to_rms": peak / (overall_rms or 1.0),
            "audio.soft_segment_ratio": len(soft_windows) / max(1, len(window_rms)),
        }

    def build_baseline_observations(self, wav_paths: list[str]) -> dict[str, list[float]]:
        observations = {cue_id: [] for cue_id in self.cue_specs}
        for wav_path in wav_paths:
            for cue_id, value in self.extract_features(wav_path).items():
                observations[cue_id].append(value)
        return observations

    def analyze(
        self,
        wav_path: str,
        question_id: str,
        baseline,
        timestamp_ms: int = 0,
    ) -> list[CueEvent]:
        features = self.extract_features(wav_path)
        cues: list[CueEvent] = []
        for cue_id, raw_value in features.items():
            spec = self.cue_specs[cue_id]
            z_score = baseline.normalize(cue_id, raw_value)
            quality = 0.9
            cues.append(
                CueEvent(
                    cue_id=cue_id,
                    modality=Modality.AUDIO,
                    timestamp_ms=timestamp_ms,
                    phase=Phase.RESPONSE,
                    raw_value=raw_value,
                    z_score=z_score,
                    llr=0.0,
                    quality=quality,
                    question_id=question_id,
                    effect_size_d=spec["effect_size_d"],
                    reliability_tier=spec["reliability_tier"],
                )
            )
        return cues
