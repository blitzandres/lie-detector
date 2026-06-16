import math
import unittest
import wave
from array import array
from pathlib import Path
from tempfile import TemporaryDirectory

from blitz_engine import BlitzEngine
from blitz_engine.cli import main


class TextMVPTests(unittest.TestCase):
    def setUp(self):
        self.engine = BlitzEngine(modalities=["linguistic"])
        self.session = self.engine.new_session(
            baseline_texts=[
                "I drove to work and grabbed coffee before the meeting.",
                "My usual breakfast is eggs, toast, and tea at home.",
                "Last weekend I cleaned the apartment and watched a movie.",
                "I usually walk to the store in the afternoon for groceries.",
                "My morning routine starts with stretching and checking messages.",
            ],
            baseline_latencies_ms=[120, 140, 100, 110, 130],
            baseline_duration_s=120,
            consent=True,
            use_case="research",
            jurisdiction="CA-US",
        )

    def test_session_returns_structured_output(self):
        result = self.session.analyze_text(
            response_text="Honestly, I really do not know, um, I was basically with someone at that place.",
            question="Where were you Tuesday night?",
            response_latency_ms=900,
        )

        self.assertGreaterEqual(result.risk_score, 0.0)
        self.assertLessEqual(result.risk_score, 1.0)
        self.assertTrue(result.top_cues)
        self.assertEqual(result.quality_flags["input_mode"], "text")
        self.assertTrue(result.compliance["not_for_sole_decision"])

    def test_missing_consent_is_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.new_session(
                baseline_texts=["Neutral baseline text"] * 5,
                consent=False,
                use_case="research",
                jurisdiction="CA-US",
            )

    def test_cli_writes_json_report(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            baseline = tmp / "baseline.txt"
            response = tmp / "response.txt"
            output = tmp / "report.json"

            baseline.write_text(
                "\n".join([
                    "I drove to work and grabbed coffee before the meeting.",
                    "My usual breakfast is eggs, toast, and tea at home.",
                    "Last weekend I cleaned the apartment and watched a movie.",
                    "I usually walk to the store in the afternoon for groceries.",
                    "My morning routine starts with stretching and checking messages.",
                ])
            )
            response.write_text("Honestly, I really do not know, um, I was basically with someone at that place.")

            exit_code = main([
                "analyze-text",
                "--baseline-file", str(baseline),
                "--response-file", str(response),
                "--question", "Where were you Tuesday night?",
                "--response-latency-ms", "900",
                "--output", str(output),
            ])

            self.assertEqual(exit_code, 0)
            payload = output.read_text()
            self.assertIn('"risk_score"', payload)
            self.assertIn('"session_id"', payload)

    def test_multimodal_session_uses_audio_and_text(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            baseline_audio_1 = tmp / "baseline_1.wav"
            baseline_audio_2 = tmp / "baseline_2.wav"
            baseline_audio_3 = tmp / "baseline_3.wav"
            response_audio = tmp / "response.wav"
            self._write_tone_wav(baseline_audio_1, frequency=220, duration_s=1.5, amplitude=0.12, silent_tail_s=0.0)
            self._write_tone_wav(baseline_audio_2, frequency=230, duration_s=1.5, amplitude=0.11, silent_tail_s=0.0)
            self._write_tone_wav(baseline_audio_3, frequency=240, duration_s=1.5, amplitude=0.13, silent_tail_s=0.0)
            self._write_tone_wav(response_audio, frequency=440, duration_s=1.5, amplitude=0.38, silent_tail_s=0.3)

            engine = BlitzEngine(modalities=["linguistic", "audio"])
            session = engine.new_session(
                baseline_texts=[
                    "I drove to work and grabbed coffee before the meeting.",
                    "My usual breakfast is eggs, toast, and tea at home.",
                    "Last weekend I cleaned the apartment and watched a movie.",
                    "I usually walk to the store in the afternoon for groceries.",
                    "My morning routine starts with stretching and checking messages.",
                ],
                baseline_audio_files=[str(baseline_audio_1), str(baseline_audio_2), str(baseline_audio_3)],
                baseline_duration_s=120,
                consent=True,
                use_case="research",
                jurisdiction="CA-US",
            )
            result = session.analyze(
                response_text="Honestly, I really do not know, um, I was basically with someone at that place.",
                audio_path=str(response_audio),
                question="Where were you Tuesday night?",
                response_latency_ms=900,
            )

            self.assertIn("linguistic", result.channel_contributions)
            self.assertIn("audio", result.channel_contributions)
            self.assertGreaterEqual(len({cue["modality"] for cue in result.top_cues}), 2)

    def _write_tone_wav(self, path: Path, frequency: float, duration_s: float, amplitude: float, silent_tail_s: float) -> None:
        sample_rate = 16000
        total_samples = int(duration_s * sample_rate)
        silent_samples = int(silent_tail_s * sample_rate)
        tone_samples = max(0, total_samples - silent_samples)

        frames = array(
            "h",
            [
                int(
                    max(-32767, min(32767, amplitude * 32767 * math.sin(2 * math.pi * frequency * index / sample_rate)))
                )
                for index in range(tone_samples)
            ]
            + [0] * silent_samples,
        )

        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(frames.tobytes())


if __name__ == "__main__":
    unittest.main()
