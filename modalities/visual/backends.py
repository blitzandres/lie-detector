"""AU backends for the offline research visual analyzer (spec 2c).

Heavy models are LAZY: importing this module never imports torch/py-feat.
One backend runs at a time (M1/8GB sequential rule) — load → run → release.
"""
from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

_INSTALL_HINT = "pip install -e '.[research]'"


@dataclass
class VisualFrame:
    """One analyzed video frame: AU intensities + emotions + head pose."""

    ts_ms: int
    face_present: bool = False
    quality: float = 0.0                              # detector confidence [0,1]
    aus: dict = field(default_factory=dict)           # "AU01".. -> intensity [0,1]
    emotions: dict = field(default_factory=dict)      # "fear".. -> prob [0,1]
    head_pose: dict = field(default_factory=dict)     # yaw/pitch/roll degrees


class AUBackend(ABC):
    name: str = ""

    @abstractmethod
    def extract(self, video_path: str) -> list[VisualFrame]:
        """Run the model over the whole clip, then release it (sequential rule)."""


class StubBackend(AUBackend):
    """Deterministic backend for tests — returns pre-built frames."""

    name = "stub"

    def __init__(self, frames: list[VisualFrame]):
        self._frames = frames

    def extract(self, video_path: str) -> list[VisualFrame]:
        return list(self._frames)


class PyFeatBackend(AUBackend):
    """Py-Feat Detector v2 — 20 AUs w/ intensity, emotions, head pose in one pass."""

    name = "pyfeat"

    def __init__(self, skip_frames: int = 2):
        self.skip_frames = skip_frames   # analyze every Nth frame (speed on CPU)

    def extract(self, video_path: str) -> list[VisualFrame]:
        try:
            from feat import Detector
        except ImportError as e:
            raise RuntimeError(
                f"Py-Feat is not installed. Install the research extra: {_INSTALL_HINT}"
            ) from e
        detector = Detector()
        try:
            fex = detector.detect_video(video_path, skip_frames=self.skip_frames)
            frames: list[VisualFrame] = []
            au_cols = [c for c in fex.columns if c.upper().startswith("AU")]
            emo_cols = [c for c in ("anger", "disgust", "fear", "happiness",
                                    "sadness", "surprise", "neutral") if c in fex.columns]
            fps = getattr(fex, "fps", None) or 30.0
            for _, row in fex.iterrows():
                frame_no = int(row.get("frame", 0))
                frames.append(VisualFrame(
                    ts_ms=int(frame_no * 1000.0 / fps),
                    face_present=bool(row.get("FaceScore", 1.0) > 0),
                    quality=float(min(1.0, max(0.0, row.get("FaceScore", 0.9)))),
                    aus={c.upper(): float(row[c]) for c in au_cols},
                    emotions={c: float(row[c]) for c in emo_cols},
                    head_pose={"yaw": float(row.get("Yaw", 0.0)),
                               "pitch": float(row.get("Pitch", 0.0)),
                               "roll": float(row.get("Roll", 0.0))},
                ))
            return frames
        finally:
            del detector
            gc.collect()   # release before any other heavyweight loads (8GB rule)


class OpenGraphAUBackend(AUBackend):
    """Complementary AU detector for ensemble robustness — integration seam.

    Wire-up is deliberate follow-up work: OpenGraphAU has no pip package; it needs a
    cloned repo + checkpoint. The seam keeps `EnsembleBackend` ready for it.
    """

    name = "opengraphau"

    def extract(self, video_path: str) -> list[VisualFrame]:
        raise RuntimeError(
            "OpenGraphAU backend is a seam: clone github.com/lingjivoo/OpenGraphAU, "
            "download a checkpoint, and implement extract() against it. "
            "Py-Feat (default) covers AUs today."
        )


class LibreFaceBackend(AUBackend):
    """Documented fallback AU backend (LibreFace) — integration seam, same contract."""

    name = "libreface"

    def extract(self, video_path: str) -> list[VisualFrame]:
        raise RuntimeError(
            "LibreFace backend is a seam: pip install libreface, then implement "
            "extract() against libreface.get_facial_attributes(). "
            "Py-Feat (default) covers AUs today."
        )


class EnsembleBackend(AUBackend):
    """Agreement ensemble: mean AU intensity across two backends, frame-aligned.

    Backends run SEQUENTIALLY (never concurrently) — 8GB rule.
    """

    name = "ensemble"

    def __init__(self, primary: AUBackend, secondary: AUBackend):
        self.primary = primary
        self.secondary = secondary

    def extract(self, video_path: str) -> list[VisualFrame]:
        a = self.primary.extract(video_path)
        b = self.secondary.extract(video_path)
        merged = []
        for fa, fb in zip(a, b, strict=False):
            aus = dict(fa.aus)
            for k, v in fb.aus.items():
                aus[k] = (aus[k] + v) / 2.0 if k in aus else v
            merged.append(VisualFrame(
                ts_ms=fa.ts_ms,
                face_present=fa.face_present and fb.face_present,
                quality=(fa.quality + fb.quality) / 2.0,
                aus=aus, emotions=dict(fa.emotions), head_pose=dict(fa.head_pose),
            ))
        return merged
