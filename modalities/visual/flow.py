"""Optical-flow seam: temporal motion dynamics for micro-expression spotting (spec 2c)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

_INSTALL_HINT = "pip install -e '.[research]'"


@dataclass
class FlowSample:
    ts_ms: int
    mean_magnitude: float    # average pixel motion this step
    peak_magnitude: float    # strongest local motion this step


class FlowSource(ABC):
    @abstractmethod
    def extract(self, video_path: str) -> list[FlowSample]: ...


class StubFlowSource(FlowSource):
    def __init__(self, samples: list[FlowSample]):
        self._samples = samples

    def extract(self, video_path: str) -> list[FlowSample]:
        return list(self._samples)


class FarnebackFlow(FlowSource):
    """Dense Farneback optical flow via OpenCV — cheap CPU pass, lazy import."""

    def __init__(self, step: int = 3, resize_width: int = 320):
        self.step = step
        self.resize_width = resize_width

    def extract(self, video_path: str) -> list[FlowSample]:
        try:
            import cv2
        except ImportError as e:
            raise RuntimeError(
                f"OpenCV is not installed. Install the research extra: {_INSTALL_HINT}"
            ) from e
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        samples: list[FlowSample] = []
        prev = None
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % self.step:
                idx += 1
                continue
            h, w = frame.shape[:2]
            scale = self.resize_width / float(w)
            gray = cv2.cvtColor(
                cv2.resize(frame, (self.resize_width, int(h * scale))),
                cv2.COLOR_BGR2GRAY)
            if prev is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag = (flow[..., 0] ** 2 + flow[..., 1] ** 2) ** 0.5
                samples.append(FlowSample(
                    ts_ms=int(idx * 1000.0 / fps),
                    mean_magnitude=float(mag.mean()),
                    peak_magnitude=float(mag.max()),
                ))
            prev = gray
            idx += 1
        cap.release()
        return samples
