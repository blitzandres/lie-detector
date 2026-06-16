"""rPPG heart-rate estimation from ROI mean-RGB time series (CHROM method).

The browser samples forehead/cheek ROI mean color each frame and streams it. The engine
buffers ~10s and estimates BPM by band-limited spectral peak-picking on the CHROM signal.
Pure DSP, CPU-only, numpy (EXECUTION_ARCHITECTURE B.2). This is the second independent
family — without it the two-gate can never FLAG (spec §11).
"""
from __future__ import annotations

import numpy as np

MIN_SAMPLES = 64
HR_LOW_HZ = 0.7   # 42 bpm
HR_HIGH_HZ = 4.0  # 240 bpm


def chrom_signal(rgb_samples: list[list[float]]) -> list[float]:
    """De Haan & Jeanne CHROM: combine normalized RGB into a pulse signal."""
    arr = np.asarray(rgb_samples, dtype=float)  # (N, 3)
    mean = arr.mean(axis=0)
    mean[mean == 0] = 1.0
    norm = arr / mean
    r, g, b = norm[:, 0], norm[:, 1], norm[:, 2]
    x = 3 * r - 2 * g
    y = 1.5 * r + g - 1.5 * b
    sx, sy = x.std(), y.std()
    alpha = (sx / sy) if sy > 1e-9 else 1.0
    signal = x - alpha * y
    return signal.tolist()


def estimate_bpm(rgb_samples: list[list[float]], fps: float) -> float | None:
    """Return dominant heart-rate (bpm) within the physiological band, or None."""
    if len(rgb_samples) < MIN_SAMPLES or fps <= 0:
        return None
    sig = np.asarray(chrom_signal(rgb_samples), dtype=float)
    sig = sig - sig.mean()
    if sig.std() < 1e-9:
        return None
    windowed = sig * np.hanning(len(sig))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fps)
    band = (freqs >= HR_LOW_HZ) & (freqs <= HR_HIGH_HZ)
    if not band.any():
        return None
    peak_freq = freqs[band][int(np.argmax(spectrum[band]))]
    return float(peak_freq * 60.0)
