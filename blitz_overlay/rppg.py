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
# Honest gate: only report a BPM when the spectral peak genuinely dominates the band
# (a real periodic pulse). On noisy / motion-corrupted webcam input the peak is not
# dominant → return None (abstain) rather than dress up noise as a heart rate.
MIN_PEAK_SNR = 6.0  # peak magnitude ÷ band-median magnitude


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
    band_spectrum = spectrum[band]
    peak_idx = int(np.argmax(band_spectrum))
    peak = float(band_spectrum[peak_idx])
    median = float(np.median(band_spectrum))
    # Abstain unless the pulse peak clearly dominates — honest "no lock" on noisy webcam input.
    if median <= 1e-9 or peak / median < MIN_PEAK_SNR:
        return None
    peak_freq = freqs[band][peak_idx]
    return float(peak_freq * 60.0)
