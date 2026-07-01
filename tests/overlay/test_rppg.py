import math

from blitz_overlay.rppg import chrom_signal, estimate_bpm


def _synth_rgb(bpm, n, fps):
    """Green channel pulsates at bpm; R/B steadier — mimics a clean rPPG signal."""
    f = bpm / 60.0
    out = []
    for i in range(n):
        t = i / fps
        g = 120 + 8 * math.sin(2 * math.pi * f * t)
        r = 180 + 1.5 * math.sin(2 * math.pi * f * t + 0.5)
        b = 110 + 1.0 * math.sin(2 * math.pi * f * t + 1.0)
        out.append([r, g, b])
    return out


def test_estimate_bpm_recovers_known_frequency():
    fps = 30
    samples = _synth_rgb(bpm=72, n=fps * 10, fps=fps)
    bpm = estimate_bpm(samples, fps=fps)
    assert 66 <= bpm <= 78  # within a few bpm of 72


def test_estimate_bpm_tracks_elevated_rate():
    fps = 30
    bpm = estimate_bpm(_synth_rgb(bpm=102, n=fps * 10, fps=fps), fps=fps)
    assert 95 <= bpm <= 110


def test_estimate_bpm_returns_none_when_too_few_samples():
    assert estimate_bpm([[180, 120, 110]] * 10, fps=30) is None


def test_chrom_signal_length_matches_input():
    sig = chrom_signal(_synth_rgb(bpm=72, n=90, fps=30))
    assert len(sig) == 90


def test_estimate_bpm_abstains_on_noise():
    """No real pulse (random skin-color noise) -> abstain, not a fake BPM."""
    import random
    rng = random.Random(0)
    samples = [[180 + rng.gauss(0, 3), 120 + rng.gauss(0, 3), 110 + rng.gauss(0, 3)]
               for _ in range(300)]
    assert estimate_bpm(samples, fps=30) is None


def test_estimate_bpm_abstains_on_flat_input():
    """Perfectly flat ROI (no variation at all) -> abstain."""
    assert estimate_bpm([[180.0, 120.0, 110.0]] * 300, fps=30) is None
