"""Personal baseline calibration for per-cue normalization."""

from __future__ import annotations

import statistics

MINIMUM_BASELINE_SECONDS = 90    # Research-updated minimum (was 30-60s)
IDEAL_BASELINE_SECONDS = 180


def compute_robust_z(value: float, baseline_values: list[float]) -> float:
    """
    Robust Z-score using Median Absolute Deviation.
    Resistant to outliers unlike standard Z.

    z_robust = (x - median) / (1.4826 × MAD)
    The 1.4826 factor makes MAD consistent with std under normality.
    """
    if len(baseline_values) < 5:
        raise ValueError(f"Need at least 5 baseline observations, got {len(baseline_values)}")

    med = statistics.median(baseline_values)
    mad = statistics.median([abs(v - med) for v in baseline_values])

    if mad == 0:
        return 0.0  # No variation in baseline

    return (value - med) / (1.4826 * mad)


def apply_empirical_bayes_shrinkage(z_raw: float, n_obs: int, n_target: int = 30) -> float:
    """
    Shrink Z-score toward 0 (population mean) when baseline is short.
    Prevents overconfident calibration from short baselines.

    lambda = min(1.0, n_obs / n_target)
    z_shrunk = lambda × z_raw + (1 - lambda) × 0
    """
    lam = min(1.0, n_obs / n_target)
    return lam * z_raw


class PersonalBaseline:
    """Stores per-cue baseline statistics for a single session."""

    REQUIRED_BASELINE_TOPICS = [
        "Tell me about your commute this morning.",
        "Describe your favorite meal.",
        "What did you do last weekend?"
    ]

    def __init__(self):
        self.trait_stats: dict[str, dict] = {}
        self.stress_stats: dict[str, dict] = {}
        self.baseline_duration_s: float = 0.0
        self.is_sufficient: bool = False
        self.observation_count: int = 0

    def record_baseline(self, cue_observations: dict[str, list[float]], duration_s: float):
        """Record baseline observations for all cues."""
        if not cue_observations:
            raise ValueError("cue_observations cannot be empty")

        trait_stats: dict[str, dict] = {}
        max_samples = 0
        for cue_id, values in cue_observations.items():
            if len(values) < 3:
                continue

            median = statistics.median(values)
            mad = statistics.median(abs(v - median) for v in values)
            spread = 1.4826 * mad if mad > 0 else 1.0
            trait_stats[cue_id] = {
                "median": median,
                "mad": mad,
                "spread": spread,
                "count": len(values),
            }
            max_samples = max(max_samples, len(values))

        if not trait_stats:
            raise ValueError("Need at least one cue with 3+ baseline observations")

        self.trait_stats = trait_stats
        self.baseline_duration_s = duration_s
        self.observation_count = max_samples
        self.is_sufficient = duration_s >= MINIMUM_BASELINE_SECONDS and len(trait_stats) >= 3

    def normalize(self, cue_id: str, raw_value: float) -> float:
        """Return robust Z-score vs personal baseline for a cue."""
        if cue_id not in self.trait_stats:
            return 0.0

        stats = self.trait_stats[cue_id]
        spread = stats["spread"] or 1.0
        z_raw = (raw_value - stats["median"]) / spread
        return apply_empirical_bayes_shrinkage(
            z_raw=z_raw,
            n_obs=stats["count"],
        )

    def quality_report(self) -> dict:
        """Return baseline quality assessment."""
        return {
            "duration_s": self.baseline_duration_s,
            "is_sufficient": self.is_sufficient,
            "minimum_required_s": MINIMUM_BASELINE_SECONDS,
            "ideal_duration_s": IDEAL_BASELINE_SECONDS,
            "tracked_cues": len(self.trait_stats),
            "observation_count": self.observation_count,
            "warning": None if self.is_sufficient else
                       f"Baseline {self.baseline_duration_s:.0f}s < required {MINIMUM_BASELINE_SECONDS}s"
        }
