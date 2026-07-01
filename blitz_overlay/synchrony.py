"""SynchronyDetector — temporal co-firing detector over the existing per-cue z-scores.

This is pure aggregation: it never inspects detection internals, only (cue_id, family, z)
tuples the pipeline already computes. A cue is "lit" when its directed z >= lit_z; a
"burst" is >=k lit cues spanning >=2 independent families within a short rolling window
(so cues that peak a few frames apart still count as "the same moment").
"""
from __future__ import annotations


class SynchronyDetector:
    def __init__(self, window_ms: int = 1000, lit_z: float = 2.0, k: int = 3):
        self.window_ms = window_ms
        self.lit_z = lit_z
        self.k = k
        # cue_id -> (last_lit_ts, family, z)
        self._lit: dict[str, tuple[int, str, float]] = {}

    def set_params(self, *, lit_z: float | None = None, k: int | None = None) -> None:
        if lit_z is not None:
            self.lit_z = lit_z
        if k is not None:
            self.k = k

    def update(self, ts: int, cue_levels: list[tuple[str, str, float]]) -> dict:
        """cue_levels: (cue_id, family, directed_z) for cues measured this frame."""
        for cue_id, family, z in cue_levels:
            if z >= self.lit_z:
                self._lit[cue_id] = (ts, family, z)
        cutoff = ts - self.window_ms
        self._lit = {cid: v for cid, v in self._lit.items() if v[0] >= cutoff}

        lit_cue_ids = list(self._lit.keys())
        families_lit = sorted({v[1] for v in self._lit.values()})
        n_lit = len(lit_cue_ids)
        n_families = len(families_lit)
        peak_z = max((v[2] for v in self._lit.values()), default=0.0)
        burst = n_lit >= self.k and n_families >= 2
        return {
            "n_lit": n_lit,
            "n_families": n_families,
            "lit_cue_ids": lit_cue_ids,
            "families_lit": families_lit,
            "peak_z": round(peak_z, 3),
            "burst": burst,
        }
