"""Rolling buffer of per-frame cue activity so the content engine can pull a turn's window.

Each frame stores the cues that were "lit" (directed z >= the lit threshold) at that moment.
window(t0, t1) summarizes the behavioural rhythm during an answer for content↔cue fusion.
"""
from __future__ import annotations

from collections import deque

LIT_Z = 2.0  # a cue is "lit" in the timeline at directed z >= this


class TimelineBuffer:
    def __init__(self, retain_ms: int = 120000):
        self.retain_ms = retain_ms
        self._frames: deque[tuple[int, list[tuple[str, str, float]]]] = deque()

    def add(self, ts: int, cue_levels: list[tuple[str, str, float]]) -> None:
        """cue_levels: (cue_id, family, directed_z) for measured cues this frame."""
        lit = [(cid, fam, z) for (cid, fam, z) in cue_levels if z >= LIT_Z]
        self._frames.append((ts, lit))
        cutoff = ts - self.retain_ms
        while self._frames and self._frames[0][0] < cutoff:
            self._frames.popleft()

    def window(self, t0: int, t1: int) -> dict:
        frames = [(ts, lit) for ts, lit in self._frames if t0 <= ts <= t1]
        cue_ids: set[str] = set()
        families: set[str] = set()
        peak_z = 0.0
        max_sync = 0
        for _, lit in frames:
            fam_here = set()
            for cid, fam, z in lit:
                cue_ids.add(cid)
                families.add(fam)
                fam_here.add(fam)
                peak_z = max(peak_z, z)
            max_sync = max(max_sync, len(fam_here))
        return {
            "n_frames": len(frames),
            "cue_ids": sorted(cue_ids),
            "families": sorted(families),
            "peak_z": round(peak_z, 3),
            "max_families_synchronous": max_sync,
        }
