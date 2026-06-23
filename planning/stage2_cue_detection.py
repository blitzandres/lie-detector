#!/usr/bin/env python3
"""Stage-2 cue-detection gate check.

Usage:
    python3 planning/stage2_cue_detection.py --check

The 2A-3 gate: at least MIN_CUES cues across at least MIN_FAMILIES independent families must be
live in the engine before STAGE2_3D_OVERLAY work is allowed to start. This script counts the
real detectors registered in the live pipeline (no mocks) and prints PASS/FAIL.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable when run as `python3 planning/stage2_cue_detection.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blitz_overlay.pipeline import OverlaySession  # noqa: E402

MIN_CUES = 12
MIN_FAMILIES = 3


def gate_status() -> dict:
    session = OverlaySession(baseline_seconds=0)
    by_family: dict[str, list[str]] = {}
    for det in session.detectors:
        by_family.setdefault(det.family, []).append(det.cue_id)
    total = sum(len(v) for v in by_family.values())
    n_families = len(by_family)
    passed = total >= MIN_CUES and n_families >= MIN_FAMILIES
    return {
        "total_cues": total,
        "n_families": n_families,
        "by_family": {k: sorted(v) for k, v in sorted(by_family.items())},
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage-2 cue-detection gate check")
    parser.add_argument("--check", action="store_true", help="run the 2A-3 gate check")
    args = parser.parse_args()
    if not args.check:
        parser.print_help()
        return 0

    s = gate_status()
    print(f"2A-3 GATE  (need >= {MIN_CUES} cues across >= {MIN_FAMILIES} families)")
    print(f"  total cues : {s['total_cues']}")
    print(f"  families   : {s['n_families']}")
    for fam, ids in s["by_family"].items():
        print(f"    - {fam:<11} {len(ids):>2}  {', '.join(c.split('.')[-1] for c in ids)}")
    verdict = "PASS ✅" if s["passed"] else "FAIL ❌"
    print(f"  GATE       : {verdict}")
    print(f"  3D overlay : {'UNBLOCKED' if s['passed'] else 'BLOCKED (deepen cue detection first)'}")
    return 0 if s["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
