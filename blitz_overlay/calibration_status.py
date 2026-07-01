"""Active-calibration status — hard-gated baseline readiness (option 2).

Pure function over per-cue observation counts (the rolling baseline already tracks these).
Calibration completes only when the time window has elapsed AND no cue is still "waiting"
(receiving data but short of its base) — so every channel that is *producing* signal gets a
full baseline before the system goes live. Cues with zero observations are "no-signal"
(their source isn't producing — muted mic, rPPG can't lock) and do NOT block, so calibration
can never hang on something physically impossible. A generous timeout is the final escape.
"""
from __future__ import annotations

MIN_OBS = 8  # observations before a cue's baseline is considered established

# Human guidance per channel when it still needs to be fed during calibration.
_HINTS = {
    "visual": "look at the camera",
    "audio": "speak to calibrate",
    "linguistic": "keep talking",
    "physio": "hold still & face the light",
}


def compute_calibration(
    detectors: list[tuple[str, str, str]],
    obs_counts: dict[str, int],
    elapsed_s: float,
    target_s: float,
    *,
    min_obs: int = MIN_OBS,
    timeout_s: float | None = None,
    already_calibrated: bool = False,
) -> tuple[bool, dict]:
    """Return (still_calibrating, calibration_payload).

    detectors: list of (cue_id, family, label). obs_counts: cue_id -> baseline sample count.
    """
    cues = []
    families: dict[str, dict] = {}
    any_waiting = False

    for cue_id, family, label in detectors:
        obs = int(obs_counts.get(cue_id, 0))
        if obs >= min_obs:
            status = "ready"
        elif obs >= 1:
            status = "waiting"
            any_waiting = True
        else:
            status = "no-signal"
        cues.append({"cue_id": cue_id, "family": family, "label": label,
                     "obs": obs, "status": status})
        fam = families.setdefault(
            family, {"ready": 0, "waiting": 0, "no_signal": 0, "total": 0, "status": "idle"})
        fam["total"] += 1
        fam["ready" if status == "ready" else "waiting" if status == "waiting" else "no_signal"] += 1

    for fam in families.values():
        if fam["waiting"]:
            fam["status"] = "waiting"
        elif fam["ready"]:
            fam["status"] = "ready"
        else:
            fam["status"] = "idle"

    time_done = elapsed_s >= target_s
    timed_out = timeout_s is not None and elapsed_s >= timeout_s
    complete = already_calibrated or (time_done and not any_waiting) or timed_out
    calibrating = not complete

    progress = max(0.0, min(1.0, elapsed_s / target_s)) if target_s > 0 else 1.0
    blocking = [c["cue_id"] for c in cues if c["status"] == "waiting"]
    needs = [name for name, fam in families.items() if fam["status"] in ("waiting", "idle")]

    payload = {
        "active": calibrating,
        "progress": round(progress, 3),
        "elapsed_s": round(elapsed_s, 1),
        "target_s": target_s,
        "min_obs": min_obs,
        "cues": cues,
        "families": families,
        "blocking": blocking,
        "needs": needs,
        "guidance": _guidance(families, calibrating, timed_out),
        "timed_out": timed_out,
    }
    return calibrating, payload


def _guidance(families: dict[str, dict], calibrating: bool, timed_out: bool) -> str:
    if not calibrating:
        return ("Calibrated — some channels had no signal." if timed_out
                else "Baseline ready.")
    needy = [name for name, fam in families.items() if fam["status"] in ("waiting", "idle")]
    if not needy:
        return "Hold steady — finishing baseline…"
    parts = [f"{name.capitalize()}: {_HINTS.get(name, 'feed this channel')}" for name in needy]
    return "Feed every channel — " + "; ".join(parts)
