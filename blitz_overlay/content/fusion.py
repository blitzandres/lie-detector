"""Content-primary fusion of a content verdict with the cue activity from the same window.

Content drives the verdict; a synchronous multi-family cue burst in the answer's window
*confirms* it (convergence → higher confidence). Gaze cues firing inside the answer's
window (question-timed gaze aversion/darting) add a smaller confirmation on top.
Honest framing: never "LIE".
"""
from __future__ import annotations

WATCH = 0.45
HIGH = 0.65

# Question-timed gaze: these cues firing within the answer's [t0, t1] window align the
# gaze pattern with the question — a small confirm boost, never a driver.
GAZE_CUES = {"visual.gaze_aversion", "visual.gaze_fixation"}
GAZE_BOOST = 0.05


def fuse_turn(verdict, cue_window: dict) -> dict:
    cue_sync = cue_window.get("max_families_synchronous", 0)
    cue_confirms = cue_sync >= 2  # ≥2 independent families lit together in the answer window
    gaze_aligned = bool(GAZE_CUES & set(cue_window.get("cue_ids", [])))

    if not verdict.available:
        # Content layer offline → cue-only honest fallback.
        risk = 0.6 if cue_confirms else 0.2
        return {
            "content_available": False,
            "combined": round(risk, 4),
            "convergence": False,
            "label": "cue-only (content layer offline)",
            "content_risk": 0.0,
            "cue_synchronous_families": cue_sync,
            "gaze_aligned": gaze_aligned,
        }

    content_risk = float(verdict.risk)
    # Content is primary; convergence with cues adds confidence.
    combined = content_risk + (0.15 if cue_confirms and content_risk >= WATCH else 0.0)
    combined += GAZE_BOOST if gaze_aligned and content_risk >= WATCH else 0.0
    combined = max(0.0, min(1.0, combined))
    convergence = cue_confirms and content_risk >= WATCH

    if convergence:
        label = "CONVERGENCE — content + behavioural cues align (high confidence)"
    elif content_risk >= HIGH:
        label = "CONTENT flag — account thin / inconsistent / unverifiable"
    elif content_risk >= WATCH:
        label = "CONTENT watch — some content-pattern risk"
    else:
        label = "CLEAR — content reads truthful"

    return {
        "content_available": True,
        "combined": round(combined, 4),
        "convergence": convergence,
        "label": label,
        "content_risk": round(content_risk, 4),
        "cue_synchronous_families": cue_sync,
        "gaze_aligned": gaze_aligned,
    }
