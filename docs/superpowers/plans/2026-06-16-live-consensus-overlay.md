# Live Consensus Overlay — Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A locally-run, real-time "AI-vision" overlay where the browser captures webcam + runs MediaPipe and rPPG sampling, a local Python engine detects ~5 visual cues + rPPG heart-rate, fuses them by family with a two-gate consensus, and the browser draws a telestrator + consensus panel that only FLAGs under the two-gate rule — all started with one command.

**Architecture:** Approach 2 (locked). The **browser** does webcam capture, MediaPipe Face/Pose landmarking (478 landmarks + 52 blendshapes + head pose), rPPG ROI color sampling, and overlay rendering. It sends **only feature vectors** (no raw video) over a localhost WebSocket. The **Python engine** (reusing `core/`) runs stateful cue detectors → rolling per-person robust-Z baseline → family-grouped Bayesian fusion with the inter-cue independence fix → two-gate convergence → consensus builder → append-only prediction log, and streams a consensus payload back. One FastAPI process serves both the static browser app and the WebSocket, so a single command starts engine + browser.

**Tech Stack:** Python 3.10+ (verified 3.14 locally), FastAPI + uvicorn (one process: static files + `/ws`), numpy (rPPG DSP only), websockets/starlette (bundled with FastAPI). Browser: vanilla JS + `@mediapipe/tasks-vision` (CDN), Canvas 2D overlay. Tooling: ruff + pytest, GitHub Actions CI.

**Locked decisions honored (do not relitigate):** browser capture + Python engine over localhost WS carrying feature vectors only; capture is a swappable source adapter, Stage 1 is webcam only; science-driven cue weights (no learning loop); honest framing with statuses CALIBRATING→CLEAR→WATCH→FLAG, never a binary "LIE"; red pulse only on a two-gate FLAG; rolling per-person robust-Z baseline; rPPG REQUIRED as the second independent family; Audio/Linguistic voters shown "not wired"; WhisperX (not CrisperWhisper) reserved for later transcription stages.

---

## File Structure

**New Python engine package `blitz_overlay/`** (added to `pyproject.toml` packages):

| File | Responsibility |
|---|---|
| `blitz_overlay/__init__.py` | Package marker + version export |
| `blitz_overlay/__main__.py` | `python -m blitz_overlay` entry: start uvicorn, print/open URL |
| `blitz_overlay/config.py` | Typed config loaded from env/.env (host, port, gate, baseline seconds, weight-set version) |
| `blitz_overlay/schemas.py` | The versioned WebSocket contract: `FeatureFrame`, `Consensus`, `FamilyVote`, `ActiveCue`, `Region`, `SCHEMA_VERSION` |
| `blitz_overlay/weights.py` | Science-driven, citation-annotated cue config + `WEIGHT_SET_VERSION` |
| `blitz_overlay/rppg.py` | rPPG DSP (CHROM signal + band-limited dominant-frequency → BPM) — numpy, unit-testable |
| `blitz_overlay/cues/__init__.py` | Cue detector exports |
| `blitz_overlay/cues/base.py` | `CueDetector` abstract interface (per-session, stateful): `update(frame, baseline) -> CueEvent \| None` |
| `blitz_overlay/cues/visual.py` | 5 visual detectors: blink rate, gaze aversion, brow flash, lip press, jaw tension |
| `blitz_overlay/cues/physio.py` | `RppgHeartRate` detector (buffers ROI samples, calls `rppg.py`, emits HR cue) |
| `blitz_overlay/consensus.py` | `ConsensusBuilder`: status machine + per-family votes + freshness/wired + risk meter |
| `blitz_overlay/logger.py` | Append-only JSONL prediction logger (no raw biometric) |
| `blitz_overlay/pipeline.py` | `OverlaySession`: wires detectors + rolling baseline + family fusion + consensus + logger per WS connection |
| `blitz_overlay/server.py` | FastAPI app: static mount at `/`, `/ws` endpoint driving an `OverlaySession` |

**Extended existing `core/`:**

| File | Change |
|---|---|
| `core/calibration/rolling.py` | NEW — `RollingBaseline` (mode="rolling"): time-windowed robust-Z, CALIBRATING gate |
| `core/calibration/__init__.py` | Export `RollingBaseline` alongside `PersonalBaseline` |
| `core/fusion/bayesian_fusion.py` | ADD `FAMILY_OF`, `fuse_by_family()`, `two_gate()` — family-grouped log-odds with within-family decorrelation (READINESS #11) + two-gate (#6) |

**New browser app `apps/overlay-web/`** (served as static files):

| File | Responsibility |
|---|---|
| `apps/overlay-web/index.html` | Page shell: video + canvas + consensus panel markup; loads MediaPipe from CDN |
| `apps/overlay-web/css/overlay.css` | Telestrator + panel + risk meter + red-pulse styles |
| `apps/overlay-web/js/schema.js` | Mirror of `SCHEMA_VERSION` + blendshape/region names |
| `apps/overlay-web/js/regions.js` | Region → FaceMesh landmark-index map (eyes, brow, mouth, jaw, forehead) |
| `apps/overlay-web/js/capture.js` | `WebcamSource` source adapter (swappable interface) |
| `apps/overlay-web/js/mediapipe-extractor.js` | FaceLandmarker (+blendshapes +head pose) + PoseLandmarker → feature frame |
| `apps/overlay-web/js/rppg-sampler.js` | Forehead/cheek ROI mean-RGB sampling from landmarks |
| `apps/overlay-web/js/ws-client.js` | Send feature frames, receive consensus, auto-reconnect |
| `apps/overlay-web/js/overlay-renderer.js` | Telestrator circles, collapsed/expanded panel, risk meter, earned red pulse, degradation messages |
| `apps/overlay-web/js/main.js` | Bootstrap + per-frame loop wiring all of the above |

**Tooling / tests:**

| File | Responsibility |
|---|---|
| `requirements.txt` | Pinned runtime + dev deps (mirror of pyproject) |
| `.env.example` | Documented env vars; real `.env` gitignored |
| `.github/workflows/ci.yml` | ruff + pytest on push/PR |
| `tests/overlay/` | One test file per module + fixtures + deterministic replay |

---

## Conventions for every task
- TDD: write the failing test, run it red, implement minimal code, run it green, commit.
- Run tests with `python -m pytest` (works on Python 3.14; `pytest`/`ruff` installed in Task 1).
- Lint with `python -m ruff check .` before each commit once Task 1 lands.
- Commit messages end with the Co-Authored-By trailer already configured for this repo.

---

### Task 1: Tooling — deps, ruff, pytest, .env, CI (READINESS #1, #16, #21)

**Files:**
- Modify: `pyproject.toml`
- Create: `requirements.txt`
- Create: `.env.example`
- Modify: `.gitignore`
- Create: `.github/workflows/ci.yml`
- Create: `tests/overlay/__init__.py`
- Create: `tests/overlay/test_tooling.py`

- [ ] **Step 1: Write the failing test**

Create `tests/overlay/__init__.py` (empty) and `tests/overlay/test_tooling.py`:

```python
"""Tooling guards: deps declared, tools importable, env handling present."""
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_pyproject_declares_overlay_packages_and_deps():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    packages = data["tool"]["setuptools"]["packages"]
    assert "blitz_overlay" in packages
    assert "blitz_overlay.cues" in packages
    deps = data["project"]["dependencies"]
    joined = " ".join(deps)
    assert "fastapi" in joined
    assert "uvicorn" in joined
    assert "numpy" in joined


def test_ruff_and_pytest_importable():
    import pytest  # noqa: F401
    import ruff  # noqa: F401 -- ruff ships an importable shim when installed


def test_env_example_and_gitignore():
    assert (ROOT / ".env.example").exists()
    gitignore = (ROOT / ".gitignore").read_text()
    assert ".env" in gitignore
    assert "logs/" in gitignore


def test_ci_workflow_runs_ruff_and_pytest():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "ruff" in ci
    assert "pytest" in ci
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/overlay/test_tooling.py -v`
Expected: collection error or FAIL — `ModuleNotFoundError: No module named 'pytest'` / `ruff`, then assertion failures.

- [ ] **Step 3: Install dev tools**

Run: `python -m pip install pytest ruff`
Expected: both install successfully.

- [ ] **Step 4: Update `pyproject.toml`**

Replace the file contents with:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "blitz-engine"
version = "0.1.0"
description = "Research-first behavioral signal analyzer MVP"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.115",
    "uvicorn>=0.30",
    "websockets>=12",
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.6"]

[project.scripts]
blitz = "blitz_engine.cli:main"
blitz-overlay = "blitz_overlay.__main__:main"

[tool.setuptools]
packages = [
    "blitz_engine",
    "core", "core.calibration", "core.fusion", "core.schemas",
    "modalities", "modalities.audio", "modalities.linguistic",
    "blitz_overlay", "blitz_overlay.cues",
]

[tool.ruff]
line-length = 110
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "W", "UP", "B"]
ignore = ["E501"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 5: Create `requirements.txt`**

```text
# Runtime
fastapi>=0.115
uvicorn>=0.30
websockets>=12
numpy>=1.26
# Dev
pytest>=8
ruff>=0.6
```

- [ ] **Step 6: Create `.env.example`**

```text
# Blitz Overlay — local config. Copy to .env and edit. .env is gitignored.
# No external API keys are required for Stage 1 (all fusion is local math).
BLITZ_OVERLAY_HOST=127.0.0.1
BLITZ_OVERLAY_PORT=8000
# Two-gate combined-risk threshold (Gate 1). Spec default 0.65.
BLITZ_OVERLAY_GATE=0.65
# Rolling baseline fill time before flags are permitted (seconds). Science default 90.
BLITZ_OVERLAY_BASELINE_SECONDS=90
# Auto-open the browser on start.
BLITZ_OVERLAY_OPEN_BROWSER=1
```

- [ ] **Step 7: Update `.gitignore`** (append)

```text

# Local config & prediction logs (never commit biometric-derived data)
.env
logs/
```

- [ ] **Step 8: Create `.github/workflows/ci.yml`**

```yaml
name: CI
on:
  push:
  pull_request:
jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install
        run: |
          python -m pip install --upgrade pip
          python -m pip install -e ".[dev]"
      - name: Ruff
        run: python -m ruff check .
      - name: Pytest
        run: python -m pytest -q
```

- [ ] **Step 9: Run test to verify it passes**

Run: `python -m pytest tests/overlay/test_tooling.py -v`
Expected: PASS (4 passed). If the `import ruff` line fails on this ruff version, change that test to assert the CLI instead:
```python
import subprocess, sys
def test_ruff_and_pytest_importable():
    import pytest  # noqa: F401
    assert subprocess.run([sys.executable, "-m", "ruff", "--version"]).returncode == 0
```

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml requirements.txt .env.example .gitignore .github/workflows/ci.yml tests/overlay/
git commit -m "chore: add overlay deps, ruff+pytest config, .env handling, CI"
```

---

### Task 2: Shared WebSocket schema + region map (spec §4 shared/, §5)

**Files:**
- Create: `blitz_overlay/__init__.py`
- Create: `blitz_overlay/schemas.py`
- Test: `tests/overlay/test_schemas.py`

- [ ] **Step 1: Write the failing test**

```python
from blitz_overlay.schemas import (
    SCHEMA_VERSION, Region, FeatureFrame, FamilyVote, ActiveCue, Consensus,
)


def test_feature_frame_roundtrip():
    raw = {
        "schema_version": SCHEMA_VERSION,
        "ts": 1000,
        "face_present": True,
        "confidence": 0.9,
        "blendshapes": {"eyeBlinkLeft": 0.1, "browInnerUp": 0.2},
        "head_pose": {"yaw": 1.0, "pitch": -2.0, "roll": 0.5},
        "geometry": {"jaw_width_ratio": 0.83, "gaze_x": 0.1, "gaze_y": -0.05},
        "rppg": {"forehead_rgb": [180.0, 120.0, 110.0], "cheek_rgb": [190.0, 130.0, 120.0]},
    }
    frame = FeatureFrame.from_dict(raw)
    assert frame.ts == 1000
    assert frame.face_present is True
    assert frame.blendshapes["browInnerUp"] == 0.2
    assert frame.geometry["jaw_width_ratio"] == 0.83
    assert frame.rppg["forehead_rgb"][0] == 180.0


def test_feature_frame_tolerates_missing_optionals():
    frame = FeatureFrame.from_dict({"ts": 5, "face_present": False})
    assert frame.face_present is False
    assert frame.blendshapes == {}
    assert frame.rppg is None


def test_region_enum_values():
    assert Region.EYES.value == "eyes"
    assert {r.value for r in Region} >= {"eyes", "brow", "mouth", "jaw", "forehead"}


def test_consensus_to_dict_is_json_serializable():
    import json
    consensus = Consensus(
        schema_version=SCHEMA_VERSION,
        ts=2000,
        status="WATCH",
        risk=0.42,
        flag=False,
        n_agree=1,
        n_required=2,
        families=[
            FamilyVote(name="visual", wired=True, fresh=True, vote=False, contribution=0.3),
            FamilyVote(name="physio", wired=True, fresh=False, vote=False, contribution=0.0),
            FamilyVote(name="audio", wired=False, fresh=False, vote=False, contribution=0.0),
        ],
        active_cues=[ActiveCue(cue_id="visual.gaze_aversion", region="eyes", z=2.1, confidence=0.8)],
    )
    payload = json.dumps(consensus.to_dict())
    assert '"status": "WATCH"' in payload
    assert '"region": "eyes"' in payload
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/overlay/test_schemas.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'blitz_overlay'`.

- [ ] **Step 3: Create `blitz_overlay/__init__.py`**

```python
"""Blitz Engine — Live Consensus Overlay (Stage 1: webcam-only walking skeleton)."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Create `blitz_overlay/schemas.py`**

```python
"""Versioned WebSocket contract between the browser and the Python engine.

Only derived feature vectors cross the wire — never raw video (spec §3, READINESS #15).
The browser mirrors SCHEMA_VERSION in apps/overlay-web/js/schema.js.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

SCHEMA_VERSION = "1.0"


class Region(str, Enum):
    """Telestrator anchor regions a cue can light up (spec §5)."""

    EYES = "eyes"
    BROW = "brow"
    MOUTH = "mouth"
    JAW = "jaw"
    FOREHEAD = "forehead"
    HEAD = "head"
    BODY = "body"


@dataclass
class FeatureFrame:
    """One browser → engine frame. Tiny: blendshapes + pose + a few derived scalars."""

    ts: int                                   # client timestamp (ms)
    face_present: bool = False
    confidence: float = 0.0                   # landmark confidence [0,1]
    blendshapes: dict = field(default_factory=dict)   # name -> coefficient [0,1]
    head_pose: dict = field(default_factory=dict)     # yaw/pitch/roll (degrees)
    geometry: dict = field(default_factory=dict)      # jaw_width_ratio, gaze_x, gaze_y, ear_*
    rppg: dict | None = None                  # {"forehead_rgb":[r,g,b], "cheek_rgb":[r,g,b]} or None
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureFrame":
        return cls(
            ts=int(d.get("ts", 0)),
            face_present=bool(d.get("face_present", False)),
            confidence=float(d.get("confidence", 0.0)),
            blendshapes=dict(d.get("blendshapes") or {}),
            head_pose=dict(d.get("head_pose") or {}),
            geometry=dict(d.get("geometry") or {}),
            rppg=(dict(d["rppg"]) if d.get("rppg") else None),
            schema_version=str(d.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass
class FamilyVote:
    """One consensus voter (a modality family)."""

    name: str
    wired: bool          # is this family implemented in this stage?
    fresh: bool          # has it produced a recent, usable signal?
    vote: bool           # does it currently vote "flag"?
    contribution: float  # its log-odds contribution to combined risk

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "wired": self.wired,
            "fresh": self.fresh,
            "vote": self.vote,
            "contribution": round(self.contribution, 4),
        }


@dataclass
class ActiveCue:
    """A currently-firing cue, for telestrator anchoring."""

    cue_id: str
    region: str
    z: float
    confidence: float

    def to_dict(self) -> dict:
        return {
            "cue_id": self.cue_id,
            "region": self.region,
            "z": round(self.z, 3),
            "confidence": round(self.confidence, 3),
        }


@dataclass
class Consensus:
    """Engine → browser payload driving the overlay."""

    schema_version: str
    ts: int
    status: str            # CALIBRATING | CLEAR | WATCH | FLAG
    risk: float            # [0,1] combined posterior
    flag: bool             # earned two-gate FLAG (drives the red pulse)
    n_agree: int           # independent families currently voting flag
    n_required: int        # families required to agree (2)
    families: list[FamilyVote] = field(default_factory=list)
    active_cues: list[ActiveCue] = field(default_factory=list)
    message: str = ""      # degradation/status message (spec §8)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "ts": self.ts,
            "status": self.status,
            "risk": round(self.risk, 4),
            "flag": self.flag,
            "n_agree": self.n_agree,
            "n_required": self.n_required,
            "families": [f.to_dict() for f in self.families],
            "active_cues": [c.to_dict() for c in self.active_cues],
            "message": self.message,
        }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/overlay/test_schemas.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add blitz_overlay/__init__.py blitz_overlay/schemas.py tests/overlay/test_schemas.py
git commit -m "feat(overlay): versioned feature-frame + consensus WS schema"
```

---

### Task 3: Science-driven cue weights config (READINESS #8, #18)

**Files:**
- Create: `blitz_overlay/weights.py`
- Test: `tests/overlay/test_weights.py`

- [ ] **Step 1: Write the failing test**

```python
from blitz_overlay.weights import CUE_WEIGHTS, WEIGHT_SET_VERSION, weight_for

WIRED = {
    "visual.blink_rate", "visual.gaze_aversion", "visual.brow_flash",
    "visual.lip_press", "visual.jaw_tension", "physio.heart_rate",
}


def test_all_wired_cues_have_annotated_weights():
    for cue_id in WIRED:
        spec = CUE_WEIGHTS[cue_id]
        assert spec["effect_size_d"] > 0
        assert spec["reliability_tier"] in (1, 2, 3, 4)
        assert spec["family"] in ("visual", "physio", "audio", "linguistic")
        assert spec["region"]
        assert spec["citation"], f"{cue_id} needs a citation (science-driven, not learned)"


def test_weight_set_is_versioned():
    assert isinstance(WEIGHT_SET_VERSION, str) and WEIGHT_SET_VERSION


def test_weight_for_helper_returns_spec():
    assert weight_for("visual.gaze_aversion")["effect_size_d"] == CUE_WEIGHTS["visual.gaze_aversion"]["effect_size_d"]


def test_gaze_is_strongest_visual_cue():
    # cue 58 (gaze aversion duration) d~0.6-0.8 should outrank brow/lip/jaw proxies
    assert CUE_WEIGHTS["visual.gaze_aversion"]["effect_size_d"] >= CUE_WEIGHTS["visual.brow_flash"]["effect_size_d"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/overlay/test_weights.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'blitz_overlay.weights'`.

- [ ] **Step 3: Create `blitz_overlay/weights.py`**

```python
"""Science-driven cue weights — fixed from the literature, NEVER learned (READINESS #8).

Each weight is a published effect size (Cohen's d / Hedges' g) with a traceable citation.
Changing this set requires bumping WEIGHT_SET_VERSION; the version is stamped into every
prediction-log line for auditability (READINESS #18). Corpora validate accuracy only —
they must not move these weights.
"""
from __future__ import annotations

WEIGHT_SET_VERSION = "stage1-2026-06-16"

# cue_id -> {effect_size_d, reliability_tier(1 strong..4 anchor), family, region, citation}
CUE_WEIGHTS: dict[str, dict] = {
    "visual.gaze_aversion": {
        "effect_size_d": 0.70, "reliability_tier": 2, "family": "visual", "region": "eyes",
        "citation": "Catalog cue 58 — sustained gaze aversion duration, d~0.6-0.8 high-stakes (CUE_CATALOG.md).",
    },
    "visual.blink_rate": {
        "effect_size_d": 0.40, "reliability_tier": 2, "family": "visual", "region": "eyes",
        "citation": "Catalog cue 60 — blink suppression→rebound; deviation-from-baseline, pattern>static rate.",
    },
    "visual.brow_flash": {
        "effect_size_d": 0.30, "reliability_tier": 3, "family": "visual", "region": "brow",
        "citation": "Catalog cue 9 — AU1/2/4 brow movement; weak-moderate single-cue diagnosticity.",
    },
    "visual.lip_press": {
        "effect_size_d": 0.30, "reliability_tier": 3, "family": "visual", "region": "mouth",
        "citation": "Catalog cue 3 — lip compression (AU23/24), withholding; weak-moderate.",
    },
    "visual.jaw_tension": {
        "effect_size_d": 0.28, "reliability_tier": 3, "family": "visual", "region": "jaw",
        "citation": "Catalog cue 8 — jaw tension (AU28) via MediaPipe landmark-distance proxy; resolves Blocker 2.",
    },
    "physio.heart_rate": {
        "effect_size_d": 0.50, "reliability_tier": 2, "family": "physio", "region": "forehead",
        "citation": "Catalog cue 38 — rPPG heart-rate elevation, autonomic arousal proxy.",
    },
}


def weight_for(cue_id: str) -> dict:
    return CUE_WEIGHTS[cue_id]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/overlay/test_weights.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add blitz_overlay/weights.py tests/overlay/test_weights.py
git commit -m "feat(overlay): citation-annotated science-driven cue weights"
```

---

### Task 4: Rolling baseline calibration mode (READINESS #7)

**Files:**
- Create: `core/calibration/rolling.py`
- Modify: `core/calibration/__init__.py`
- Test: `tests/overlay/test_rolling_baseline.py`

- [ ] **Step 1: Write the failing test**

```python
from core.calibration import RollingBaseline


def test_calibrating_until_window_filled_then_ready():
    rb = RollingBaseline(baseline_seconds=10)
    rb.update({"visual.blink_rate": 12.0}, ts_ms=0)
    assert rb.mode == "rolling"
    assert rb.is_calibrating is True
    # advance time past the fill window with steady values
    for t in range(1, 12):
        rb.update({"visual.blink_rate": 12.0 + (t % 3) * 0.1}, ts_ms=t * 1000)
    assert rb.is_calibrating is False
    assert rb.ready is True


def test_normalize_returns_zero_while_calibrating():
    rb = RollingBaseline(baseline_seconds=10)
    rb.update({"visual.blink_rate": 12.0}, ts_ms=0)
    assert rb.normalize("visual.blink_rate", 40.0) == 0.0  # no flags during calibration


def test_robust_z_after_calibration_flags_deviation():
    rb = RollingBaseline(baseline_seconds=5, window_seconds=60)
    for t in range(0, 7):
        rb.update({"visual.blink_rate": 12.0}, ts_ms=t * 1000)  # steady ~12
    # a steady baseline has MAD 0 -> guard returns 0; add small jitter then test a spike
    rb2 = RollingBaseline(baseline_seconds=5, window_seconds=60)
    for t, v in enumerate([11.0, 12.0, 13.0, 12.0, 11.0, 13.0, 12.0]):
        rb2.update({"visual.blink_rate": v}, ts_ms=t * 1000)
    z = rb2.normalize("visual.blink_rate", 30.0)
    assert z > 3.0  # large positive deviation


def test_unknown_cue_normalizes_to_zero():
    rb = RollingBaseline(baseline_seconds=1)
    rb.update({"visual.blink_rate": 12.0}, ts_ms=0)
    rb.update({"visual.blink_rate": 12.0}, ts_ms=2000)
    assert rb.normalize("visual.never_seen", 5.0) == 0.0


def test_window_evicts_old_observations():
    rb = RollingBaseline(baseline_seconds=1, window_seconds=5)
    for t in range(0, 10):
        rb.update({"c": float(t)}, ts_ms=t * 1000)
    # only observations within the last 5s should remain
    assert rb.observation_count("c") <= 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/overlay/test_rolling_baseline.py -v`
Expected: FAIL — `ImportError: cannot import name 'RollingBaseline'`.

- [ ] **Step 3: Create `core/calibration/rolling.py`**

```python
"""Rolling per-person baseline (calibration mode="rolling") — READINESS #7.

Builds an in-session baseline from a sliding time window of recent feature values
instead of a fixed enrollment clip, then scores incoming values as robust-Z (median/MAD)
deviations. Used by the live overlay where there is no enrollment step. No flags are
permitted until the fill window (default 90s, spec §8) has elapsed (status CALIBRATING).
"""
from __future__ import annotations

from collections import defaultdict, deque

from core.calibration.baseline import compute_robust_z

DEFAULT_BASELINE_SECONDS = 90
DEFAULT_WINDOW_SECONDS = 180


class RollingBaseline:
    mode = "rolling"

    def __init__(self, baseline_seconds: int = DEFAULT_BASELINE_SECONDS,
                 window_seconds: int = DEFAULT_WINDOW_SECONDS):
        self.baseline_seconds = baseline_seconds
        self.window_seconds = window_seconds
        self._values: dict[str, deque] = defaultdict(deque)  # cue_id -> deque[(ts_ms, value)]
        self._first_ts: int | None = None
        self._last_ts: int = 0

    def update(self, features: dict[str, float], ts_ms: int) -> None:
        if self._first_ts is None:
            self._first_ts = ts_ms
        self._last_ts = ts_ms
        cutoff = ts_ms - self.window_seconds * 1000
        for cue_id, value in features.items():
            dq = self._values[cue_id]
            dq.append((ts_ms, float(value)))
            while dq and dq[0][0] < cutoff:
                dq.popleft()

    @property
    def elapsed_seconds(self) -> float:
        if self._first_ts is None:
            return 0.0
        return (self._last_ts - self._first_ts) / 1000.0

    @property
    def is_calibrating(self) -> bool:
        return self.elapsed_seconds < self.baseline_seconds

    @property
    def ready(self) -> bool:
        return not self.is_calibrating

    def observation_count(self, cue_id: str) -> int:
        return len(self._values.get(cue_id, ()))

    def normalize(self, cue_id: str, raw_value: float) -> float:
        """Robust-Z vs the rolling window. Returns 0.0 while calibrating or if underpowered."""
        if self.is_calibrating:
            return 0.0
        dq = self._values.get(cue_id)
        if not dq or len(dq) < 5:
            return 0.0
        return compute_robust_z(raw_value, [v for _, v in dq])
```

- [ ] **Step 4: Update `core/calibration/__init__.py`**

Read the current file first; it likely only has a docstring. Set its contents to:

```python
"""Calibration: enrollment and rolling baseline modes."""

from core.calibration.baseline import PersonalBaseline, compute_robust_z
from core.calibration.rolling import RollingBaseline

__all__ = ["PersonalBaseline", "RollingBaseline", "compute_robust_z"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/overlay/test_rolling_baseline.py -v`
Expected: PASS (5 passed).

- [ ] **Step 6: Commit**

```bash
git add core/calibration/rolling.py core/calibration/__init__.py tests/overlay/test_rolling_baseline.py
git commit -m "feat(core): rolling robust-Z baseline calibration mode (READINESS #7)"
```

---

### Task 5: Cue detector base interface (READINESS #2 applied to detectors)

**Files:**
- Create: `blitz_overlay/cues/__init__.py`
- Create: `blitz_overlay/cues/base.py`
- Test: `tests/overlay/test_cue_base.py`

- [ ] **Step 1: Write the failing test**

```python
from blitz_overlay.cues.base import CueDetector
from blitz_overlay.schemas import FeatureFrame
from core.calibration import RollingBaseline


class _Dummy(CueDetector):
    cue_id = "visual.blink_rate"

    def measure(self, frame):
        return frame.blendshapes.get("eyeBlinkLeft", None)


def test_detector_exposes_metadata_from_weights():
    d = _Dummy()
    assert d.family == "visual"
    assert d.region == "eyes"
    assert d.effect_size_d > 0
    assert d.reliability_tier in (1, 2, 3, 4)


def test_update_returns_none_when_measure_is_none():
    d = _Dummy()
    rb = RollingBaseline(baseline_seconds=0)
    rb.update({"visual.blink_rate": 0.1}, ts_ms=0)
    rb.update({"visual.blink_rate": 0.1}, ts_ms=1000)
    frame = FeatureFrame.from_dict({"ts": 2000, "face_present": True, "confidence": 0.9})
    assert d.update(frame, rb) is None  # eyeBlinkLeft absent -> measure None


def test_update_emits_cue_event_on_deviation():
    d = _Dummy()
    rb = RollingBaseline(baseline_seconds=0, window_seconds=60)
    for t, v in enumerate([0.1, 0.12, 0.09, 0.11, 0.1, 0.13]):
        rb.update({"visual.blink_rate": v}, ts_ms=t * 1000)
    frame = FeatureFrame.from_dict({
        "ts": 7000, "face_present": True, "confidence": 0.9,
        "blendshapes": {"eyeBlinkLeft": 0.9},
    })
    event = d.update(frame, rb)
    assert event is not None
    assert event.cue_id == "visual.blink_rate"
    assert event.modality.value == "visual"
    assert event.z_score > 2.0
    assert event.quality > 0


def test_low_confidence_widens_uncertainty_via_quality():
    d = _Dummy()
    rb = RollingBaseline(baseline_seconds=0, window_seconds=60)
    for t, v in enumerate([0.1, 0.12, 0.09, 0.11, 0.1, 0.13]):
        rb.update({"visual.blink_rate": v}, ts_ms=t * 1000)
    low = FeatureFrame.from_dict({"ts": 7000, "face_present": True, "confidence": 0.2,
                                  "blendshapes": {"eyeBlinkLeft": 0.9}})
    high = FeatureFrame.from_dict({"ts": 7000, "face_present": True, "confidence": 0.95,
                                   "blendshapes": {"eyeBlinkLeft": 0.9}})
    assert d.update(low, rb).quality < d.update(high, rb).quality
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/overlay/test_cue_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'blitz_overlay.cues.base'`.

- [ ] **Step 3: Create `blitz_overlay/cues/__init__.py`**

```python
"""Cue detectors: one stateful detector per cue, feature frame -> CueEvent | None."""
```

- [ ] **Step 4: Create `blitz_overlay/cues/base.py`**

```python
"""Abstract cue-detector interface (the per-cue plugin contract for the live overlay).

Each detector is stateful and lives for one session. It reads a FeatureFrame, derives a
single scalar measurement, normalizes it against the rolling baseline, and emits a
CueEvent when the deviation passes the cue's direction/threshold. Metadata (family,
region, effect size, tier) comes from the science-driven weights config.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from blitz_overlay.schemas import FeatureFrame
from blitz_overlay.weights import weight_for
from core.calibration import RollingBaseline
from core.schemas.cue_event import CueEvent, Modality, Phase

_MODALITY = {
    "visual": Modality.VISUAL,
    "physio": Modality.PHYSIOLOGICAL,
    "audio": Modality.AUDIO,
    "linguistic": Modality.LINGUISTIC,
}

# A cue fires when |robust-Z| meets this threshold (direction-aware below).
DEFAULT_Z_THRESHOLD = 2.0


class CueDetector(ABC):
    cue_id: str = ""
    z_threshold: float = DEFAULT_Z_THRESHOLD
    direction: int = 1  # +1: high values suspicious; -1: low values suspicious

    def __init__(self) -> None:
        spec = weight_for(self.cue_id)
        self.family: str = spec["family"]
        self.region: str = spec["region"]
        self.effect_size_d: float = spec["effect_size_d"]
        self.reliability_tier: int = spec["reliability_tier"]
        self.modality: Modality = _MODALITY[self.family]

    @abstractmethod
    def measure(self, frame: FeatureFrame) -> float | None:
        """Return this cue's scalar measurement for the frame, or None if unavailable."""

    def quality(self, frame: FeatureFrame) -> float:
        """Extraction confidence — scales with landmark confidence (spec §8 low-light path)."""
        return max(0.0, min(1.0, frame.confidence))

    def update(self, frame: FeatureFrame, baseline: RollingBaseline) -> CueEvent | None:
        value = self.measure(frame)
        if value is None:
            return None
        baseline.update({self.cue_id: value}, ts_ms=frame.ts)
        z = baseline.normalize(self.cue_id, value)
        directed_z = z * self.direction
        if directed_z < self.z_threshold:
            return None
        return CueEvent(
            cue_id=self.cue_id,
            modality=self.modality,
            timestamp_ms=frame.ts,
            phase=Phase.RESPONSE,
            raw_value=float(value),
            z_score=directed_z,
            llr=0.0,
            quality=self.quality(frame),
            question_id="live",
            effect_size_d=self.effect_size_d,
            reliability_tier=self.reliability_tier,
        )
```

> Note: `update()` calls `baseline.update(...)` so the rolling window keeps filling from live values; the engine's `OverlaySession` (Task 11) updates the baseline once per frame across all cues, but a detector updating its own key is idempotent within a frame because deques dedupe by append, not by ts. To avoid double-append, the `OverlaySession` will own baseline updates and detectors will only call `baseline.normalize`. **Adjust now:** remove the `baseline.update(...)` line from `update()` (the session owns updates). Keep only `z = baseline.normalize(self.cue_id, value)`.

- [ ] **Step 5: Apply the note — final `update()` body**

Replace the `update` method body so it does NOT mutate the baseline:

```python
    def update(self, frame: FeatureFrame, baseline: RollingBaseline) -> CueEvent | None:
        value = self.measure(frame)
        if value is None:
            return None
        z = baseline.normalize(self.cue_id, value)
        directed_z = z * self.direction
        if directed_z < self.z_threshold:
            return None
        return CueEvent(
            cue_id=self.cue_id,
            modality=self.modality,
            timestamp_ms=frame.ts,
            phase=Phase.RESPONSE,
            raw_value=float(value),
            z_score=directed_z,
            llr=0.0,
            quality=self.quality(frame),
            question_id="live",
            effect_size_d=self.effect_size_d,
            reliability_tier=self.reliability_tier,
        )
```

The Task 5 test seeds the baseline directly, so `measure` returning a value + `normalize` is enough.

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/overlay/test_cue_base.py -v`
Expected: PASS (4 passed).

- [ ] **Step 7: Commit**

```bash
git add blitz_overlay/cues/__init__.py blitz_overlay/cues/base.py tests/overlay/test_cue_base.py
git commit -m "feat(overlay): abstract stateful CueDetector interface"
```

---

### Task 6: Visual cue detectors (5 cues) (spec §5, §11)

**Files:**
- Create: `blitz_overlay/cues/visual.py`
- Test: `tests/overlay/test_visual_cues.py`

All five share the base; blink and gaze are stateful (windowed/sustained), brow/lip/jaw are per-frame scalars.

- [ ] **Step 1: Write the failing test**

```python
from blitz_overlay.cues.visual import (
    BlinkRate, GazeAversion, BrowFlash, LipPress, JawTension, VISUAL_DETECTORS,
)
from blitz_overlay.schemas import FeatureFrame
from core.calibration import RollingBaseline


def _frame(ts, **bs_geo):
    bs = {k: v for k, v in bs_geo.items() if not k.startswith("g_")}
    geo = {k[2:]: v for k, v in bs_geo.items() if k.startswith("g_")}
    return FeatureFrame.from_dict({
        "ts": ts, "face_present": True, "confidence": 0.9,
        "blendshapes": bs, "geometry": geo,
        "head_pose": {"yaw": geo.get("yaw", 0.0), "pitch": geo.get("pitch", 0.0)},
    })


def test_registry_has_five_visual_detectors():
    assert len(VISUAL_DETECTORS) == 5
    ids = {d().cue_id for d in VISUAL_DETECTORS}
    assert ids == {
        "visual.blink_rate", "visual.gaze_aversion", "visual.brow_flash",
        "visual.lip_press", "visual.jaw_tension",
    }


def test_brow_flash_measure_combines_inner_up_and_brow_down():
    d = BrowFlash()
    m = d.measure(_frame(0, browInnerUp=0.6, browDownLeft=0.2, browDownRight=0.4))
    assert m == 0.6  # max of browInnerUp and mean browDown


def test_lip_press_measure_combines_press_and_pucker():
    d = LipPress()
    m = d.measure(_frame(0, mouthPressLeft=0.3, mouthPressRight=0.5, mouthPucker=0.2))
    assert abs(m - 0.4) < 1e-9  # mean of press L/R = 0.4 dominates pucker 0.2


def test_jaw_tension_reads_geometry_ratio():
    d = JawTension()
    assert d.measure(_frame(0, g_jaw_width_ratio=0.81)) == 0.81
    assert d.measure(_frame(0)) is None  # missing geometry -> unavailable


def test_blink_rate_counts_blinks_per_minute():
    d = BlinkRate()
    # simulate 3 blinks over 6 seconds = 30 blinks/min
    ts = 0
    rate = None
    for i in range(3):
        rate = d.measure(_frame(ts, eyeBlinkLeft=0.05, eyeBlinkRight=0.05)); ts += 1000  # open
        d.measure(_frame(ts, eyeBlinkLeft=0.8, eyeBlinkRight=0.8)); ts += 200            # closed (blink)
        rate = d.measure(_frame(ts, eyeBlinkLeft=0.05, eyeBlinkRight=0.05)); ts += 800   # open
    assert rate is not None and rate > 0


def test_gaze_aversion_requires_sustained_offset():
    d = GazeAversion()
    # momentary look-away should not count; sustained > 2s should
    assert d.measure(_frame(0, g_gaze_x=0.5, g_gaze_y=0.0)) == 0.0  # not yet sustained
    m = None
    for t in range(0, 3000, 200):
        m = d.measure(_frame(t, g_gaze_x=0.6, g_gaze_y=0.1))
    assert m is not None and m > 2.0  # sustained aversion duration in seconds


def test_visual_cue_emits_event_after_calibration():
    d = GazeAversion()
    rb = RollingBaseline(baseline_seconds=0, window_seconds=120)
    # baseline: centered gaze -> low sustained-aversion values
    for t in range(0, 6000, 200):
        v = d.measure(_frame(t, g_gaze_x=0.02, g_gaze_y=0.0))
        rb.update({"visual.gaze_aversion": v}, ts_ms=t)
    # now sustain a strong aversion
    ts = 6000
    event = None
    for _ in range(20):
        v = d.measure(_frame(ts, g_gaze_x=0.7, g_gaze_y=0.2))
        rb.update({"visual.gaze_aversion": v}, ts_ms=ts)
        event = d.update(_frame(ts, g_gaze_x=0.7, g_gaze_y=0.2), rb)
        ts += 200
    assert event is not None and event.region == "eyes"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/overlay/test_visual_cues.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'blitz_overlay.cues.visual'`.

- [ ] **Step 3: Create `blitz_overlay/cues/visual.py`**

```python
"""Five visual cue detectors mapped to MediaPipe blendshapes / landmark geometry (spec §5)."""
from __future__ import annotations

from collections import deque

from blitz_overlay.cues.base import CueDetector
from blitz_overlay.schemas import FeatureFrame

BLINK_CLOSED_THRESHOLD = 0.5     # eyeBlink coefficient above this = eye closed
BLINK_WINDOW_MS = 30_000         # rolling window for blink-rate estimate
GAZE_OFFSET_THRESHOLD = 0.30     # combined gaze magnitude considered "averted"


class BlinkRate(CueDetector):
    """Blinks/min from eyeBlink blendshapes vs baseline (catalog cue 1/60)."""

    cue_id = "visual.blink_rate"
    direction = 1  # both directions matter, but elevated rate is the flag signal here

    def __init__(self) -> None:
        super().__init__()
        self._closed = False
        self._blink_ts: deque[int] = deque()

    def measure(self, frame: FeatureFrame) -> float | None:
        bs = frame.blendshapes
        if "eyeBlinkLeft" not in bs and "eyeBlinkRight" not in bs:
            return None
        closed_amt = max(bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0))
        now = frame.ts
        if closed_amt >= BLINK_CLOSED_THRESHOLD and not self._closed:
            self._closed = True
            self._blink_ts.append(now)          # rising edge = one blink
        elif closed_amt < BLINK_CLOSED_THRESHOLD:
            self._closed = False
        while self._blink_ts and self._blink_ts[0] < now - BLINK_WINDOW_MS:
            self._blink_ts.popleft()
        span_ms = max(1000, now - (self._blink_ts[0] if self._blink_ts else now))
        return len(self._blink_ts) * 60_000.0 / span_ms


class GazeAversion(CueDetector):
    """Sustained gaze-aversion *duration* in seconds (catalog cue 58)."""

    cue_id = "visual.gaze_aversion"
    direction = 1

    def __init__(self) -> None:
        super().__init__()
        self._averted_since: int | None = None
        self._last_ts: int = 0

    def measure(self, frame: FeatureFrame) -> float | None:
        g = frame.geometry
        gx, gy = g.get("gaze_x"), g.get("gaze_y")
        if gx is None and gy is None:
            return None
        magnitude = (float(gx or 0.0) ** 2 + float(gy or 0.0) ** 2) ** 0.5
        now = frame.ts
        if magnitude >= GAZE_OFFSET_THRESHOLD:
            if self._averted_since is None:
                self._averted_since = now
            duration_s = (now - self._averted_since) / 1000.0
        else:
            self._averted_since = None
            duration_s = 0.0
        self._last_ts = now
        return duration_s


class BrowFlash(CueDetector):
    """Brow movement AU1/2 (browInnerUp) and AU4 (browDown) spikes (catalog cue 9)."""

    cue_id = "visual.brow_flash"
    direction = 1

    def measure(self, frame: FeatureFrame) -> float | None:
        bs = frame.blendshapes
        keys = ("browInnerUp", "browDownLeft", "browDownRight")
        if not any(k in bs for k in keys):
            return None
        inner = bs.get("browInnerUp", 0.0)
        down = (bs.get("browDownLeft", 0.0) + bs.get("browDownRight", 0.0)) / 2.0
        return max(inner, down)


class LipPress(CueDetector):
    """Lip compression / pucker (catalog cue 3)."""

    cue_id = "visual.lip_press"
    direction = 1

    def measure(self, frame: FeatureFrame) -> float | None:
        bs = frame.blendshapes
        keys = ("mouthPressLeft", "mouthPressRight", "mouthPucker")
        if not any(k in bs for k in keys):
            return None
        press = (bs.get("mouthPressLeft", 0.0) + bs.get("mouthPressRight", 0.0)) / 2.0
        return max(press, bs.get("mouthPucker", 0.0))


class JawTension(CueDetector):
    """Jaw-tension proxy from landmark-distance ratio (catalog cue 8, resolves Blocker 2/AU28)."""

    cue_id = "visual.jaw_tension"
    direction = 1

    def measure(self, frame: FeatureFrame) -> float | None:
        ratio = frame.geometry.get("jaw_width_ratio")
        return None if ratio is None else float(ratio)


VISUAL_DETECTORS = [BlinkRate, GazeAversion, BrowFlash, LipPress, JawTension]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/overlay/test_visual_cues.py -v`
Expected: PASS (7 passed). If `test_blink_rate_counts_blinks_per_minute` is brittle on timing, confirm the rising-edge logic counts 3 blinks; the assertion only checks `> 0`.

- [ ] **Step 5: Commit**

```bash
git add blitz_overlay/cues/visual.py tests/overlay/test_visual_cues.py
git commit -m "feat(overlay): 5 visual cue detectors (blink, gaze, brow, lip, jaw)"
```

---

### Task 7: rPPG DSP module (spec §11 — required second family)

**Files:**
- Create: `blitz_overlay/rppg.py`
- Test: `tests/overlay/test_rppg.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/overlay/test_rppg.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'blitz_overlay.rppg'`.

- [ ] **Step 3: Create `blitz_overlay/rppg.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/overlay/test_rppg.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add blitz_overlay/rppg.py tests/overlay/test_rppg.py
git commit -m "feat(overlay): CHROM rPPG heart-rate DSP"
```

---

### Task 8: rPPG heart-rate cue detector (physio family)

**Files:**
- Create: `blitz_overlay/cues/physio.py`
- Test: `tests/overlay/test_physio_cue.py`

- [ ] **Step 1: Write the failing test**

```python
import math
from blitz_overlay.cues.physio import RppgHeartRate
from blitz_overlay.schemas import FeatureFrame
from core.calibration import RollingBaseline


def _frame(ts, bpm):
    f = bpm / 60.0
    t = ts / 1000.0
    g = 120 + 8 * math.sin(2 * math.pi * f * t)
    return FeatureFrame.from_dict({
        "ts": ts, "face_present": True, "confidence": 0.9,
        "rppg": {"forehead_rgb": [180.0, g, 110.0], "cheek_rgb": [185.0, g, 112.0]},
    })


def test_measure_none_until_buffer_fills():
    d = RppgHeartRate(fps=30)
    assert d.measure(_frame(0, 72)) is None  # buffer not full


def test_measure_returns_bpm_once_buffer_fills():
    d = RppgHeartRate(fps=30)
    bpm = None
    for i in range(300):                       # 10s at 30fps
        bpm = d.measure(_frame(int(i * 1000 / 30), 72))
    assert bpm is not None and 60 <= bpm <= 84


def test_emits_event_when_hr_elevated_vs_baseline():
    d = RppgHeartRate(fps=30)
    rb = RollingBaseline(baseline_seconds=0, window_seconds=600)
    i = 0
    # baseline ~72 bpm
    for _ in range(450):
        ts = int(i * 1000 / 30); i += 1
        v = d.measure(_frame(ts, 72))
        if v is not None:
            rb.update({"physio.heart_rate": v}, ts_ms=ts)
    # elevate to ~105 bpm
    event = None
    for _ in range(450):
        ts = int(i * 1000 / 30); i += 1
        v = d.measure(_frame(ts, 105))
        if v is not None:
            rb.update({"physio.heart_rate": v}, ts_ms=ts)
            event = d.update(_frame(ts, 105), rb)
    assert event is not None
    assert event.modality.value == "physiological"
    assert event.region == "forehead"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/overlay/test_physio_cue.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'blitz_overlay.cues.physio'`.

- [ ] **Step 3: Create `blitz_overlay/cues/physio.py`**

```python
"""rPPG heart-rate cue — the physiological family voter (spec §11)."""
from __future__ import annotations

from collections import deque

from blitz_overlay.cues.base import CueDetector
from blitz_overlay.rppg import estimate_bpm
from blitz_overlay.schemas import FeatureFrame

WINDOW_SECONDS = 10
ESTIMATE_EVERY_MS = 1000  # recompute BPM at most once per second


class RppgHeartRate(CueDetector):
    cue_id = "physio.heart_rate"
    direction = 1  # elevated HR is the suspicious direction (autonomic arousal)

    def __init__(self, fps: float = 30.0) -> None:
        super().__init__()
        self.fps = fps
        self._buf: deque[tuple[int, list[float]]] = deque()  # (ts, [r,g,b]) forehead ROI
        self._last_estimate_ts = -ESTIMATE_EVERY_MS
        self._last_bpm: float | None = None

    def measure(self, frame: FeatureFrame) -> float | None:
        if not frame.rppg or "forehead_rgb" not in frame.rppg:
            return None
        now = frame.ts
        self._buf.append((now, [float(c) for c in frame.rppg["forehead_rgb"]]))
        cutoff = now - WINDOW_SECONDS * 1000
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()
        if now - self._last_estimate_ts < ESTIMATE_EVERY_MS:
            return self._last_bpm
        self._last_estimate_ts = now
        samples = [rgb for _, rgb in self._buf]
        self._last_bpm = estimate_bpm(samples, fps=self.fps)
        return self._last_bpm

    def quality(self, frame: FeatureFrame) -> float:
        base = super().quality(frame)
        fill = min(1.0, len(self._buf) / (WINDOW_SECONDS * self.fps))
        return base * fill
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/overlay/test_physio_cue.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add blitz_overlay/cues/physio.py tests/overlay/test_physio_cue.py
git commit -m "feat(overlay): rPPG heart-rate cue (physio family voter)"
```

---

### Task 9: Family-grouped fusion + two-gate (READINESS #6, #11)

**Files:**
- Modify: `core/fusion/bayesian_fusion.py`
- Test: `tests/overlay/test_family_fusion.py`

- [ ] **Step 1: Write the failing test**

```python
from core.fusion.bayesian_fusion import (
    fuse_by_family, two_gate, family_of, FAMILY_THRESHOLD,
)
from core.schemas.cue_event import CueEvent, Modality, Phase


def _cue(cue_id, modality, z, d=0.5, tier=2, quality=0.9):
    return CueEvent(cue_id=cue_id, modality=modality, timestamp_ms=0, phase=Phase.RESPONSE,
                    raw_value=0.0, z_score=z, llr=0.0, quality=quality, question_id="live",
                    effect_size_d=d, reliability_tier=tier)


def test_family_of_maps_modalities():
    assert family_of(Modality.VISUAL) == "visual"
    assert family_of(Modality.PHYSIOLOGICAL) == "physio"


def test_within_family_correlated_cues_do_not_stack_like_independent():
    # five strong visual cues should NOT add five independent log-odds chunks
    five = [_cue(f"visual.c{i}", Modality.VISUAL, z=3.0) for i in range(5)]
    one = [_cue("visual.c0", Modality.VISUAL, z=3.0)]
    res5 = fuse_by_family(five)
    res1 = fuse_by_family(one)
    # decorrelation: 5 correlated cues contribute clearly less than 5x a single cue
    assert res5["families"]["visual"] < 5 * res1["families"]["visual"]
    assert res5["families"]["visual"] >= res1["families"]["visual"]  # but at least the strongest


def test_two_gate_requires_two_independent_families():
    visual_only = [_cue("visual.a", Modality.VISUAL, z=4.0),
                   _cue("visual.b", Modality.VISUAL, z=4.0)]
    res = fuse_by_family(visual_only)
    gate = two_gate(res, threshold=0.65)
    assert gate["flag"] is False          # only one family
    assert gate["n_agree"] <= 1


def test_two_gate_flags_with_two_families_and_high_risk():
    cues = [_cue("visual.gaze", Modality.VISUAL, z=4.0, d=0.7),
            _cue("physio.hr", Modality.PHYSIOLOGICAL, z=4.0, d=0.5)]
    res = fuse_by_family(cues)
    gate = two_gate(res, threshold=0.65)
    assert gate["n_agree"] == 2
    assert res["posterior"] >= 0.65
    assert gate["flag"] is True


def test_family_vote_threshold_constant_exposed():
    assert 0.0 < FAMILY_THRESHOLD < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/overlay/test_family_fusion.py -v`
Expected: FAIL — `ImportError: cannot import name 'fuse_by_family'`.

- [ ] **Step 3: Append to `core/fusion/bayesian_fusion.py`**

Add these imports/usages at the bottom of the file (reusing existing `logit`, `sigmoid`, `compute_llr`, `cue_weight`):

```python
# --- Family-grouped fusion + two-gate (READINESS #6, #11) -------------------

from core.schemas.cue_event import Modality  # noqa: E402

# Map each modality to a consensus family. Visual+Physio are wired in Stage 1.
_FAMILY_OF = {
    Modality.VISUAL: "visual",
    Modality.PHYSIOLOGICAL: "physio",
    Modality.AUDIO: "audio",
    Modality.LINGUISTIC: "linguistic",
    Modality.CBCA: "linguistic",
}

# Within a family, the k-th strongest cue is down-weighted by DECORRELATION**k.
# This stops correlated cues in one family from faking independent agreement.
DECORRELATION = 0.5

# A family "votes flag" when its own log-odds contribution implies P >= this.
FAMILY_THRESHOLD = 0.60


def family_of(modality: Modality) -> str:
    return _FAMILY_OF[modality]


def fuse_by_family(cues, prior: float = DEFAULT_PRIOR) -> dict:
    """Group cues into families; decorrelate within family; sum family contributions.

    Returns combined posterior + per-family log-odds contributions + per-family vote.
    """
    grouped: Dict[str, List[CueEvent]] = {}
    for cue in cues:
        cue.llr = compute_llr(cue)
        grouped.setdefault(family_of(cue.modality), []).append(cue)

    family_contrib: Dict[str, float] = {}
    family_votes: Dict[str, bool] = {}
    base_log_odds = logit(prior)
    combined_log_odds = base_log_odds

    for family, members in grouped.items():
        # strongest-first; down-weight each subsequent (correlated) cue geometrically
        ranked = sorted(members, key=lambda c: abs(c.llr * cue_weight(c)), reverse=True)
        contribution = 0.0
        for rank, cue in enumerate(ranked):
            contribution += cue.llr * cue_weight(cue) * (DECORRELATION ** rank)
        family_contrib[family] = contribution
        combined_log_odds += contribution
        family_votes[family] = sigmoid(base_log_odds + contribution) >= FAMILY_THRESHOLD

    return {
        "posterior": sigmoid(combined_log_odds),
        "posterior_log_odds": combined_log_odds,
        "families": family_contrib,
        "family_votes": family_votes,
    }


def two_gate(fused: dict, threshold: float = 0.65) -> dict:
    """Two-gate convergence: >=2 independent families vote flag AND combined risk >= threshold."""
    agreeing = [fam for fam, vote in fused.get("family_votes", {}).items() if vote]
    n_agree = len(agreeing)
    gate1 = fused["posterior"] >= threshold
    gate2 = n_agree >= 2
    return {
        "flag": bool(gate1 and gate2),
        "n_agree": n_agree,
        "n_required": 2,
        "agreeing_families": agreeing,
        "gate1_risk": gate1,
        "gate2_convergence": gate2,
    }
```

> Verify `Dict`, `List`, `CueEvent` are already imported at the top of the file (they are). The `# noqa: E402` import of `Modality` is intentional to keep the change localized.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/overlay/test_family_fusion.py -v`
Expected: PASS (5 passed). If `test_two_gate_flags_with_two_families_and_high_risk` does not reach 0.65, raise the synthetic `z` in the test to 5.0 — the math is monotonic, the assertion checks the rule, not a tuned constant.

- [ ] **Step 5: Commit**

```bash
git add core/fusion/bayesian_fusion.py tests/overlay/test_family_fusion.py
git commit -m "feat(core): family-grouped fusion with decorrelation + two-gate (READINESS #6,#11)"
```

---

### Task 10: Consensus builder — status machine + voters + risk (spec §6, §7, §8)

**Files:**
- Create: `blitz_overlay/consensus.py`
- Test: `tests/overlay/test_consensus.py`

- [ ] **Step 1: Write the failing test**

```python
from blitz_overlay.consensus import ConsensusBuilder
from blitz_overlay.schemas import Consensus
from core.schemas.cue_event import CueEvent, Modality, Phase


def _cue(cue_id, modality, z, region, d=0.6):
    return CueEvent(cue_id=cue_id, modality=modality, timestamp_ms=0, phase=Phase.RESPONSE,
                    raw_value=0.0, z_score=z, llr=0.0, quality=0.9, question_id="live",
                    effect_size_d=d, reliability_tier=2)


def test_calibrating_blocks_flags():
    cb = ConsensusBuilder()
    out = cb.build(cues=[_cue("visual.gaze_aversion", Modality.VISUAL, 5.0, "eyes")],
                   calibrating=True, ts=1000, regions={"visual.gaze_aversion": "eyes"})
    assert isinstance(out, Consensus)
    assert out.status == "CALIBRATING"
    assert out.flag is False


def test_clear_when_no_cues():
    cb = ConsensusBuilder()
    out = cb.build(cues=[], calibrating=False, ts=1000, regions={})
    assert out.status == "CLEAR"
    assert out.risk < 0.65


def test_watch_when_single_family_elevated():
    cb = ConsensusBuilder()
    cues = [_cue("visual.gaze_aversion", Modality.VISUAL, 6.0, "eyes"),
            _cue("visual.brow_flash", Modality.VISUAL, 6.0, "brow")]
    out = cb.build(cues=cues, calibrating=False, ts=1000,
                   regions={"visual.gaze_aversion": "eyes", "visual.brow_flash": "brow"})
    assert out.status == "WATCH"   # one family cannot satisfy the two-gate
    assert out.flag is False
    assert out.n_required == 2


def test_flag_only_under_two_gate():
    cb = ConsensusBuilder()
    cues = [_cue("visual.gaze_aversion", Modality.VISUAL, 7.0, "eyes", d=0.7),
            _cue("physio.heart_rate", Modality.PHYSIOLOGICAL, 7.0, "forehead", d=0.5)]
    out = cb.build(cues=cues, calibrating=False, ts=1000,
                   regions={"visual.gaze_aversion": "eyes", "physio.heart_rate": "forehead"})
    assert out.status == "FLAG"
    assert out.flag is True
    assert out.n_agree == 2


def test_unwired_families_shown_not_fresh():
    cb = ConsensusBuilder()
    out = cb.build(cues=[], calibrating=False, ts=1000, regions={})
    names = {f.name: f for f in out.families}
    assert names["audio"].wired is False and names["audio"].fresh is False
    assert names["linguistic"].wired is False
    assert names["visual"].wired is True


def test_active_cues_carry_region_for_telestrator():
    cb = ConsensusBuilder()
    cues = [_cue("visual.lip_press", Modality.VISUAL, 4.0, "mouth")]
    out = cb.build(cues=cues, calibrating=False, ts=1000,
                   regions={"visual.lip_press": "mouth"})
    assert out.active_cues[0].region == "mouth"
    assert out.active_cues[0].cue_id == "visual.lip_press"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/overlay/test_consensus.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'blitz_overlay.consensus'`.

- [ ] **Step 3: Create `blitz_overlay/consensus.py`**

```python
"""Consensus builder: turns scored cues into the honest status payload (spec §6–§8)."""
from __future__ import annotations

from blitz_overlay.schemas import ActiveCue, Consensus, FamilyVote, SCHEMA_VERSION
from core.fusion.bayesian_fusion import family_of, fuse_by_family, two_gate

# Families displayed as voters, in panel order. Stage 1 wires visual + physio only.
WIRED_FAMILIES = {"visual", "physio"}
PANEL_FAMILIES = ["visual", "physio", "audio", "linguistic"]

WATCH_RISK = 0.45  # risk above this (but not a FLAG) shows WATCH


class ConsensusBuilder:
    def __init__(self, gate_threshold: float = 0.65):
        self.gate_threshold = gate_threshold

    def build(self, cues, calibrating: bool, ts: int, regions: dict[str, str],
              message: str = "") -> Consensus:
        fused = fuse_by_family(cues)
        gate = two_gate(fused, threshold=self.gate_threshold)
        risk = fused["posterior"]
        votes = fused.get("family_votes", {})
        contrib = fused.get("families", {})
        fresh_families = {family_of(c.modality) for c in cues}

        families = []
        for name in PANEL_FAMILIES:
            wired = name in WIRED_FAMILIES
            families.append(FamilyVote(
                name=name,
                wired=wired,
                fresh=wired and name in fresh_families,
                vote=bool(votes.get(name, False)) and wired,
                contribution=float(contrib.get(name, 0.0)),
            ))

        active = [
            ActiveCue(cue_id=c.cue_id, region=regions.get(c.cue_id, "head"),
                      z=c.z_score, confidence=c.quality)
            for c in sorted(cues, key=lambda c: abs(c.z_score), reverse=True)
        ]

        if calibrating:
            status, flag = "CALIBRATING", False
            risk = 0.0
            message = message or "Calibrating personal baseline — no flags permitted yet."
        elif gate["flag"]:
            status, flag = "FLAG", True
        elif risk >= WATCH_RISK or any(f.vote for f in families):
            status, flag = "WATCH", False
        else:
            status, flag = "CLEAR", False

        # Honest cap (spec §8): if fewer than 2 wired families are fresh, FLAG is unreachable.
        if status == "WATCH" and len([f for f in families if f.fresh]) < 2 and not message:
            message = "Only one family active — capped at WATCH until a second family agrees."

        return Consensus(
            schema_version=SCHEMA_VERSION,
            ts=ts,
            status=status,
            risk=risk,
            flag=flag,
            n_agree=gate["n_agree"] if not calibrating else 0,
            n_required=gate["n_required"],
            families=families,
            active_cues=active,
            message=message,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/overlay/test_consensus.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add blitz_overlay/consensus.py tests/overlay/test_consensus.py
git commit -m "feat(overlay): consensus builder with CALIBRATING/CLEAR/WATCH/FLAG status machine"
```

---

### Task 11: Prediction logger (READINESS #8, #15)

**Files:**
- Create: `blitz_overlay/logger.py`
- Test: `tests/overlay/test_logger.py`

- [ ] **Step 1: Write the failing test**

```python
import json
from pathlib import Path
from blitz_overlay.logger import PredictionLogger
from blitz_overlay.schemas import Consensus, FamilyVote, ActiveCue, SCHEMA_VERSION


def _consensus():
    return Consensus(
        schema_version=SCHEMA_VERSION, ts=1234, status="WATCH", risk=0.5, flag=False,
        n_agree=1, n_required=2,
        families=[FamilyVote("visual", True, True, False, 0.3)],
        active_cues=[ActiveCue("visual.gaze_aversion", "eyes", 2.5, 0.8)],
    )


def test_logger_writes_jsonl_line(tmp_path):
    log = PredictionLogger(session_id="sess1", log_dir=tmp_path)
    log.log(_consensus(), baseline_mode="rolling")
    files = list(Path(tmp_path).glob("*.jsonl"))
    assert len(files) == 1
    line = json.loads(files[0].read_text().strip())
    assert line["status"] == "WATCH"
    assert line["posterior"] == 0.5
    assert line["baseline_mode"] == "rolling"
    assert line["weight_set_version"]
    assert line["schema_version"] == SCHEMA_VERSION


def test_logger_records_cue_contributions_not_raw_biometric(tmp_path):
    log = PredictionLogger(session_id="sess2", log_dir=tmp_path)
    log.log(_consensus(), baseline_mode="rolling")
    line = json.loads(next(Path(tmp_path).glob("*.jsonl")).read_text().strip())
    assert "active_cues" in line
    assert line["active_cues"][0]["cue_id"] == "visual.gaze_aversion"
    # privacy: no landmarks/blendshapes/raw frame data must appear
    raw = next(Path(tmp_path).glob("*.jsonl")).read_text()
    assert "blendshapes" not in raw and "landmarks" not in raw and "rgb" not in raw


def test_logger_appends_multiple_lines(tmp_path):
    log = PredictionLogger(session_id="sess3", log_dir=tmp_path)
    log.log(_consensus(), baseline_mode="rolling")
    log.log(_consensus(), baseline_mode="rolling")
    lines = next(Path(tmp_path).glob("*.jsonl")).read_text().strip().splitlines()
    assert len(lines) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/overlay/test_logger.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'blitz_overlay.logger'`.

- [ ] **Step 3: Create `blitz_overlay/logger.py`**

```python
"""Append-only prediction logger (READINESS #8). Logging != learning.

Stores only derived decision data — status, posterior, per-family contributions, active
cue z-scores, baseline mode, weight-set version, schema version. NEVER raw biometric
(no landmarks/blendshapes/RGB), honoring the privacy posture (READINESS #15).
"""
from __future__ import annotations

import json
from pathlib import Path

from blitz_overlay.schemas import Consensus
from blitz_overlay.weights import WEIGHT_SET_VERSION


class PredictionLogger:
    def __init__(self, session_id: str, log_dir: str | Path = "logs"):
        self.session_id = session_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / f"predictions-{session_id}.jsonl"

    def log(self, consensus: Consensus, baseline_mode: str) -> None:
        record = {
            "session_id": self.session_id,
            "ts": consensus.ts,
            "status": consensus.status,
            "posterior": consensus.risk,
            "flag": consensus.flag,
            "n_agree": consensus.n_agree,
            "n_required": consensus.n_required,
            "families": [f.to_dict() for f in consensus.families],
            "active_cues": [c.to_dict() for c in consensus.active_cues],
            "baseline_mode": baseline_mode,
            "weight_set_version": WEIGHT_SET_VERSION,
            "schema_version": consensus.schema_version,
            "ground_truth": None,  # empty audit slot; never auto-filled (no learning loop)
        }
        with self.path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/overlay/test_logger.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add blitz_overlay/logger.py tests/overlay/test_logger.py
git commit -m "feat(overlay): append-only prediction logger (no raw biometric)"
```

---

### Task 12: Pipeline — OverlaySession wiring (spec §3, §4 engine/)

**Files:**
- Create: `blitz_overlay/pipeline.py`
- Test: `tests/overlay/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
import math
from blitz_overlay.pipeline import OverlaySession


def _feature(ts, *, gaze=0.02, bpm=72, blink=0.05, jaw=0.80, confidence=0.9, face=True):
    f = bpm / 60.0
    t = ts / 1000.0
    g = 120 + 8 * math.sin(2 * math.pi * f * t)
    return {
        "ts": ts, "face_present": face, "confidence": confidence,
        "blendshapes": {"eyeBlinkLeft": blink, "eyeBlinkRight": blink,
                        "browInnerUp": 0.05, "mouthPressLeft": 0.05, "mouthPressRight": 0.05},
        "geometry": {"gaze_x": gaze, "gaze_y": 0.0, "jaw_width_ratio": jaw},
        "head_pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
        "rppg": {"forehead_rgb": [180.0, g, 110.0], "cheek_rgb": [185.0, g, 112.0]},
    }


def test_starts_calibrating(tmp_path):
    sess = OverlaySession(gate_threshold=0.65, baseline_seconds=5, log_dir=tmp_path)
    c = sess.process(_feature(0))
    assert c.status == "CALIBRATING"


def test_no_face_pauses_cues(tmp_path):
    sess = OverlaySession(gate_threshold=0.65, baseline_seconds=0, log_dir=tmp_path)
    c = sess.process(_feature(0, face=False))
    assert c.active_cues == []
    assert "no subject" in c.message.lower() or c.status in ("CLEAR", "CALIBRATING")


def test_calm_session_stays_clear_or_watch_never_flags(tmp_path):
    sess = OverlaySession(gate_threshold=0.65, baseline_seconds=3, log_dir=tmp_path)
    last = None
    for i in range(400):
        last = sess.process(_feature(int(i * 1000 / 30), gaze=0.02, bpm=72))
    assert last.flag is False
    assert last.status in ("CLEAR", "WATCH")


def test_writes_prediction_log(tmp_path):
    sess = OverlaySession(gate_threshold=0.65, baseline_seconds=0, log_dir=tmp_path)
    sess.process(_feature(0))
    sess.process(_feature(100))
    logs = list(tmp_path.glob("*.jsonl"))
    assert logs and logs[0].read_text().strip()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/overlay/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'blitz_overlay.pipeline'`.

- [ ] **Step 3: Create `blitz_overlay/pipeline.py`**

```python
"""Per-connection live pipeline: feature frame -> consensus payload (spec §3)."""
from __future__ import annotations

import uuid
from pathlib import Path

from blitz_overlay.consensus import ConsensusBuilder
from blitz_overlay.cues.physio import RppgHeartRate
from blitz_overlay.cues.visual import VISUAL_DETECTORS
from blitz_overlay.logger import PredictionLogger
from blitz_overlay.schemas import Consensus, FeatureFrame
from core.calibration import RollingBaseline

EMIT_EVERY_MS = 100  # throttle consensus emission to ~10 Hz


class OverlaySession:
    def __init__(self, gate_threshold: float = 0.65, baseline_seconds: int = 90,
                 fps: float = 30.0, log_dir: str | Path = "logs"):
        self.session_id = uuid.uuid4().hex[:12]
        self.detectors = [cls() for cls in VISUAL_DETECTORS] + [RppgHeartRate(fps=fps)]
        self.baseline = RollingBaseline(baseline_seconds=baseline_seconds)
        self.consensus = ConsensusBuilder(gate_threshold=gate_threshold)
        self.logger = PredictionLogger(self.session_id, log_dir=log_dir)
        self.regions = {d.cue_id: d.region for d in self.detectors}
        self._last_emit_ts = -EMIT_EVERY_MS
        self._last_consensus: Consensus | None = None

    def process(self, raw: dict) -> Consensus:
        frame = FeatureFrame.from_dict(raw)

        if not frame.face_present:
            out = self.consensus.build(
                cues=[], calibrating=self.baseline.is_calibrating, ts=frame.ts,
                regions=self.regions, message="No subject detected — cues paused.")
            self._last_consensus = out
            self.logger.log(out, baseline_mode=self.baseline.mode)
            return out

        # 1) measure every cue, 2) feed the baseline once per frame, 3) score deviations
        measurements: dict[str, float] = {}
        for det in self.detectors:
            value = det.measure(frame)
            if value is not None:
                measurements[det.cue_id] = value
        self.baseline.update(measurements, ts_ms=frame.ts)

        cues = []
        for det in self.detectors:
            if det.cue_id not in measurements:
                continue
            event = det.update(frame, self.baseline)
            if event is not None:
                cues.append(event)

        out = self.consensus.build(
            cues=cues, calibrating=self.baseline.is_calibrating, ts=frame.ts,
            regions=self.regions)
        self._last_consensus = out
        self.logger.log(out, baseline_mode=self.baseline.mode)
        return out

    def should_emit(self, ts: int) -> bool:
        if ts - self._last_emit_ts >= EMIT_EVERY_MS:
            self._last_emit_ts = ts
            return True
        return False
```

> Note: `det.update` recomputes `measure` internally. That double-call is cheap and keeps detectors self-contained; the baseline was already updated from `measurements`, so `det.update` only normalizes. This avoids the stateful blink/gaze detectors advancing their internal edge state twice — to prevent that, `det.update` must reuse the measured value. **Apply the fix in Step 4.**

- [ ] **Step 4: Fix double-measure on stateful detectors**

In `blitz_overlay/cues/base.py`, add an optional precomputed value to `update`:

```python
    def update(self, frame: FeatureFrame, baseline: RollingBaseline,
               value: float | None = None) -> CueEvent | None:
        if value is None:
            value = self.measure(frame)
        if value is None:
            return None
        z = baseline.normalize(self.cue_id, value)
        directed_z = z * self.direction
        if directed_z < self.z_threshold:
            return None
        return CueEvent(
            cue_id=self.cue_id, modality=self.modality, timestamp_ms=frame.ts,
            phase=Phase.RESPONSE, raw_value=float(value), z_score=directed_z, llr=0.0,
            quality=self.quality(frame), question_id="live",
            effect_size_d=self.effect_size_d, reliability_tier=self.reliability_tier,
        )
```

Then in `pipeline.py` pass the cached value:

```python
            event = det.update(frame, self.baseline, value=measurements[det.cue_id])
```

Re-run the Task 5 cue-base tests to confirm the new signature is backward compatible:
Run: `python -m pytest tests/overlay/test_cue_base.py -v` → Expected: PASS.

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/overlay/test_pipeline.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add blitz_overlay/pipeline.py blitz_overlay/cues/base.py tests/overlay/test_pipeline.py
git commit -m "feat(overlay): OverlaySession pipeline wiring detectors+baseline+consensus+log"
```

---

### Task 13: Deterministic replay test (spec §10, READINESS #17)

**Files:**
- Create: `tests/overlay/fixtures/replay_session.py` (fixture generator → deterministic frames)
- Test: `tests/overlay/test_replay.py`

We generate the frame stream programmatically (deterministic, seedless math) rather than committing a large JSONL, then assert the consensus sequence: CALIBRATING → … → a two-gate FLAG once gaze aversion (visual) + elevated HR (physio) converge.

- [ ] **Step 1: Write the failing test**

```python
from tests.overlay.fixtures.replay_session import replay_frames
from blitz_overlay.pipeline import OverlaySession


def test_replay_is_deterministic(tmp_path):
    def run():
        sess = OverlaySession(gate_threshold=0.65, baseline_seconds=30, log_dir=tmp_path)
        return [sess.process(f).status for f in replay_frames()]
    assert run() == run()  # identical output across runs


def test_replay_progresses_calibrating_then_reaches_flag(tmp_path):
    sess = OverlaySession(gate_threshold=0.65, baseline_seconds=30, log_dir=tmp_path)
    statuses = [sess.process(f).status for f in replay_frames()]
    assert statuses[0] == "CALIBRATING"
    assert "FLAG" in statuses
    assert statuses.index("CALIBRATING") < statuses.index("FLAG")


def test_replay_flag_requires_two_families(tmp_path):
    sess = OverlaySession(gate_threshold=0.65, baseline_seconds=30, log_dir=tmp_path)
    flag_consensus = None
    for f in replay_frames():
        c = sess.process(f)
        if c.flag:
            flag_consensus = c
            break
    assert flag_consensus is not None
    assert flag_consensus.n_agree >= 2
    fresh = [fam.name for fam in flag_consensus.families if fam.fresh]
    assert "visual" in fresh and "physio" in fresh
```

- [ ] **Step 2: Create the fixture generator `tests/overlay/fixtures/__init__.py` (empty) and `tests/overlay/fixtures/replay_session.py`**

```python
"""Deterministic synthetic feature stream for the replay test (spec §10).

Three phases over ~95s at 30fps:
  - calm baseline (centered gaze, ~72 bpm)        -> CALIBRATING then CLEAR
  - stress onset (sustained gaze aversion + ~104 bpm) -> WATCH then FLAG (two families)
No randomness: every value is a deterministic function of frame index.
"""
from __future__ import annotations

import math

FPS = 30
DT_MS = int(1000 / FPS)


def _rgb(ts_ms: int, bpm: float) -> list[float]:
    f = bpm / 60.0
    t = ts_ms / 1000.0
    green = 120 + 8 * math.sin(2 * math.pi * f * t)
    return [180.0, green, 110.0]


def _frame(idx: int, *, gaze: float, bpm: float) -> dict:
    ts = idx * DT_MS
    # deterministic micro-jitter so MAD != 0 in the baseline window
    jitter = 0.01 * math.sin(idx * 0.7)
    return {
        "ts": ts,
        "face_present": True,
        "confidence": 0.92,
        "blendshapes": {
            "eyeBlinkLeft": 0.05, "eyeBlinkRight": 0.05,
            "browInnerUp": 0.05 + abs(jitter), "browDownLeft": 0.04, "browDownRight": 0.04,
            "mouthPressLeft": 0.05, "mouthPressRight": 0.05, "mouthPucker": 0.03,
        },
        "geometry": {"gaze_x": gaze + jitter, "gaze_y": 0.0, "jaw_width_ratio": 0.80 + jitter},
        "head_pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
        "rppg": {"forehead_rgb": _rgb(ts, bpm), "cheek_rgb": _rgb(ts, bpm)},
    }


def replay_frames() -> list[dict]:
    frames: list[dict] = []
    idx = 0
    # Phase A: 40s calm baseline (centered gaze ~0.02, 72 bpm)
    for _ in range(FPS * 40):
        frames.append(_frame(idx, gaze=0.02, bpm=72)); idx += 1
    # Phase B: 20s sustained stress (strong gaze aversion + elevated HR 104 bpm)
    for _ in range(FPS * 20):
        frames.append(_frame(idx, gaze=0.75, bpm=104)); idx += 1
    return frames
```

- [ ] **Step 3: Run test to verify it fails then passes**

Run: `python -m pytest tests/overlay/test_replay.py -v`
Expected first: it may FAIL if `FLAG` is never reached. Diagnose with:
`python -c "from tests.overlay.fixtures.replay_session import replay_frames; from blitz_overlay.pipeline import OverlaySession; s=OverlaySession(baseline_seconds=30, log_dir='/tmp/r'); import collections; print(collections.Counter(s.process(f).status for f in replay_frames()))"`

If `FLAG` is absent, the physio family likely isn't reaching its vote threshold because rPPG needs the 10s buffer to fill within Phase B. Phase B is 20s, so HR estimates appear ~10s in. Two adjustments, in order, until FLAG appears:
1. Extend Phase B to `FPS * 25`.
2. If gaze z or HR z is below the family threshold, raise Phase B `gaze` to `0.9` and `bpm` to `110` (still honest synthetic stress, the test asserts the *rule*, not tuned constants).

Re-run: `python -m pytest tests/overlay/test_replay.py -v`
Expected: PASS (3 passed).

- [ ] **Step 4: Commit**

```bash
git add tests/overlay/fixtures/ tests/overlay/test_replay.py
git commit -m "test(overlay): deterministic replay reaching two-gate FLAG (READINESS #17)"
```

---

### Task 14: Config + FastAPI server + entrypoint (one-command start)

**Files:**
- Create: `blitz_overlay/config.py`
- Create: `blitz_overlay/server.py`
- Create: `blitz_overlay/__main__.py`
- Test: `tests/overlay/test_server.py`

- [ ] **Step 1: Write the failing test**

```python
from fastapi.testclient import TestClient
from blitz_overlay.server import create_app
from blitz_overlay.config import OverlayConfig


def test_config_reads_env(monkeypatch):
    monkeypatch.setenv("BLITZ_OVERLAY_PORT", "9001")
    monkeypatch.setenv("BLITZ_OVERLAY_GATE", "0.7")
    monkeypatch.setenv("BLITZ_OVERLAY_BASELINE_SECONDS", "45")
    cfg = OverlayConfig.from_env()
    assert cfg.port == 9001
    assert cfg.gate == 0.7
    assert cfg.baseline_seconds == 45


def test_index_served(tmp_path):
    app = create_app(OverlayConfig(log_dir=str(tmp_path)))
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Live Consensus Overlay" in resp.text


def test_ws_returns_consensus_for_a_feature_frame(tmp_path):
    app = create_app(OverlayConfig(baseline_seconds=0, log_dir=str(tmp_path)))
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"ts": 0, "face_present": True, "confidence": 0.9,
                      "blendshapes": {"eyeBlinkLeft": 0.05},
                      "geometry": {"gaze_x": 0.02, "gaze_y": 0.0, "jaw_width_ratio": 0.8}})
        msg = ws.receive_json()
        assert msg["schema_version"]
        assert msg["status"] in ("CALIBRATING", "CLEAR", "WATCH", "FLAG")
        assert "families" in msg
```

> `TestClient` needs `httpx`; if missing, install with `python -m pip install httpx` and add it to the `dev` extra in `pyproject.toml`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/overlay/test_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'blitz_overlay.server'`.

- [ ] **Step 3: Create `blitz_overlay/config.py`**

```python
"""Typed config loaded from environment / .env (READINESS #16). No secrets needed in Stage 1."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


@dataclass
class OverlayConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    gate: float = 0.65
    baseline_seconds: int = 90
    open_browser: bool = True
    log_dir: str = "logs"

    @classmethod
    def from_env(cls) -> "OverlayConfig":
        _load_dotenv()
        return cls(
            host=os.environ.get("BLITZ_OVERLAY_HOST", "127.0.0.1"),
            port=int(os.environ.get("BLITZ_OVERLAY_PORT", "8000")),
            gate=float(os.environ.get("BLITZ_OVERLAY_GATE", "0.65")),
            baseline_seconds=int(os.environ.get("BLITZ_OVERLAY_BASELINE_SECONDS", "90")),
            open_browser=os.environ.get("BLITZ_OVERLAY_OPEN_BROWSER", "1") == "1",
            log_dir=os.environ.get("BLITZ_OVERLAY_LOG_DIR", "logs"),
        )
```

- [ ] **Step 4: Create `blitz_overlay/server.py`**

```python
"""FastAPI app: serves the browser overlay (static) and the /ws feature-frame endpoint.

One process = engine + browser host, so the whole thing starts with one command.
Raw video never reaches here — only feature frames (spec §3).
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from blitz_overlay.config import OverlayConfig
from blitz_overlay.pipeline import OverlaySession

WEB_DIR = Path(__file__).resolve().parents[1] / "apps" / "overlay-web"


def create_app(config: OverlayConfig | None = None) -> FastAPI:
    config = config or OverlayConfig.from_env()
    app = FastAPI(title="Blitz Live Consensus Overlay")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        session = OverlaySession(
            gate_threshold=config.gate,
            baseline_seconds=config.baseline_seconds,
            log_dir=config.log_dir,
        )
        try:
            while True:
                raw = await websocket.receive_json()
                consensus = session.process(raw)
                if session.should_emit(consensus.ts):
                    await websocket.send_json(consensus.to_dict())
        except WebSocketDisconnect:
            return

    # Static assets (css/js) under /static; index.html is served at "/".
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app
```

> The test sends a single frame and expects exactly one consensus back. `should_emit` starts at `-100`, and the first frame ts is 0, so `0 - (-100) >= 100` → emits. Good. For multi-frame clients the 10 Hz throttle applies.

- [ ] **Step 5: Create `blitz_overlay/__main__.py`**

```python
"""`python -m blitz_overlay` / `blitz-overlay` — start engine + browser host (one command)."""
from __future__ import annotations

import threading
import webbrowser

import uvicorn

from blitz_overlay.config import OverlayConfig
from blitz_overlay.server import create_app


def main() -> None:
    config = OverlayConfig.from_env()
    app = create_app(config)
    url = f"http://{config.host}:{config.port}/"
    print("\n  Blitz Live Consensus Overlay")
    print(f"  → open {url} and allow camera access")
    print("  Raw video never leaves your device; only feature vectors reach the engine.\n")
    if config.open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=config.host, port=config.port, log_level="warning")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/overlay/test_server.py -v`
Expected: PASS (3 passed). The index test requires `apps/overlay-web/index.html` to exist and contain "Live Consensus Overlay" — Task 15 creates it. **Order note:** if running Task 14 before 15, create a minimal `apps/overlay-web/index.html` placeholder containing the title now, then flesh it out in Task 15. Add this placeholder:

```bash
mkdir -p apps/overlay-web
printf '<!doctype html><title>Live Consensus Overlay</title>\n' > apps/overlay-web/index.html
```

- [ ] **Step 7: Commit**

```bash
git add blitz_overlay/config.py blitz_overlay/server.py blitz_overlay/__main__.py apps/overlay-web/index.html tests/overlay/test_server.py
git commit -m "feat(overlay): FastAPI server (static + /ws) + one-command entrypoint + config"
```

---

### Task 15: Browser app — capture, MediaPipe, rPPG, ws-client, overlay (spec §4 browser/, §5–§8)

The browser is verified manually (Step 9), but each module is small and isolated. Create files in dependency order. No bundler — native ES modules loaded by `index.html`.

**Files:**
- Create: `apps/overlay-web/js/schema.js`
- Create: `apps/overlay-web/js/regions.js`
- Create: `apps/overlay-web/js/capture.js`
- Create: `apps/overlay-web/js/mediapipe-extractor.js`
- Create: `apps/overlay-web/js/rppg-sampler.js`
- Create: `apps/overlay-web/js/ws-client.js`
- Create: `apps/overlay-web/js/overlay-renderer.js`
- Create: `apps/overlay-web/js/main.js`
- Create: `apps/overlay-web/css/overlay.css`
- Replace: `apps/overlay-web/index.html`

- [ ] **Step 1: `apps/overlay-web/js/schema.js`** (mirror of Python contract)

```javascript
// Mirror of blitz_overlay/schemas.py — keep SCHEMA_VERSION in sync.
export const SCHEMA_VERSION = "1.0";

// Blendshape coefficient names we forward (MediaPipe FaceLandmarker, 52 categories).
export const USED_BLENDSHAPES = [
  "eyeBlinkLeft", "eyeBlinkRight", "eyeWideLeft", "eyeWideRight",
  "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeLookDownLeft",
  "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeLookDownRight",
  "browInnerUp", "browDownLeft", "browDownRight",
  "mouthPressLeft", "mouthPressRight", "mouthPucker",
];
```

- [ ] **Step 2: `apps/overlay-web/js/regions.js`** (telestrator anchors → FaceMesh landmark indices)

```javascript
// Region -> representative FaceMesh landmark indices (478-point model). Used to draw
// telestrator circles where a cue fired. Indices are stable across the canonical mesh.
export const REGION_LANDMARKS = {
  eyes: [33, 133, 362, 263],     // outer/inner corners L & R
  brow: [105, 334, 70, 300],     // brow ridge L & R
  mouth: [61, 291, 13, 14],      // mouth corners + lip center
  jaw: [172, 397, 152],          // gonial L/R + chin
  forehead: [10, 67, 297],       // forehead center + sides (rPPG ROI)
  head: [1],                     // nose tip
  body: [],                      // pose-driven (unused in v1 telestrator)
};

// Average a set of landmark {x,y} (normalized) into one canvas point.
export function regionCenter(region, landmarks, w, h) {
  const idxs = REGION_LANDMARKS[region] || [1];
  let sx = 0, sy = 0, n = 0;
  for (const i of idxs) {
    const p = landmarks[i];
    if (!p) continue;
    sx += p.x; sy += p.y; n++;
  }
  if (n === 0) return null;
  return { x: (sx / n) * w, y: (sy / n) * h };
}
```

- [ ] **Step 3: `apps/overlay-web/js/capture.js`** (swappable source adapter — webcam only in v1)

```javascript
// Capture source adapter. Stage 1 = WebcamSource. Builds 2/3 (screen region, native
// draw-anywhere) implement the same interface: start() -> HTMLVideoElement, stop().
export class WebcamSource {
  constructor(video) { this.video = video; this.stream = null; }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, frameRate: 30 }, audio: false,
    });
    this.video.srcObject = this.stream;
    await this.video.play();
    return this.video;
  }

  stop() {
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    this.stream = null;
  }
}
```

- [ ] **Step 4: `apps/overlay-web/js/mediapipe-extractor.js`** (Face Landmarker → feature frame)

```javascript
import { SCHEMA_VERSION, USED_BLENDSHAPES } from "./schema.js";
import { FaceLandmarker, FilesetResolver } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

export class MediaPipeExtractor {
  constructor() { this.landmarker = null; this.lastLandmarks = null; }

  async init() {
    const fileset = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm");
    this.landmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    });
  }

  // Returns a feature frame (the WS payload) or a face_present:false frame.
  extract(video, tsMs) {
    const result = this.landmarker.detectForVideo(video, tsMs);
    const faces = result.faceLandmarks;
    if (!faces || faces.length === 0) {
      this.lastLandmarks = null;
      return { schema_version: SCHEMA_VERSION, ts: tsMs, face_present: false, confidence: 0 };
    }
    const landmarks = faces[0];
    this.lastLandmarks = landmarks;

    const blendshapes = {};
    const cats = (result.faceBlendshapes?.[0]?.categories) || [];
    for (const c of cats) {
      if (USED_BLENDSHAPES.includes(c.categoryName)) blendshapes[c.categoryName] = c.score;
    }

    const headPose = this._headPose(result.facialTransformationMatrixes?.[0]);
    const geometry = this._geometry(landmarks, blendshapes);

    return {
      schema_version: SCHEMA_VERSION,
      ts: tsMs,
      face_present: true,
      confidence: 0.9,                 // FaceLandmarker has no per-face score; proxy when present
      blendshapes,
      head_pose: headPose,
      geometry,
    };
  }

  _headPose(matrix) {
    if (!matrix || !matrix.data) return { yaw: 0, pitch: 0, roll: 0 };
    const m = matrix.data; // column-major 4x4
    const yaw = Math.atan2(m[8], m[10]) * 180 / Math.PI;
    const pitch = Math.atan2(-m[9], Math.hypot(m[8], m[10])) * 180 / Math.PI;
    const roll = Math.atan2(m[1], m[5]) * 180 / Math.PI;
    return { yaw, pitch, roll };
  }

  _dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

  _geometry(lm, bs) {
    // Stable normalizer: outer-eye-corner distance (33 <-> 263).
    const eyeSpan = this._dist(lm[33], lm[263]) || 1e-6;
    // Jaw width ratio (gonial 172 <-> 397) normalized by eye span -> AU28 proxy (Blocker 2).
    const jawWidthRatio = this._dist(lm[172], lm[397]) / eyeSpan;
    // Gaze offset from look blendshapes (left/right, up/down), [-1,1].
    const gx = ((bs.eyeLookOutLeft || 0) + (bs.eyeLookInRight || 0))
             - ((bs.eyeLookInLeft || 0) + (bs.eyeLookOutRight || 0));
    const gy = ((bs.eyeLookUpLeft || 0) + (bs.eyeLookUpRight || 0))
             - ((bs.eyeLookDownLeft || 0) + (bs.eyeLookDownRight || 0));
    return { jaw_width_ratio: jawWidthRatio, gaze_x: gx / 2, gaze_y: gy / 2 };
  }
}
```

- [ ] **Step 5: `apps/overlay-web/js/rppg-sampler.js`** (ROI mean RGB from landmarks)

```javascript
import { REGION_LANDMARKS } from "./regions.js";

// Samples mean RGB of forehead + cheek ROIs from the live video, off-screen.
// Only the 3-number means are forwarded — never pixels (privacy, spec §3).
export class RppgSampler {
  constructor() {
    this.canvas = document.createElement("canvas");
    this.canvas.width = 64; this.canvas.height = 64;
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
  }

  _roiMean(video, landmarks, idxs) {
    let cx = 0, cy = 0, n = 0;
    for (const i of idxs) { const p = landmarks[i]; if (p) { cx += p.x; cy += p.y; n++; } }
    if (n === 0) return [0, 0, 0];
    cx /= n; cy /= n;
    const vw = video.videoWidth, vh = video.videoHeight;
    const boxW = vw * 0.12, boxH = vh * 0.08;
    const sx = Math.max(0, cx * vw - boxW / 2), sy = Math.max(0, cy * vh - boxH / 2);
    this.ctx.drawImage(video, sx, sy, boxW, boxH, 0, 0, 64, 64);
    const data = this.ctx.getImageData(0, 0, 64, 64).data;
    let r = 0, g = 0, b = 0;
    for (let i = 0; i < data.length; i += 4) { r += data[i]; g += data[i + 1]; b += data[i + 2]; }
    const px = data.length / 4;
    return [r / px, g / px, b / px];
  }

  sample(video, landmarks) {
    if (!landmarks) return null;
    return {
      forehead_rgb: this._roiMean(video, landmarks, REGION_LANDMARKS.forehead),
      cheek_rgb: this._roiMean(video, landmarks, [50, 280]),  // left/right cheek
    };
  }
}
```

- [ ] **Step 6: `apps/overlay-web/js/ws-client.js`** (send frames, receive consensus, reconnect)

```javascript
export class WsClient {
  constructor(url, onConsensus, onStatus) {
    this.url = url; this.onConsensus = onConsensus; this.onStatus = onStatus;
    this.ws = null; this.connected = false; this._reconnectTimer = null;
  }

  connect() {
    this.ws = new WebSocket(this.url);
    this.ws.onopen = () => { this.connected = true; this.onStatus("engine-online"); };
    this.ws.onmessage = (e) => this.onConsensus(JSON.parse(e.data));
    this.ws.onclose = () => {
      this.connected = false; this.onStatus("engine-offline");
      this._reconnectTimer = setTimeout(() => this.connect(), 1500);  // auto-reconnect (spec §8)
    };
    this.ws.onerror = () => this.ws.close();
  }

  send(frame) {
    if (this.connected && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(frame));
    }
  }
}
```

- [ ] **Step 7: `apps/overlay-web/js/overlay-renderer.js`** (telestrator + panel + risk + pulse)

```javascript
import { regionCenter } from "./regions.js";

const STATUS_COLORS = {
  CALIBRATING: "#5b8def", CLEAR: "#28c76f", WATCH: "#ff9f43", FLAG: "#ea5455",
};

export class OverlayRenderer {
  constructor(canvas, panelEls) {
    this.canvas = canvas; this.ctx = canvas.getContext("2d");
    this.panel = panelEls; this.lastConsensus = null;
  }

  setConsensus(c) { this.lastConsensus = c; this._renderPanel(c); }

  // Called every animation frame with the latest landmarks for smooth telestrator anchoring.
  draw(landmarks) {
    const ctx = this.ctx, w = this.canvas.width, h = this.canvas.height;
    ctx.clearRect(0, 0, w, h);
    const c = this.lastConsensus;
    if (!c || !landmarks) return;
    const color = STATUS_COLORS[c.status] || "#888";
    for (const cue of c.active_cues) {
      const pt = regionCenter(cue.region, landmarks, w, h);
      if (!pt) continue;
      const radius = 26 + Math.min(40, Math.abs(cue.z) * 6);
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, radius, 0, Math.PI * 2);
      ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.globalAlpha = 0.9; ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = color; ctx.font = "12px monospace";
      ctx.fillText(cue.cue_id.split(".").pop(), pt.x + radius + 4, pt.y);
    }
    if (c.flag) this._redPulse();   // earned red pulse only on two-gate FLAG (spec §7)
  }

  _redPulse() {
    const t = (Date.now() % 1000) / 1000;
    this.ctx.strokeStyle = `rgba(234,84,85,${0.8 - t * 0.6})`;
    this.ctx.lineWidth = 8 + t * 10;
    this.ctx.strokeRect(4, 4, this.canvas.width - 8, this.canvas.height - 8);
  }

  _renderPanel(c) {
    const color = STATUS_COLORS[c.status] || "#888";
    this.panel.status.textContent = c.status;
    this.panel.status.style.color = color;
    this.panel.risk.style.width = `${Math.round(c.risk * 100)}%`;
    this.panel.risk.style.background = color;
    this.panel.agree.textContent = `${c.n_agree} of ${c.n_required} families agree`;
    this.panel.message.textContent = c.message || "";
    this.panel.voters.innerHTML = "";
    for (const f of c.families) {
      const li = document.createElement("li");
      const state = !f.wired ? "—" : f.fresh ? (f.vote ? "FLAG" : "fresh") : "stale";
      li.textContent = `${f.name.padEnd(11)} ${state}`;
      li.className = `voter ${f.wired ? "wired" : "unwired"} ${f.vote ? "voting" : ""}`;
      this.panel.voters.appendChild(li);
    }
  }
}
```

- [ ] **Step 8: `apps/overlay-web/js/main.js`** (bootstrap loop)

```javascript
import { WebcamSource } from "./capture.js";
import { MediaPipeExtractor } from "./mediapipe-extractor.js";
import { RppgSampler } from "./rppg-sampler.js";
import { WsClient } from "./ws-client.js";
import { OverlayRenderer } from "./overlay-renderer.js";

const video = document.getElementById("cam");
const canvas = document.getElementById("overlay");
const panel = {
  status: document.getElementById("status"),
  risk: document.getElementById("risk-fill"),
  agree: document.getElementById("agree"),
  message: document.getElementById("message"),
  voters: document.getElementById("voters"),
  toggle: document.getElementById("toggle"),
  body: document.getElementById("panel-body"),
};

const extractor = new MediaPipeExtractor();
const sampler = new RppgSampler();
const renderer = new OverlayRenderer(canvas, panel);
const wsUrl = `ws://${location.host}/ws`;
const ws = new WsClient(wsUrl, (c) => renderer.setConsensus(c),
  (s) => { if (s === "engine-offline") panel.message.textContent = "Engine offline — reconnecting…"; });

panel.toggle.addEventListener("click", () => panel.body.classList.toggle("collapsed"));

async function start() {
  try {
    const source = new WebcamSource(video);
    await source.start();
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    await extractor.init();
    ws.connect();
    requestAnimationFrame(loop);
  } catch (err) {
    panel.message.textContent = `Camera unavailable: ${err.message}`;  // graceful (spec §8)
  }
}

function loop() {
  const ts = Math.round(performance.now());
  const frame = extractor.extract(video, ts);
  if (frame.face_present && extractor.lastLandmarks) {
    frame.rppg = sampler.sample(video, extractor.lastLandmarks);
  }
  ws.send(frame);
  renderer.draw(extractor.lastLandmarks);
  requestAnimationFrame(loop);
}

start();
```

- [ ] **Step 9: `apps/overlay-web/css/overlay.css`**

```css
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; background: #0b0f14; color: #e6edf3; font-family: system-ui, sans-serif; }
header { padding: 10px 16px; font-size: 14px; letter-spacing: .12em; color: #7d8da3; text-transform: uppercase; }
.stage { position: relative; width: 640px; max-width: 100vw; margin: 0 auto; }
#cam { width: 100%; transform: scaleX(-1); border-radius: 8px; display: block; }
#overlay { position: absolute; inset: 0; transform: scaleX(-1); pointer-events: none; }
.panel { width: 640px; max-width: 100vw; margin: 12px auto; background: #11161d; border: 1px solid #1f2733; border-radius: 8px; }
.panel-head { display: flex; align-items: center; justify-content: space-between; padding: 10px 14px; cursor: default; }
#status { font-weight: 700; font-size: 18px; letter-spacing: .08em; }
.risk-meter { height: 8px; background: #1f2733; border-radius: 4px; overflow: hidden; margin: 0 14px; }
#risk-fill { height: 100%; width: 0; transition: width .15s ease, background .15s ease; }
#agree { padding: 8px 14px; font-size: 13px; color: #9fb0c3; }
#message { padding: 0 14px 8px; font-size: 12px; color: #ff9f43; min-height: 16px; }
#toggle { background: none; border: 1px solid #2a3645; color: #9fb0c3; border-radius: 6px; padding: 4px 10px; cursor: pointer; }
#panel-body.collapsed { display: none; }
#voters { list-style: none; margin: 0; padding: 8px 14px 14px; font-family: monospace; font-size: 13px; }
.voter { padding: 3px 0; white-space: pre; }
.voter.unwired { color: #5b6675; }
.voter.wired { color: #9fb0c3; }
.voter.voting { color: #ea5455; font-weight: 700; }
```

- [ ] **Step 10: Replace `apps/overlay-web/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Live Consensus Overlay</title>
  <link rel="stylesheet" href="/static/css/overlay.css" />
</head>
<body>
  <header>Blitz Engine · Live Consensus Overlay · webcam · local-only</header>
  <div class="stage">
    <video id="cam" playsinline muted></video>
    <canvas id="overlay"></canvas>
  </div>
  <section class="panel">
    <div class="panel-head">
      <span id="status">CONNECTING</span>
      <button id="toggle">details</button>
    </div>
    <div class="risk-meter"><div id="risk-fill"></div></div>
    <div id="agree">— of 2 families agree</div>
    <div id="message"></div>
    <div id="panel-body">
      <ul id="voters"></ul>
    </div>
  </section>
  <script type="module" src="/static/js/main.js"></script>
</body>
</html>
```

- [ ] **Step 11: Verify the server serves everything and the WS handshake works**

Run: `python -m pytest tests/overlay/test_server.py -v`
Expected: PASS (index now contains the title; static mount serves `/static/...`).

Then start it and confirm the page loads (manual, real webcam):
Run: `python -m blitz_overlay`
Expected: console prints the URL, browser opens, camera permission prompt appears, video shows mirrored, status starts `CALIBRATING`, telestrator circles appear on eyes/brow/mouth/jaw as you move, panel toggles collapsed/expanded, audio + linguistic voters show "—". (A FLAG requires the baseline window to fill, then two-family convergence.)

- [ ] **Step 12: Commit**

```bash
git add apps/overlay-web/
git commit -m "feat(overlay): browser app — webcam, MediaPipe, rPPG, ws-client, telestrator + panel"
```

---

### Task 16: Ruff clean + full suite + README + RESEARCH blocker updates

**Files:**
- Modify: `RESEARCH.md` (blockers 1 & 2 → resolved)
- Create: `docs/OVERLAY_README.md` (or a top-level `README` section)
- Modify: `progress.json`

- [ ] **Step 1: Lint everything**

Run: `python -m ruff check . --fix && python -m ruff check .`
Expected: "All checks passed!" Fix any residual import-order/unused issues.

- [ ] **Step 2: Run the whole suite**

Run: `python -m pytest -q`
Expected: all tests pass (tooling, schemas, weights, rolling baseline, cue base, visual cues, rppg, physio cue, family fusion, consensus, logger, pipeline, replay, server) plus the pre-existing `tests/test_text_mvp.py`.

- [ ] **Step 3: Resolve the two blockers in `planning/RESEARCH.md`**

Find the Blocker 1 and Blocker 2 sections and append resolutions (read the file first to match exact headings):
- Blocker 1 (CrisperWhisper): "RESOLVED (2026-06-16): WhisperX (BSD-2) is the transcription path for later audio stages; also a RAM decision per EXECUTION_ARCHITECTURE C.1. Stage 1 wires no live transcription."
- Blocker 2 (AU28 jaw tension): "RESOLVED (2026-06-16): implemented as `visual.jaw_tension` via MediaPipe landmark-distance ratio (gonial 172↔397 / outer-eye-corner span 33↔263) in `blitz_overlay/cues/visual.py`."

- [ ] **Step 4: Create `docs/OVERLAY_README.md`** (the one-command quickstart)

````markdown
# Live Consensus Overlay — Quickstart (Stage 1, webcam-only)

Real-time "AI-vision" deception **overlay** that runs entirely on your machine. The browser
captures your webcam and runs MediaPipe + rPPG; a local Python engine detects visual cues +
heart rate, fuses them by family with a two-gate consensus, and draws a telestrator + consensus
panel. **Raw video never leaves your device** — only tiny feature vectors reach the engine over
a localhost WebSocket. GitHub stores the code and runs CI; it never runs the live app.

Honest framing: statuses are **CALIBRATING → CLEAR → WATCH → FLAG**, never a binary "LIE".
A red pulse fires *only* on a two-gate FLAG (≥2 independent families agree AND combined risk ≥ 0.65).

## One command

```bash
pip install -e .          # first time only (Python 3.10+)
blitz-overlay             # starts engine + browser host, opens http://127.0.0.1:8000
```

(or `python -m blitz_overlay`). Allow camera access when prompted.

- First ~90s = **CALIBRATING** (builds your rolling baseline; no flags permitted).
- Then **CLEAR/WATCH**; a **FLAG** needs both the Visual family and the Physio (heart-rate)
  family to agree — that's why rPPG is required in Stage 1.
- Audio and Linguistic voters show "—" (not wired in Stage 1).

## Config (optional)

Copy `.env.example` to `.env` to change the port, gate threshold, or baseline length:

```bash
cp .env.example .env
```

No API keys are needed — all fusion is local math.

## Tests

```bash
pip install -e ".[dev]"
python -m ruff check .
python -m pytest -q
```

Includes a deterministic replay test that drives the engine to a two-gate FLAG from a
synthetic feature stream — no camera required.

## Privacy

The browser samples only blendshape coefficients, head pose, a few landmark-derived scalars,
and mean ROI colors for rPPG. The engine logs only derived decisions (status, posterior,
per-family contributions, cue z-scores) to `logs/` — never raw biometric. `logs/` and `.env`
are gitignored.
````

- [ ] **Step 5: Link the quickstart from the main README and INDEX**

In `README.md` add near the top: `**Run the Live Consensus Overlay:** see docs/OVERLAY_README.md — one command: \`blitz-overlay\`.`
In `planning/INDEX.md` line 19, replace the placeholder with: `5. docs/superpowers/plans/2026-06-16-live-consensus-overlay.md + docs/OVERLAY_README.md — the implementation plan + quickstart.`

- [ ] **Step 6: Update `progress.json`**

Add an overlay entry to `completed` and update `summary`/`phase` to reflect the walking skeleton being built (read the file first; append rather than rewrite history).

- [ ] **Step 7: Final verification + commit**

Run: `python -m ruff check . && python -m pytest -q`
Expected: clean + all green.

```bash
git add planning/RESEARCH.md docs/OVERLAY_README.md README.md planning/INDEX.md progress.json
git commit -m "docs: overlay quickstart, resolve blockers 1&2, link plan + update progress"
```

---

## Self-Review (run against the spec)

**Spec coverage check:**
- §3 architecture (browser capture + Python engine, WS, feature vectors only) → Tasks 14 (server/ws), 15 (browser). ✅
- §4 components: capture (T15 capture.js), mediapipe-extractor (T15), rppg-sampler (T15), ws-client (T15), overlay-renderer (T15); ws-server (T14), cue-detectors (T5,6,8), baseline rolling (T4), family fusion (T9), consensus-builder (T10), prediction-logger (T11); shared schema (T2). ✅
- §5 cue→landmark mapping incl. jaw tension/AU28 (Blocker 2) → T6 + T16 RESEARCH update. ✅
- §6 consensus: family voters, independence/decorrelation (T9), two-gate, freshness, statuses (T10). ✅
- §7 framing: never binary LIE, red pulse only on FLAG, families always visible, collapsed/expanded (T10, T15). ✅
- §8 graceful degradation: no camera (T15 start catch), no face (T12 pause + message), low confidence→quality (T5), WS drop+reconnect (T15 ws-client), CALIBRATING no flags (T4,10), <2 families capped at WATCH (T10), audio/linguistic "—" (T10). ✅
- §9 swappable capture adapter (T15 WebcamSource interface). ✅
- §10 testing: unit per cue (T6,8), fusion/baseline math (T4,9), deterministic replay (T13), prediction logging (T11). ✅
- §11 first-build scope: 5 visual cues + rPPG, rolling baseline, family fusion, two-gate, consensus payload, prediction log, browser overlay, audio/linguistic "not wired", webcam only, one command. ✅
- READINESS items in scope: #1 manifest/repro (T1), #5 walking skeleton (whole plan), #6 math tests (T4,9), #7 rolling baseline (T4), #8 science weights + logging (T3,11), #11 independence fix (T9), #12 graceful degradation (T10,12,15), #15 privacy/no raw biometric (T2 contract, T11 logger), #16 secrets/.env (T1,14), #17 reproducibility/replay (T13), #18 weight-set version (T3,11), #21 CI (T1). ✅

**Placeholder scan:** every code step contains full code; no TBD/TODO. Two steps contain explicit refactor notes (T5 Step 4→5, T12 Step 3→4) that the engineer applies inline — the final code is shown.

**Type consistency:** `FeatureFrame.from_dict`/fields, `Consensus.to_dict`, `FamilyVote`, `ActiveCue`, `Region` used identically across schemas/consensus/logger/server. `CueDetector.update(frame, baseline, value=None)` signature is consistent between base, visual, physio, and pipeline call sites. `fuse_by_family`/`two_gate`/`family_of`/`FAMILY_THRESHOLD` names match between fusion module, family-fusion test, and consensus builder. `RollingBaseline(baseline_seconds, window_seconds)`, `.update(features, ts_ms)`, `.normalize(cue_id, value)`, `.is_calibrating`, `.mode` consistent across baseline, detectors, pipeline. SCHEMA_VERSION mirrored in `schema.js`.

**Known soft spots flagged for execution:** the replay test (T13) and the two-gate fusion test (T9) assert *rules*, not tuned magic numbers; both steps include explicit "if it doesn't reach FLAG, raise the synthetic stimulus" guidance so weights stay science-driven and untouched.
