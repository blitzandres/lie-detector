# Parallel Cue Verifier + Synchrony Bell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Layer a live "Parallel Cue Verifier" checklist + a synchrony-driven, earned bell on top of the existing cue engine — without modifying any detector, weight, or fusion code.

**Architecture:** New aggregation modules (`SynchronyDetector`, `BellController`) read the per-cue z-scores and fused posterior the pipeline already computes, detect temporal co-firing bursts (≥K lit cues across ≥2 families), and ring a debounced bell when a burst + a posterior risk-floor hold ~1.5s. Additive schema fields carry `cue_rows`/`convergence`/`bell` to the browser, which renders a live checklist, plays a WebAudio chime, shows a trust meter, and sends a sensitivity operating-point back on each frame.

**Tech Stack:** Python 3.14 (run via `python3`, no venv), pytest, ruff; vanilla ES-module browser app (Canvas 2D, WebAudio), FastAPI WebSocket.

**Reference spec:** `docs/superpowers/specs/2026-06-16-parallel-cue-verifier-design.md`. Verify after every task: `python3 -m pytest -q` and `python3 -m ruff check .` stay green. Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

**ENGINE-SACRED RULE:** Do NOT edit `blitz_overlay/cues/*`, `blitz_overlay/weights.py`, `core/fusion/*`, or `core/calibration/*`. Only read their outputs.

---

## File Structure

**Python (new aggregation layer):**
- Create `blitz_overlay/synchrony.py` — `SynchronyDetector` (rolling lit-window → convergence burst).
- Create `blitz_overlay/bell.py` — `BellController` (burst + risk-floor, debounced) + `map_sensitivity`.
- Modify `blitz_overlay/schemas.py` — `CueRow` dataclass; `config` on `FeatureFrame`; `cue_rows`/`convergence`/`bell` on `Consensus`.
- Modify `blitz_overlay/consensus.py` — accept + attach `cue_rows`/`convergence`/`bell`.
- Modify `blitz_overlay/pipeline.py` — compute directed-z per cue, build cue_rows, drive synchrony + bell, apply `frame.config` sensitivity.
- Modify `blitz_overlay/logger.py` — append a bell record when the bell fires.

**Browser:**
- Create `apps/overlay-web/js/cue-verifier.js` — live checklist + convergence counter + verdict.
- Create `apps/overlay-web/js/bell.js` — WebAudio chime + trust meter.
- Modify `apps/overlay-web/js/main.js` — wire both, sensitivity slider, attach `frame.config`.
- Modify `apps/overlay-web/index.html` + `apps/overlay-web/css/overlay.css` — checklist/verdict/slider/trust DOM + styles.

**Tests:**
- Create `tests/overlay/test_synchrony.py`, `tests/overlay/test_bell.py`, `tests/overlay/test_cue_verifier_pipeline.py`.
- Modify `tests/overlay/test_schemas.py`, `tests/overlay/test_consensus.py`.

---

## Task 1: SynchronyDetector

**Files:**
- Create: `blitz_overlay/synchrony.py`
- Test: `tests/overlay/test_synchrony.py`

- [ ] **Step 1: Write the failing test** — create `tests/overlay/test_synchrony.py`:

```python
"""Tests for SynchronyDetector — temporal co-firing burst over the existing per-cue z-scores."""
from blitz_overlay.synchrony import SynchronyDetector


def test_burst_requires_k_cues_and_two_families():
    d = SynchronyDetector(window_ms=1000, lit_z=2.0, k=3)
    # 3 lit cues but only ONE family -> not a burst (correlated, not convergent)
    snap = d.update(0, [("visual.gaze_aversion", "visual", 3.0),
                        ("visual.blink_rate", "visual", 2.5),
                        ("visual.lip_press", "visual", 2.2)])
    assert snap["n_lit"] == 3
    assert snap["n_families"] == 1
    assert snap["burst"] is False


def test_burst_true_with_three_cues_across_two_families():
    d = SynchronyDetector(window_ms=1000, lit_z=2.0, k=3)
    snap = d.update(0, [("visual.gaze_aversion", "visual", 3.0),
                        ("visual.blink_rate", "visual", 2.5),
                        ("physio.heart_rate", "physio", 2.4)])
    assert snap["n_lit"] == 3
    assert snap["n_families"] == 2
    assert snap["burst"] is True
    assert set(snap["families_lit"]) == {"visual", "physio"}


def test_below_lit_z_is_not_lit():
    d = SynchronyDetector(window_ms=1000, lit_z=2.0, k=3)
    snap = d.update(0, [("visual.gaze_aversion", "visual", 1.9),
                        ("physio.heart_rate", "physio", 5.0)])
    assert snap["n_lit"] == 1
    assert snap["burst"] is False


def test_rolling_window_expires_stale_lit_cues():
    d = SynchronyDetector(window_ms=1000, lit_z=2.0, k=3)
    d.update(0, [("visual.gaze_aversion", "visual", 3.0),
                 ("visual.blink_rate", "visual", 3.0)])
    # 1500ms later only physio lights; the two visual cues are now stale (>1000ms old)
    snap = d.update(1500, [("physio.heart_rate", "physio", 3.0)])
    assert snap["n_lit"] == 1
    assert snap["n_families"] == 1
    assert snap["burst"] is False


def test_recent_lit_cues_within_window_aggregate():
    d = SynchronyDetector(window_ms=1000, lit_z=2.0, k=3)
    d.update(0, [("visual.gaze_aversion", "visual", 3.0)])
    d.update(300, [("audio.tremor", "audio", 3.0)])
    snap = d.update(600, [("physio.heart_rate", "physio", 3.0)])
    # all three lit within the 1000ms window -> burst across 3 families
    assert snap["n_lit"] == 3
    assert snap["n_families"] == 3
    assert snap["burst"] is True


def test_set_params_updates_thresholds():
    d = SynchronyDetector(window_ms=1000, lit_z=2.0, k=3)
    d.set_params(lit_z=1.5, k=2)
    snap = d.update(0, [("visual.gaze_aversion", "visual", 1.6),
                        ("physio.heart_rate", "physio", 1.6)])
    assert snap["n_lit"] == 2
    assert snap["burst"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/overlay/test_synchrony.py -q`
Expected: FAIL — `ModuleNotFoundError: blitz_overlay.synchrony`.

- [ ] **Step 3: Implement** — create `blitz_overlay/synchrony.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/overlay/test_synchrony.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add blitz_overlay/synchrony.py tests/overlay/test_synchrony.py
git commit -m "feat(overlay): SynchronyDetector — temporal co-firing burst over per-cue z

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: BellController + sensitivity map

**Files:**
- Create: `blitz_overlay/bell.py`
- Test: `tests/overlay/test_bell.py`

- [ ] **Step 1: Write the failing test** — create `tests/overlay/test_bell.py`:

```python
"""Tests for BellController (earned, debounced, sustained) + map_sensitivity."""
from blitz_overlay.bell import BellController, map_sensitivity


def _burst(ok=True, families=("visual", "physio"), cues=("a", "b", "c")):
    return {"burst": ok, "lit_cue_ids": list(cues), "families_lit": list(families)}


def test_silent_until_sustained():
    b = BellController(hold_ms=1500, risk_floor=0.65)
    s = b.update(0, _burst(), posterior=0.8)
    assert s["ringing"] is False and s["just_rang"] is False
    s = b.update(1000, _burst(), posterior=0.8)   # only 1.0s held
    assert s["just_rang"] is False
    s = b.update(1600, _burst(), posterior=0.8)   # 1.6s held -> ring
    assert s["just_rang"] is True
    assert s["ringing"] is True
    assert s["label"] == "strong deception-pattern convergence"


def test_just_rang_fires_once_per_episode():
    b = BellController(hold_ms=1500, risk_floor=0.65)
    b.update(0, _burst(), posterior=0.8)
    b.update(1600, _burst(), posterior=0.8)       # rings
    s = b.update(1700, _burst(), posterior=0.8)   # still held -> no second edge
    assert s["just_rang"] is False
    assert s["ringing"] is True


def test_re_arms_after_condition_drops():
    b = BellController(hold_ms=1500, risk_floor=0.65)
    b.update(0, _burst(), posterior=0.8)
    b.update(1600, _burst(), posterior=0.8)       # rings
    s = b.update(1700, _burst(ok=False), posterior=0.8)  # burst drops -> reset
    assert s["ringing"] is False and s["just_rang"] is False
    b.update(1800, _burst(), posterior=0.8)
    s = b.update(3400, _burst(), posterior=0.8)   # held 1.6s again -> rings again
    assert s["just_rang"] is True


def test_risk_floor_blocks_bell():
    b = BellController(hold_ms=1500, risk_floor=0.65)
    b.update(0, _burst(), posterior=0.5)          # below risk_floor
    s = b.update(2000, _burst(), posterior=0.5)
    assert s["just_rang"] is False and s["ringing"] is False


def test_record_on_ring():
    b = BellController(hold_ms=1500, risk_floor=0.65)
    b.update(0, _burst(cues=("x", "y", "z"), families=("visual", "audio")), posterior=0.9)
    s = b.update(1600, _burst(cues=("x", "y", "z"), families=("visual", "audio")), posterior=0.9)
    assert s["just_rang"] is True
    rec = s["record"]
    assert rec["cue_ids"] == ["x", "y", "z"]
    assert rec["families"] == ["visual", "audio"]
    assert rec["risk"] == 0.9
    assert rec["ts"] == 1600


def test_map_sensitivity_endpoints():
    lo = map_sensitivity(0.0)
    assert lo == {"k": 3, "lit_z": 2.0, "risk_floor": 0.65}
    hi = map_sensitivity(1.0)
    assert hi == {"k": 2, "lit_z": 1.5, "risk_floor": 0.45}


def test_map_sensitivity_clamps():
    assert map_sensitivity(-5.0) == map_sensitivity(0.0)
    assert map_sensitivity(5.0) == map_sensitivity(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/overlay/test_bell.py -q`
Expected: FAIL — `ModuleNotFoundError: blitz_overlay.bell`.

- [ ] **Step 3: Implement** — create `blitz_overlay/bell.py`:

```python
"""BellController — the earned, sustained, synchrony-gated alarm.

Reads (read-only) the SynchronyDetector burst + the fused posterior. Rings only when a
burst AND posterior >= risk_floor hold continuously for hold_ms (debounced). Honest label,
never "lie". The operating point (k / lit_z / risk_floor) is moved by the sensitivity slider
via map_sensitivity — never the science cue weights.
"""
from __future__ import annotations

BELL_LABEL = "strong deception-pattern convergence"


class BellController:
    def __init__(self, hold_ms: int = 1500, risk_floor: float = 0.65):
        self.hold_ms = hold_ms
        self.risk_floor = risk_floor
        self._since_ts: int | None = None  # when the current satisfied streak began
        self._ringing = False

    def set_params(self, *, hold_ms: int | None = None, risk_floor: float | None = None) -> None:
        if hold_ms is not None:
            self.hold_ms = hold_ms
        if risk_floor is not None:
            self.risk_floor = risk_floor

    def update(self, ts: int, convergence: dict, posterior: float) -> dict:
        condition = bool(convergence.get("burst")) and posterior >= self.risk_floor
        just_rang = False
        if condition:
            if self._since_ts is None:
                self._since_ts = ts
            if ts - self._since_ts >= self.hold_ms and not self._ringing:
                self._ringing = True
                just_rang = True
        else:
            self._since_ts = None
            self._ringing = False

        sustained_ms = (ts - self._since_ts) if self._since_ts is not None else 0
        record = None
        if just_rang:
            record = {
                "ts": ts,
                "cue_ids": list(convergence.get("lit_cue_ids", [])),
                "families": list(convergence.get("families_lit", [])),
                "risk": round(posterior, 4),
            }
        return {
            "ringing": self._ringing,
            "just_rang": just_rang,
            "sustained_ms": sustained_ms,
            "label": BELL_LABEL,
            "record": record,
        }


def map_sensitivity(sensitivity: float) -> dict:
    """Map a 0..1 slider to the bell operating point. 0 = conservative, 1 = max (more alarms).

    Never touches science weights or the >=2-families requirement.
    """
    s = max(0.0, min(1.0, sensitivity))
    return {
        "k": int(round(3 - s)),               # 3 -> 2
        "lit_z": round(2.0 - 0.5 * s, 3),     # 2.0 -> 1.5
        "risk_floor": round(0.65 - 0.20 * s, 3),  # 0.65 -> 0.45
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/overlay/test_bell.py -q`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add blitz_overlay/bell.py tests/overlay/test_bell.py
git commit -m "feat(overlay): BellController + sensitivity operating-point map

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Schema — CueRow, config, and Consensus fields

**Files:**
- Modify: `blitz_overlay/schemas.py`
- Test: `tests/overlay/test_schemas.py`

- [ ] **Step 1: Write the failing test** — append to `tests/overlay/test_schemas.py`:

```python
def test_feature_frame_carries_config_block():
    from blitz_overlay.schemas import FeatureFrame
    frame = FeatureFrame.from_dict({"ts": 1, "face_present": True,
                                    "config": {"sensitivity": 0.7}})
    assert frame.config == {"sensitivity": 0.7}
    assert FeatureFrame.from_dict({"ts": 1}).config is None


def test_cue_row_to_dict():
    from blitz_overlay.schemas import CueRow
    row = CueRow(cue_id="visual.gaze_aversion", family="visual", region="eyes",
                 label="gaze_aversion", z=3.21, lit=True, online=True)
    d = row.to_dict()
    assert d == {"cue_id": "visual.gaze_aversion", "family": "visual", "region": "eyes",
                 "label": "gaze_aversion", "z": 3.21, "lit": True, "online": True}


def test_consensus_to_dict_includes_verifier_fields():
    from blitz_overlay.schemas import Consensus, CueRow, SCHEMA_VERSION
    c = Consensus(schema_version=SCHEMA_VERSION, ts=0, status="CLEAR", risk=0.1,
                  flag=False, n_agree=0, n_required=2,
                  cue_rows=[CueRow("visual.gaze_aversion", "visual", "eyes",
                                   "gaze_aversion", 0.0, False, True)],
                  convergence={"n_lit": 0, "n_families": 0, "burst": False},
                  bell={"ringing": False, "just_rang": False})
    d = c.to_dict()
    assert d["cue_rows"][0]["cue_id"] == "visual.gaze_aversion"
    assert d["convergence"]["burst"] is False
    assert d["bell"]["ringing"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/overlay/test_schemas.py::test_cue_row_to_dict -v`
Expected: FAIL — `ImportError: cannot import name 'CueRow'`.

- [ ] **Step 3: Implement** — in `blitz_overlay/schemas.py`:

(a) Add `config` to `FeatureFrame` (after the `transcript` field):

```python
    config: dict | None = None                # {"sensitivity": 0..1} or None
```

and in `FeatureFrame.from_dict` (after the `transcript=...` line):

```python
            config=(dict(d["config"]) if d.get("config") else None),
```

(b) Add the `CueRow` dataclass (place it just above `@dataclass class Consensus`):

```python
@dataclass
class CueRow:
    """One row in the live Parallel Cue Verifier checklist."""

    cue_id: str
    family: str
    region: str
    label: str
    z: float
    lit: bool
    online: bool

    def to_dict(self) -> dict:
        return {
            "cue_id": self.cue_id,
            "family": self.family,
            "region": self.region,
            "label": self.label,
            "z": round(self.z, 3),
            "lit": self.lit,
            "online": self.online,
        }
```

(c) Add three fields to `Consensus` (after `active_cues`):

```python
    cue_rows: list[CueRow] = field(default_factory=list)
    convergence: dict = field(default_factory=dict)
    bell: dict = field(default_factory=dict)
```

(d) In `Consensus.to_dict`, add these keys to the returned dict (before `"message"`):

```python
            "cue_rows": [r.to_dict() for r in self.cue_rows],
            "convergence": self.convergence,
            "bell": self.bell,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/overlay/test_schemas.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add blitz_overlay/schemas.py tests/overlay/test_schemas.py
git commit -m "feat(overlay): schema — CueRow + config + verifier payload fields

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Consensus builder attaches verifier payload

**Files:**
- Modify: `blitz_overlay/consensus.py`
- Test: `tests/overlay/test_consensus.py`

- [ ] **Step 1: Write the failing test** — append to `tests/overlay/test_consensus.py`:

```python
def test_build_attaches_cue_rows_convergence_bell():
    from blitz_overlay.schemas import CueRow
    cb = ConsensusBuilder()
    rows = [CueRow("visual.gaze_aversion", "visual", "eyes", "gaze_aversion", 3.0, True, True)]
    conv = {"n_lit": 1, "n_families": 1, "burst": False, "lit_cue_ids": ["visual.gaze_aversion"]}
    bell = {"ringing": False, "just_rang": False, "label": "strong deception-pattern convergence"}
    out = cb.build(cues=[], calibrating=False, ts=1000, regions={},
                   cue_rows=rows, convergence=conv, bell=bell)
    assert out.cue_rows == rows
    assert out.convergence == conv
    assert out.bell == bell


def test_build_defaults_verifier_fields_empty():
    cb = ConsensusBuilder()
    out = cb.build(cues=[], calibrating=False, ts=1000, regions={})
    assert out.cue_rows == []
    assert out.convergence == {}
    assert out.bell == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/overlay/test_consensus.py::test_build_attaches_cue_rows_convergence_bell -v`
Expected: FAIL — `build()` got an unexpected keyword argument `cue_rows`.

- [ ] **Step 3: Implement** — in `blitz_overlay/consensus.py`, extend `ConsensusBuilder.build`:

(a) Add the three parameters to the signature (after `family_activity`):

```python
    def build(self, cues, calibrating: bool, ts: int, regions: dict[str, str],
              message: str = "",
              online_families: set[str] | None = None,
              family_activity: dict[str, float] | None = None,
              cue_rows=None, convergence=None, bell=None) -> Consensus:
```

(b) In the `return Consensus(...)` call, add the three fields (after `active_cues=active,`):

```python
            cue_rows=cue_rows or [],
            convergence=convergence or {},
            bell=bell or {},
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/overlay/test_consensus.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add blitz_overlay/consensus.py tests/overlay/test_consensus.py
git commit -m "feat(overlay): consensus attaches cue_rows/convergence/bell payload

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Logger records bell events

**Files:**
- Modify: `blitz_overlay/logger.py`
- Test: `tests/overlay/test_logger.py`

- [ ] **Step 1: Write the failing test** — append to `tests/overlay/test_logger.py` (import `json`/`Path` at top if not present):

```python
def test_logger_writes_bell_record(tmp_path):
    import json
    from blitz_overlay.logger import PredictionLogger
    from blitz_overlay.schemas import Consensus, SCHEMA_VERSION

    logger = PredictionLogger("sess-bell", log_dir=tmp_path)
    c = Consensus(schema_version=SCHEMA_VERSION, ts=1600, status="FLAG", risk=0.9,
                  flag=True, n_agree=2, n_required=2,
                  bell={"just_rang": True, "ringing": True,
                        "record": {"ts": 1600, "cue_ids": ["a", "b", "c"],
                                   "families": ["visual", "physio"], "risk": 0.9}})
    logger.log(c, baseline_mode="rolling")
    line = json.loads(tmp_path.joinpath("predictions-sess-bell.jsonl").read_text().strip())
    assert line["bell_record"] == {"ts": 1600, "cue_ids": ["a", "b", "c"],
                                   "families": ["visual", "physio"], "risk": 0.9}


def test_logger_bell_record_null_when_silent(tmp_path):
    import json
    from blitz_overlay.logger import PredictionLogger
    from blitz_overlay.schemas import Consensus, SCHEMA_VERSION

    logger = PredictionLogger("sess-quiet", log_dir=tmp_path)
    c = Consensus(schema_version=SCHEMA_VERSION, ts=1, status="CLEAR", risk=0.1,
                  flag=False, n_agree=0, n_required=2,
                  bell={"just_rang": False, "ringing": False})
    logger.log(c, baseline_mode="rolling")
    line = json.loads(tmp_path.joinpath("predictions-sess-quiet.jsonl").read_text().strip())
    assert line["bell_record"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/overlay/test_logger.py::test_logger_writes_bell_record -v`
Expected: FAIL — `KeyError: 'bell_record'`.

- [ ] **Step 3: Implement** — in `blitz_overlay/logger.py`, inside `PredictionLogger.log`, add one key to the `record` dict (after `"flag": consensus.flag,`):

```python
            "bell_record": (consensus.bell or {}).get("record") if (consensus.bell or {}).get("just_rang") else None,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/overlay/test_logger.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add blitz_overlay/logger.py tests/overlay/test_logger.py
git commit -m "feat(overlay): log bell records for the honest trust log

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Pipeline wires synchrony + bell + cue_rows + sensitivity

**Files:**
- Modify: `blitz_overlay/pipeline.py`
- Test: `tests/overlay/test_cue_verifier_pipeline.py`

This is the integration task. It (a) computes a directed-z per measured cue, (b) builds a `CueRow`
for every registered detector, (c) drives `SynchronyDetector` + `BellController` from the lit set and
the fused posterior, (d) applies `frame.config["sensitivity"]` via `map_sensitivity`, and (e) passes
all of it into `consensus.build`.

- [ ] **Step 1: Write the failing test** — create `tests/overlay/test_cue_verifier_pipeline.py`:

```python
"""End-to-end: the existing replay (visual gaze + physio HR) rings the bell at max
sensitivity (k=2) but stays silent at the conservative default (k=3, only 2 cues lit)."""
from blitz_overlay.pipeline import OverlaySession
from tests.overlay.fixtures.replay_session import replay_frames


def _run(sensitivity):
    s = OverlaySession(baseline_seconds=40)  # replay calibrates for 40s then stresses
    bell_rang = False
    rows_seen = 0
    for raw in replay_frames():
        raw = {**raw, "config": {"sensitivity": sensitivity}}
        out = s.process(raw)
        rows_seen = max(rows_seen, len(out.cue_rows))
        if out.bell.get("just_rang"):
            bell_rang = True
    return bell_rang, rows_seen


def test_cue_rows_cover_all_registered_detectors():
    s = OverlaySession(baseline_seconds=0)
    out = s.process(next(iter(replay_frames())))
    ids = {r.cue_id for r in out.cue_rows}
    # every registered detector appears as a row (visual + audio + linguistic + physio)
    assert {d.cue_id for d in s.detectors}.issubset(ids)


def test_bell_rings_at_max_sensitivity():
    rang, rows = _run(sensitivity=1.0)
    assert rang is True
    assert rows >= 8  # all detectors present as rows


def test_bell_silent_at_conservative_default():
    rang, _ = _run(sensitivity=0.0)
    # only gaze + heart_rate lit (2 cues) -> below k=3 -> no burst -> no bell
    assert rang is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/overlay/test_cue_verifier_pipeline.py -q`
Expected: FAIL — `Consensus` has no populated `cue_rows` / `bell` (AttributeError or empty assertions).

- [ ] **Step 3: Implement** — edit `blitz_overlay/pipeline.py`:

(a) Add imports (after the existing `from blitz_overlay.*` imports):

```python
from blitz_overlay.bell import BellController, map_sensitivity
from blitz_overlay.synchrony import SynchronyDetector
from blitz_overlay.schemas import CueRow
from core.fusion.bayesian_fusion import fuse_by_family
```

(Keep the existing `from blitz_overlay.schemas import Consensus, FeatureFrame` line; add `CueRow`
there instead if you prefer a single import — either is fine as long as ruff passes.)

(b) In `__init__`, after `self._last_transcript_seq = None`, instantiate the new layer:

```python
        self.synchrony = SynchronyDetector()
        self.bell = BellController()
```

(c) Add a small private helper method to the class (place after `__init__`):

```python
    def _apply_sensitivity(self, frame: FeatureFrame) -> None:
        if not frame.config:
            return
        s = frame.config.get("sensitivity")
        if s is None:
            return
        params = map_sensitivity(float(s))
        self.synchrony.set_params(lit_z=params["lit_z"], k=params["k"])
        self.bell.set_params(risk_floor=params["risk_floor"])
```

(d) Add a helper that builds a `CueRow` for every registered detector given the directed-z map:

```python
    def _build_cue_rows(self, directed_z: dict[str, float], measured: set[str]) -> list[CueRow]:
        rows = []
        for det in self.detectors:
            z = directed_z.get(det.cue_id, 0.0)
            rows.append(CueRow(
                cue_id=det.cue_id, family=det.family, region=det.region,
                label=det.cue_id.split(".")[-1],
                z=z, lit=z >= self.synchrony.lit_z, online=det.cue_id in measured,
            ))
        return rows
```

(e) At the **top of `process`**, after `frame = FeatureFrame.from_dict(raw)` and BEFORE the transcript
gate, apply sensitivity:

```python
        self._apply_sensitivity(frame)
```

(f) In the **face-absent branch**, keep synchrony/bell state coherent and attach empty rows. Replace
the existing face-absent block body with:

```python
        if not frame.face_present:
            convergence = self.synchrony.update(frame.ts, [])
            bell = self.bell.update(frame.ts, convergence, 0.0)
            out = self.consensus.build(
                cues=[], calibrating=self.baseline.is_calibrating, ts=frame.ts,
                regions=self.regions, message="No subject detected — cues paused.",
                cue_rows=self._build_cue_rows({}, set()),
                convergence=convergence, bell=bell)
            self._last_consensus = out
            self.logger.log(out, baseline_mode=self.baseline.mode)
            return out
```

(g) In the main path, change the liveness loop to also capture a **directed** z per measured cue.
Replace the existing liveness loop (the `for det in self.detectors:` block that builds
`online_families`/`family_activity`) with:

```python
        online_families: set[str] = set()
        family_activity: dict[str, float] = {}
        directed_z: dict[str, float] = {}
        for det in self.detectors:
            if det.cue_id not in measurements:
                continue
            fam = det.family
            online_families.add(fam)
            raw_z = self.baseline.normalize(det.cue_id, measurements[det.cue_id])
            directed_z[det.cue_id] = raw_z * det.direction
            level = max(0.0, min(1.0, abs(raw_z) / 6.0))
            family_activity[fam] = max(family_activity.get(fam, 0.0), level)
```

(h) After the `cues = [...]` event-collection loop and BEFORE the `out = self.consensus.build(...)`
call, compute synchrony + bell:

```python
        cue_levels = [(cid, self._family_of_cue(cid), z) for cid, z in directed_z.items()]
        convergence = self.synchrony.update(frame.ts, cue_levels)
        posterior = 0.0 if self.baseline.is_calibrating else fuse_by_family(cues)["posterior"]
        bell = self.bell.update(frame.ts, convergence, posterior)
        cue_rows = self._build_cue_rows(directed_z, set(measurements.keys()))
```

(i) Add the family lookup helper (place after `_build_cue_rows`):

```python
    def _family_of_cue(self, cue_id: str) -> str:
        return self._cue_family.get(cue_id, "visual")
```

and build `self._cue_family` once in `__init__` (after `self.regions = {...}`):

```python
        self._cue_family = {d.cue_id: d.family for d in self.detectors}
```

(j) Update the main-path `out = self.consensus.build(...)` call to pass the new payload:

```python
        out = self.consensus.build(
            cues=cues, calibrating=self.baseline.is_calibrating, ts=frame.ts,
            regions=self.regions,
            online_families=online_families, family_activity=family_activity,
            cue_rows=cue_rows, convergence=convergence, bell=bell)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/overlay/test_cue_verifier_pipeline.py -q`
Expected: PASS (3 tests). If `test_bell_rings_at_max_sensitivity` fails, confirm the replay's stress
phase drives both `visual.gaze_aversion` and `physio.heart_rate` directed-z ≥ 1.5 (max-sensitivity
lit_z) for ≥1.5s; print `convergence` mid-run to debug. Do NOT change the replay weights — only the
synchrony/bell wiring.

- [ ] **Step 5: Run the full suite + lint**

Run: `python3 -m pytest -q` then `python3 -m ruff check .`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add blitz_overlay/pipeline.py tests/overlay/test_cue_verifier_pipeline.py
git commit -m "feat(overlay): pipeline drives synchrony + bell + cue_rows + sensitivity

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Browser — WebAudio bell + trust meter

**Files:**
- Create: `apps/overlay-web/js/bell.js`

> No JS test runner exists; browser tasks are verified in-browser (Task 10). Keep it defensive —
> WebAudio must lazy-init on first ring (autoplay policy) and never throw.

- [ ] **Step 1: Implement** — create `apps/overlay-web/js/bell.js`:

```javascript
/**
 * BellPlayer — plays an earned WebAudio chime on bell.just_rang and tracks a trust meter.
 *
 * Honest framing: the chime means "strong deception-pattern convergence," not "lie".
 * The trust meter = recent bell frequency (more bells in the window -> lower trust reading).
 */
const TRUST_WINDOW_MS = 60000;  // bells in the last minute drive the trust meter

export class BellPlayer {
  constructor() {
    this._ctx = null;
    this._bellTimes = [];   // timestamps (ms) of recent rings
  }

  /** Call every consensus frame with consensus.bell. */
  handle(bell) {
    if (bell && bell.just_rang) this._ring();
  }

  _ring() {
    try {
      if (!this._ctx) this._ctx = new (window.AudioContext || window.webkitAudioContext)();
      const ctx = this._ctx;
      const now = ctx.currentTime;
      // Two-tone chime (G5 -> C6), short decay.
      [784, 1047].forEach((freq, i) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = "sine";
        osc.frequency.value = freq;
        const t0 = now + i * 0.12;
        gain.gain.setValueAtTime(0.0001, t0);
        gain.gain.exponentialRampToValueAtTime(0.25, t0 + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.5);
        osc.connect(gain).connect(ctx.destination);
        osc.start(t0);
        osc.stop(t0 + 0.55);
      });
    } catch (err) {
      console.warn("[BellPlayer] chime failed (non-fatal):", err.message);
    }
    this._bellTimes.push(Date.now());
  }

  /** 0..1 trust reading: 1 = no recent bells, decreasing as bells accumulate. */
  trust() {
    const cutoff = Date.now() - TRUST_WINDOW_MS;
    this._bellTimes = this._bellTimes.filter((t) => t >= cutoff);
    // Each bell in the window knocks 20% off trust, floored at 0.
    return Math.max(0, 1 - this._bellTimes.length * 0.2);
  }

  bellCount() {
    const cutoff = Date.now() - TRUST_WINDOW_MS;
    return this._bellTimes.filter((t) => t >= cutoff).length;
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add apps/overlay-web/js/bell.js
git commit -m "feat(overlay): WebAudio bell chime + trust meter

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Browser — live Cue Verifier checklist + verdict

**Files:**
- Create: `apps/overlay-web/js/cue-verifier.js`

- [ ] **Step 1: Implement** — create `apps/overlay-web/js/cue-verifier.js`:

```javascript
/**
 * CueVerifier — renders the live "Parallel Cue Verifier" checklist from consensus.cue_rows,
 * the convergence counter, and an honest verdict line. A row lights up when its cue is lit.
 */
const STATUS_COLORS = {
  CALIBRATING: "#5b8def", CLEAR: "#28c76f", WATCH: "#ff9f43", FLAG: "#ea5455",
};
const VERDICT_TEXT = {
  CALIBRATING: "Calibrating baseline…",
  CLEAR: "No deception pattern",
  WATCH: "Deception-pattern risk rising",
  FLAG: "⚠ HIGH deception-pattern risk",
};

export class CueVerifier {
  constructor(els) {
    this.rows = els.rows;          // <ul> for cue rows
    this.verdict = els.verdict;    // verdict line element
    this.convergence = els.convergence;  // convergence counter element
    this._rowEls = new Map();      // cue_id -> {li, bar, z}
  }

  setConsensus(c) {
    const color = STATUS_COLORS[c.status] || "#888";

    // Verdict line
    this.verdict.textContent = VERDICT_TEXT[c.status] || c.status;
    this.verdict.style.color = color;

    // Convergence counter
    const cv = c.convergence || {};
    const burst = cv.burst ? " · BURST" : "";
    this.convergence.textContent =
      `${cv.n_lit || 0} cues · ${cv.n_families || 0} channels firing${burst}`;
    this.convergence.style.color = cv.burst ? "#ea5455" : "#7d8da3";

    // Rows (build once, then update in place)
    for (const row of c.cue_rows || []) {
      let entry = this._rowEls.get(row.cue_id);
      if (!entry) entry = this._createRow(row);
      const intensity = Math.max(0, Math.min(1, Math.abs(row.z) / 6));
      entry.bar.style.width = `${Math.round(intensity * 100)}%`;
      entry.bar.style.background = color;
      entry.li.classList.toggle("lit", !!row.lit);
      entry.li.classList.toggle("offline", !row.online);
      entry.z.textContent = row.z ? row.z.toFixed(1) : "—";
    }
  }

  _createRow(row) {
    const li = document.createElement("li");
    li.className = "cue-row";
    const fam = document.createElement("span");
    fam.className = "cue-fam";
    fam.textContent = row.family[0].toUpperCase();
    fam.title = row.family;
    const name = document.createElement("span");
    name.className = "cue-name";
    name.textContent = row.label;
    const barWrap = document.createElement("span");
    barWrap.className = "cue-bar-wrap";
    const bar = document.createElement("span");
    bar.className = "cue-bar";
    barWrap.appendChild(bar);
    const z = document.createElement("span");
    z.className = "cue-z";
    li.append(fam, name, barWrap, z);
    this.rows.appendChild(li);
    const entry = { li, bar, z };
    this._rowEls.set(row.cue_id, entry);
    return entry;
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add apps/overlay-web/js/cue-verifier.js
git commit -m "feat(overlay): live Parallel Cue Verifier checklist + honest verdict

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: Browser — DOM, styles, and wiring

**Files:**
- Modify: `apps/overlay-web/index.html`
- Modify: `apps/overlay-web/css/overlay.css`
- Modify: `apps/overlay-web/js/main.js`

- [ ] **Step 1: Add DOM** — in `apps/overlay-web/index.html`, inside the `<section class="panel">`,
add a verifier block after the `#panel-body` `<div>` (before `</section>`):

```html
      <div class="verifier">
        <div id="verdict" class="verdict">—</div>
        <div id="convergence" class="convergence">0 cues · 0 channels firing</div>
        <ul id="cue-rows" class="cue-rows"></ul>
        <div class="sensitivity">
          <label for="sens">sensitivity</label>
          <input id="sens" type="range" min="0" max="1" step="0.05" value="0" />
          <span class="sens-warn">↑ raises false alarms</span>
        </div>
        <div id="trust" class="trust">trust: 100% · bells/min: 0</div>
      </div>
```

- [ ] **Step 2: Add styles** — append to `apps/overlay-web/css/overlay.css`:

```css
/* ─── Parallel Cue Verifier ─── */
.verifier { padding: 8px 14px 14px; border-top: 1px solid #1f2733; }
.verdict { font-weight: 700; font-size: 15px; margin-bottom: 4px; }
.convergence { font: 11px/1.4 monospace; color: #7d8da3; margin-bottom: 8px; }
.cue-rows { list-style: none; margin: 0 0 10px; padding: 0; }
.cue-row {
  display: flex; align-items: center; gap: 8px;
  font: 11px/1.5 monospace; opacity: 0.5; transition: opacity .15s ease;
}
.cue-row.offline { opacity: 0.25; }
.cue-row.lit { opacity: 1; }
.cue-fam {
  width: 14px; height: 14px; flex-shrink: 0; border-radius: 3px;
  background: #2a3645; color: #cdd9e5; text-align: center; font-size: 9px; line-height: 14px;
}
.cue-row.lit .cue-fam { background: #ea5455; color: #fff; }
.cue-name { width: 96px; flex-shrink: 0; color: #9fb0c3; }
.cue-bar-wrap { flex: 1; height: 5px; background: #1f2733; border-radius: 3px; overflow: hidden; }
.cue-bar { display: block; height: 100%; width: 0%; transition: width .15s ease; }
.cue-z { width: 28px; text-align: right; flex-shrink: 0; color: #cdd9e5; }
.sensitivity { display: flex; align-items: center; gap: 6px; font: 11px/1 monospace; color: #9fb0c3; }
.sensitivity input { flex: 1; }
.sens-warn { color: #ff9f43; font-size: 10px; }
.trust { margin-top: 8px; font: 11px/1.4 monospace; color: #7d8da3; }
```

- [ ] **Step 3: Wire it in `main.js`** — edit `apps/overlay-web/js/main.js`:

(a) Add imports (after the `Enneagram` import):

```javascript
import { CueVerifier } from "./cue-verifier.js";
import { BellPlayer } from "./bell.js";
```

(b) Construct them (after `const enneagram = ...`):

```javascript
const cueVerifier = new CueVerifier({
  rows: document.getElementById("cue-rows"),
  verdict: document.getElementById("verdict"),
  convergence: document.getElementById("convergence"),
});
const bellPlayer = new BellPlayer();
const trustEl = document.getElementById("trust");
let _sensitivity = 0;
document.getElementById("sens").addEventListener("input", (e) => {
  _sensitivity = parseFloat(e.target.value);
});
```

(c) Extend the WS consensus callback to drive the verifier + bell + trust. Replace the existing
`new WsClient(...)` consensus arrow with:

```javascript
const ws = new WsClient(wsUrl, (c) => {
  renderer.setConsensus(c);
  enneagram.setConsensus(c);
  cueVerifier.setConsensus(c);
  bellPlayer.handle(c.bell);
  trustEl.textContent =
    `trust: ${Math.round(bellPlayer.trust() * 100)}% · bells/min: ${bellPlayer.bellCount()}`;
},
  (s) => { if (s === "engine-offline") panel.message.textContent = "Engine offline — reconnecting…"; });
```

(d) Attach sensitivity to each outgoing frame. In `loop()`, just before `ws.send(frame);`, add:

```javascript
  frame.config = { sensitivity: _sensitivity };
```

- [ ] **Step 4: Verify modules load** — start the server and load the page:

Run: `BLITZ_OVERLAY_OPEN_BROWSER=0 python3 -m blitz_overlay &` then open `http://127.0.0.1:8000` in
Chrome; confirm no console errors about `cue-verifier.js`/`bell.js` and the checklist rows render.
Stop the server after (`kill %1`).

- [ ] **Step 5: Commit**

```bash
git add apps/overlay-web/index.html apps/overlay-web/css/overlay.css apps/overlay-web/js/main.js
git commit -m "feat(overlay): wire checklist + bell + sensitivity slider + trust meter

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 10: Manual browser verification (user-confirmed)

**Files:** none.

- [ ] **Step 1: Launch**

Run: `BLITZ_OVERLAY_BASELINE_SECONDS=20 python3 -m blitz_overlay`
Open `http://127.0.0.1:8000` in Chrome; allow camera + mic.

- [ ] **Step 2: Verify**
  - The **cue checklist** renders one row per cue (visual/audio/linguistic/physio), dim when idle,
    lighting up + filling its bar as cues fire while you talk/move.
  - The **convergence counter** updates ("N cues · M channels firing"); shows **BURST** in red when
    ≥3 cues across ≥2 channels co-fire.
  - The **verdict line** climbs CLEAR → WATCH → "⚠ HIGH deception-pattern risk".
  - Dragging the **sensitivity slider up** makes the bell reachable; a **WebAudio chime** rings on a
    sustained burst; **trust %** drops and **bells/min** rises with each ring.
  - At slider = 0 (conservative) the bell stays rare/earned.

- [ ] **Step 3: Report to the user.** This is the in-browser confirmation gate. Do NOT merge/push to
  `main` until the user confirms the checklist + bell behave right (and the still-pending audio +
  linguistic confirmations).
```
