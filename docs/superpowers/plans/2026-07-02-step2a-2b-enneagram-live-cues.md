# Step 2a+2b — Family Organigram Enneagram + Eight Live Cues Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewire the enneagram canvas into an always-alive family-level organigram, and grow the live Visual family from 21 to 29 cues using only the existing MediaPipe stream.

**Architecture:** 2a is browser-only (`enneagram.js` rewrite + `main.js` hookup) — the consensus payload already carries everything needed (`families[].online/activity`, `active_cues`, `convergence`, `bell`, `turn_result`). 2b is Python-only — all needed blendshapes are already forwarded (see `USED_BLENDSHAPES` in `apps/overlay-web/js/schema.js`); new `CueDetector`s in `blitz_overlay/cues/visual.py` + science weights in `blitz_overlay/weights.py`, TDD throughout. Question-timed gaze folds into `content/fusion.py` (not a new detector).

**Tech Stack:** Vanilla Canvas 2D (browser), Python 3.14, pytest, ruff. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-02-step2-visual-deepening-design.md`

---

### Task 1: Enneagram rewrite — the family organigram

**Files:**
- Rewrite: `apps/overlay-web/js/enneagram.js`
- Modify: `apps/overlay-web/js/main.js` (turn_result + trust forwarding)

No Python changes → no pytest. Verify in-browser.

- [ ] **Step 1: Replace `enneagram.js` with the family organigram**

Nine permanent slots. Points 1–6 = channels (Visual, Audio, Linguistic, Physio, Content, Body-reserved), points 7–9 = meta (Synchrony, Consensus, Trust). Pull is driven by continuous family `activity` (base) and strongest active-cue z (spike) — **never multiplied by risk**. Risk drives fill alpha/tone. Idle breathing keeps the figure alive. The 3-6-9 triangle lights on the two-gate; hexad lines brighten with pairwise slot co-activity.

```js
// Enneagram — the family ORGANIGRAM: 9 permanent slots, always alive.
// Points 1-6 = channels (Visual Audio Linguistic Physio Content Body-reserved),
// points 7-9 = meta (Synchrony, Consensus, Trust). The Cue Polygon is the per-cue
// view; this is the family-level view. Pull = continuous activity + strongest cue z
// (NEVER gated by risk — risk drives fill/tone only).
const STATUS_COLORS = {
  CALIBRATING: "#5b8def", CLEAR: "#28c76f", WATCH: "#ff9f43", FLAG: "#ea5455",
};
// Family accents match cue-polygon.js FAMILY_COLORS; meta points get their own.
const SLOT_COLORS = [
  "#5b8def", // 1 visual
  "#28c76f", // 2 audio
  "#ff9f43", // 3 linguistic
  "#ea5455", // 4 physio
  "#b07cf7", // 5 content (Q&A engine)
  "#3a4a5c", // 6 body — reserved, dim until the family ships
  "#e6c94c", // 7 synchrony
  "#e8edf4", // 8 consensus
  "#7d8da3", // 9 trust
];
const SLOT_LABELS = ["VIS", "AUD", "LING", "PHY", "CNT", "BODY", "SYN", "CON", "TRU"];
const SLOT_FAMILY = ["visual", "audio", "linguistic", "physio", null, "body", null, null, null];
const INNER_LINES = [
  [0, 3], [3, 1], [1, 7], [7, 4], [4, 6], [6, 0], // hexad 1-4-2-8-5-7
];
const TRIANGLE = [[2, 5], [5, 8], [8, 2]];          // 3-6-9 — lights on the two-gate
const CONTENT_FADE_MS = 30_000;

export class Enneagram {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this._consensus = null;
    this._turn = null;        // {combined, ts}
    this._trust = 1.0;        // from BellPlayer (1 = trustworthy)
    this._running = false;
    this._pull = new Float32Array(9);
    this._lineLight = new Float32Array(INNER_LINES.length);
    this._triLight = 0;
    this._riskCur = 0;
    this._statusCur = "CALIBRATING";
    this._flagCur = false;
    this._phase = 0;
  }

  setConsensus(c) { this._consensus = c; }
  setTurn(r) { this._turn = { combined: r.combined ?? r.content_risk ?? 0, ts: performance.now() }; }
  setTrust(t) { this._trust = t; }

  start() {
    if (this._running) return;
    this._running = true;
    const tick = () => {
      if (!this._running) return;
      this._interpolate();
      this._draw();
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  stop() { this._running = false; }

  _slotTargets(c) {
    const t = new Float32Array(9);
    if (!c) return t;
    const fams = {};
    for (const f of c.families || []) fams[f.name] = f;
    const cues = c.active_cues || [];
    // Channels 1-4 (+6 body when it ships): continuous activity ∨ strongest cue spike
    for (let i = 0; i < 9; i++) {
      const famName = SLOT_FAMILY[i];
      if (!famName) continue;
      const f = fams[famName];
      if (!f || !f.online) { t[i] = 0; continue; }
      let spike = 0;
      for (const cu of cues) {
        if (cu.cue_id.startsWith(famName + ".")) spike = Math.max(spike, Math.abs(cu.z) / 4);
      }
      t[i] = Math.min(1, Math.max(f.activity || 0, spike));
    }
    // 5 content: last turn verdict, fading over 30 s
    if (this._turn) {
      const age = performance.now() - this._turn.ts;
      t[4] = Math.min(1, Math.max(0, this._turn.combined * (1 - age / CONTENT_FADE_MS)));
    }
    // 7 synchrony: channel-led convergence, flares on burst
    const cv = c.convergence || {};
    t[6] = Math.min(1, (cv.n_families || 0) / 4 + (cv.burst ? 0.5 : 0));
    // 8 consensus: fused posterior
    t[7] = Math.max(0, Math.min(1, c.risk || 0));
    // 9 trust: inverse trust (frequent bells push the point out)
    t[8] = Math.min(1, Math.max(0, 1 - this._trust));
    return t;
  }

  _interpolate() {
    const c = this._consensus;
    const EASE = 0.15;
    this._riskCur += ((c ? Math.max(0, Math.min(1, c.risk)) : 0) - this._riskCur) * EASE;
    if (c) { this._statusCur = c.status || "CALIBRATING"; this._flagCur = !!c.flag; }
    const targets = this._slotTargets(c);
    for (let i = 0; i < 9; i++) this._pull[i] += (targets[i] - this._pull[i]) * EASE;
    // Hexad: pairwise co-activity of the endpoints
    for (let li = 0; li < INNER_LINES.length; li++) {
      const [a, b] = INNER_LINES[li];
      const target = Math.sqrt(this._pull[a] * this._pull[b]);
      this._lineLight[li] += (target - this._lineLight[li]) * EASE;
    }
    // Triangle: the two-gate (≥2 families agree)
    const gate = c && (c.flag || (c.n_agree || 0) >= (c.n_required || 2)) ? 1 : 0;
    this._triLight += (gate - this._triLight) * EASE;
    this._phase += 0.03;
  }

  _draw() {
    const { canvas, ctx } = this;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0b0f14";
    ctx.fillRect(0, 0, W, H);
    const cx = W / 2, cy = H / 2 - 10;
    const baseR = Math.min(W, H) * 0.34;
    const statusColor = STATUS_COLORS[this._statusCur] || "#5b8def";
    const risk = this._riskCur;
    const pts = this._points(cx, cy, baseR);

    if (this._flagCur) {
      const pulse = 0.5 + 0.5 * Math.sin(this._phase * 2.4);
      const grad = ctx.createRadialGradient(cx, cy, baseR * 0.3, cx, cy, baseR * 1.45);
      grad.addColorStop(0, "rgba(234,84,85,0)");
      grad.addColorStop(1, `rgba(234,84,85,${0.3 * pulse})`);
      ctx.fillStyle = grad;
      ctx.beginPath(); ctx.arc(cx, cy, baseR * 1.5, 0, Math.PI * 2); ctx.fill();
    }

    // Hexad lines — co-activity brightening
    for (let li = 0; li < INNER_LINES.length; li++) {
      const [a, b] = INNER_LINES[li];
      const lit = this._lineLight[li];
      ctx.beginPath(); ctx.moveTo(pts[a].x, pts[a].y); ctx.lineTo(pts[b].x, pts[b].y);
      ctx.strokeStyle = this._alpha(lit > 0.15 ? statusColor : "#3a4a5c", 0.12 + lit * 0.5);
      ctx.lineWidth = 1 + lit * 1.5;
      ctx.stroke();
    }
    // Triangle 3-6-9 — the two-gate
    for (const [a, b] of TRIANGLE) {
      ctx.beginPath(); ctx.moveTo(pts[a].x, pts[a].y); ctx.lineTo(pts[b].x, pts[b].y);
      ctx.strokeStyle = this._alpha(this._triLight > 0.1 ? "#ea5455" : "#3a4a5c",
        0.15 + this._triLight * 0.7);
      ctx.lineWidth = 1 + this._triLight * 2;
      ctx.stroke();
    }

    // Outer ring — gradient stroke, fill intensity from risk (NOT deformation)
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < 9; i++) ctx.lineTo(pts[i].x, pts[i].y);
    ctx.closePath();
    ctx.fillStyle = this._alpha(statusColor, 0.05 + risk * 0.18);
    ctx.fill();
    const ringGrad = ctx.createLinearGradient(cx - baseR, cy - baseR, cx + baseR, cy + baseR);
    ringGrad.addColorStop(0, this._alpha(statusColor, 0.85));
    ringGrad.addColorStop(1, this._alpha(statusColor, 0.35));
    ctx.strokeStyle = ringGrad;
    ctx.lineWidth = 1.6;
    ctx.stroke();

    // Points: per-slot accent color, glow ∝ pull, label on every point
    let maxIdx = 0;
    for (let i = 1; i < 9; i++) if (this._pull[i] > this._pull[maxIdx]) maxIdx = i;
    for (let i = 0; i < 9; i++) {
      const p = pts[i];
      const pull = this._pull[i];
      const col = SLOT_COLORS[i];
      const offline = this._isOffline(i);
      if (pull > 0.03 && !offline) {
        const glowR = 6 + pull * 20;
        const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, glowR);
        g.addColorStop(0, this._alpha(col, 0.65 * pull));
        g.addColorStop(1, this._alpha(col, 0));
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(p.x, p.y, glowR, 0, Math.PI * 2); ctx.fill();
      }
      ctx.beginPath();
      ctx.arc(p.x, p.y, offline ? 2 : 2.5 + pull * 4, 0, Math.PI * 2);
      ctx.fillStyle = this._alpha(col, offline ? 0.25 : 0.5 + pull * 0.5);
      ctx.fill();
      const ang = this._angle(i);
      ctx.font = i === maxIdx && pull > 0.15 ? "bold 9px monospace" : "8px monospace";
      ctx.fillStyle = this._alpha(col, offline ? 0.3 : 0.55 + pull * 0.45);
      ctx.textAlign = "center";
      ctx.fillText(SLOT_LABELS[i], p.x + Math.cos(ang) * 16, p.y + Math.sin(ang) * 16 + 3);
      ctx.textAlign = "left";
    }

    ctx.font = "11px monospace";
    ctx.fillStyle = "#7d8da3";
    ctx.textAlign = "center";
    ctx.fillText(`RISK  ${Math.round(risk * 100)}%`, cx, H - 8);
    ctx.textAlign = "left";
  }

  _isOffline(i) {
    const famName = SLOT_FAMILY[i];
    if (!famName) return false;
    const fams = this._consensus ? this._consensus.families || [] : [];
    const f = fams.find((x) => x.name === famName);
    return !f || !f.wired || !f.online;
  }

  _angle(i) { return -Math.PI / 2 + (i / 9) * Math.PI * 2; }

  _points(cx, cy, baseR) {
    const maxDeform = baseR * 0.42;
    const pts = [];
    for (let i = 0; i < 9; i++) {
      const ang = this._angle(i);
      // Idle breathing: subtle per-point wobble so the figure lives even at rest
      const breathe = Math.sin(this._phase + i * 0.7) * (this._isOffline(i) ? 0.6 : 1.8);
      const r = baseR + this._pull[i] * maxDeform + breathe;
      pts.push({ x: cx + Math.cos(ang) * r, y: cy + Math.sin(ang) * r });
    }
    return pts;
  }

  _alpha(hex, a) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${Math.max(0, Math.min(1, a)).toFixed(3)})`;
  }
}
```

- [ ] **Step 2: Hook `main.js` — turn_result + trust forwarding**

In the WS handler, forward turn results to the enneagram and push trust after each bell update:

```js
const ws = new WsClient(wsUrl, (c) => {
  if (c.type === "turn_result") { qaPanel.showResult(c); enneagram.setTurn(c); return; }
  renderer.setConsensus(c);
  enneagram.setConsensus(c);
  cueVerifier.setConsensus(c);
  cuePolygon.setConsensus(c);
  calibration.setConsensus(c);
  bellPlayer.handle(c.bell);
  enneagram.setTrust(bellPlayer.trust());
  trustEl.textContent =
    `trust: ${Math.round(bellPlayer.trust() * 100)}% · bells/min: ${bellPlayer.bellCount()}`;
},
```

- [ ] **Step 3: Sanity run**

Run: `python3 -m ruff check . && python3 -m pytest -q` (no Python touched — must stay 159 passed)
Then `BLITZ_OVERLAY_BASELINE_SECONDS=20 python3 -m blitz_overlay` and confirm: all 9 labels visible, points breathe at idle, BODY dim, moving/talking makes VIS/AUD/LING pull outward without needing risk.

- [ ] **Step 4: Commit**

```bash
git add apps/overlay-web/js/enneagram.js apps/overlay-web/js/main.js
git commit -m "feat(overlay): enneagram rewired as family organigram — 6 channels + 3 meta, activity-driven, risk decoupled"
```

---

### Task 2: Frame cues — duchenne_absence, stress_brow, face_asymmetry

**Files:**
- Modify: `blitz_overlay/cues/visual.py` (append detectors + registry)
- Modify: `blitz_overlay/weights.py` (3 entries + version bump)
- Test: `tests/overlay/test_visual_cues.py`

- [ ] **Step 1: Write failing tests** (append to `tests/overlay/test_visual_cues.py`; `_frame` helper already exists there)

```python
def test_duchenne_absence_high_when_smile_without_cheeks():
    from blitz_overlay.cues.visual import DuchenneAbsence
    d = DuchenneAbsence()
    masked = d.measure(_frame(0, mouthSmileLeft=0.8, mouthSmileRight=0.8,
                              cheekSquintLeft=0.05, cheekSquintRight=0.05))
    genuine = d.measure(_frame(0, mouthSmileLeft=0.8, mouthSmileRight=0.8,
                               cheekSquintLeft=0.7, cheekSquintRight=0.7))
    assert masked > genuine
    assert d.measure(_frame(0, mouthSmileLeft=0.1, mouthSmileRight=0.1,
                            cheekSquintLeft=0.0, cheekSquintRight=0.0)) == 0.0  # no smile → no signal
    assert d.measure(_frame(0, browInnerUp=0.5)) is None  # inputs absent → abstain


def test_stress_brow_requires_co_occurrence():
    from blitz_overlay.cues.visual import StressBrow
    d = StressBrow()
    all_up = d.measure(_frame(0, browInnerUp=0.6, browOuterUpLeft=0.5, browOuterUpRight=0.5,
                              browDownLeft=0.4, browDownRight=0.4))
    inner_only = d.measure(_frame(0, browInnerUp=0.6, browOuterUpLeft=0.0, browOuterUpRight=0.0,
                                  browDownLeft=0.0, browDownRight=0.0))
    assert all_up > inner_only
    assert inner_only == 0.0  # AU1 alone is not the AU1+2+4 combo
    assert d.measure(_frame(0, jawOpen=0.5)) is None


def test_face_asymmetry_averages_lr_pairs():
    from blitz_overlay.cues.visual import FaceAsymmetry
    d = FaceAsymmetry()
    sym = d.measure(_frame(0, eyeSquintLeft=0.4, eyeSquintRight=0.4,
                           browDownLeft=0.3, browDownRight=0.3))
    asym = d.measure(_frame(0, eyeSquintLeft=0.8, eyeSquintRight=0.1,
                            browDownLeft=0.6, browDownRight=0.1))
    assert asym > sym
    assert sym == 0.0
    assert d.measure(_frame(0, browInnerUp=0.5)) is None  # no paired keys → abstain
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/overlay/test_visual_cues.py -q`
Expected: FAIL — `ImportError: cannot import name 'DuchenneAbsence'`

- [ ] **Step 3: Implement detectors** (append to `blitz_overlay/cues/visual.py`, before `VISUAL_DETECTORS`)

```python
SMILE_FLOOR = 0.3   # below this there is no smile to authenticate


class DuchenneAbsence(CueDetector):
    """Smile without eye involvement (AU12 high, AU6 low) — social/masked smile.

    Ekman's Duchenne marker: genuine enjoyment recruits orbicularis oculi (cheekSquint).
    Signal only exists while smiling; no smile → 0 (not suspicious).
    """

    cue_id = "visual.duchenne_absence"
    direction = 1

    def measure(self, frame: FeatureFrame) -> float | None:
        bs = frame.blendshapes
        keys = ("mouthSmileLeft", "mouthSmileRight", "cheekSquintLeft", "cheekSquintRight")
        if not any(k in bs for k in keys):
            return None
        smile = (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0)) / 2.0
        if smile < SMILE_FLOOR:
            return 0.0
        cheek = (bs.get("cheekSquintLeft", 0.0) + bs.get("cheekSquintRight", 0.0)) / 2.0
        return smile * max(0.0, smile - cheek)


class StressBrow(CueDetector):
    """AU1+AU2+AU4 co-occurrence — the fear/stress brow. All three must be present."""

    cue_id = "visual.stress_brow"
    direction = 1

    def measure(self, frame: FeatureFrame) -> float | None:
        bs = frame.blendshapes
        keys = ("browInnerUp", "browOuterUpLeft", "browOuterUpRight",
                "browDownLeft", "browDownRight")
        if not any(k in bs for k in keys):
            return None
        inner = bs.get("browInnerUp", 0.0)
        outer = (bs.get("browOuterUpLeft", 0.0) + bs.get("browOuterUpRight", 0.0)) / 2.0
        down = (bs.get("browDownLeft", 0.0) + bs.get("browDownRight", 0.0)) / 2.0
        return min(inner, outer, down)  # co-occurrence: the weakest component gates the combo


# L/R blendshape pairs for the multi-region asymmetry index (smile/dimple asymmetry
# already has dedicated cues — excluded to avoid double counting).
ASYMMETRY_PAIRS = (
    ("eyeBlinkLeft", "eyeBlinkRight"),
    ("eyeSquintLeft", "eyeSquintRight"),
    ("browDownLeft", "browDownRight"),
    ("mouthStretchLeft", "mouthStretchRight"),
    ("mouthFrownLeft", "mouthFrownRight"),
    ("mouthPressLeft", "mouthPressRight"),
)


class FaceAsymmetry(CueDetector):
    """Multi-region left/right deviation (eye + brow + mouth) beyond the smile cues."""

    cue_id = "visual.face_asymmetry"
    direction = 1

    def measure(self, frame: FeatureFrame) -> float | None:
        bs = frame.blendshapes
        deltas = [abs(bs[left] - bs[right])
                  for left, right in ASYMMETRY_PAIRS if left in bs and right in bs]
        if not deltas:
            return None
        return sum(deltas) / len(deltas)
```

Add to the registry:

```python
VISUAL_DETECTORS = [
    BlinkRate, GazeAversion, BrowFlash, LipPress, JawTension,
    GazeFixation, PupilDilation, EyeBlocking, EyeWiden, NoseWrinkle, AsymmetricSmile,
    HeadMovement, EyeSquint, MouthStretch, MouthFrown, MouthShrug,
    JawShift, JawDrop, LipRoll, BrowOuterRaise, ContemptAsymmetry,
    DuchenneAbsence, StressBrow, FaceAsymmetry,
]
```

- [ ] **Step 4: Add weights** (append to `CUE_WEIGHTS` in `blitz_overlay/weights.py`; bump `WEIGHT_SET_VERSION` to `"step2-2026-07-02"`)

```python
    "visual.duchenne_absence": {
        "effect_size_d": 0.35,
        "reliability_tier": 3,
        "family": "visual",
        "region": "mouth",
        "citation": (
            "Ekman & Friesen — Duchenne marker: AU12 without AU6 = social/masked smile; "
            "moderate diagnosticity for masked affect (CUE_CATALOG.md cue 5 family)."
        ),
    },
    "visual.stress_brow": {
        "effect_size_d": 0.30,
        "reliability_tier": 3,
        "family": "visual",
        "region": "brow",
        "citation": (
            "FACS — AU1+AU2+AU4 combination: fear/stress brow; combo more specific than "
            "single-AU brow movement (catalog cue 9 family)."
        ),
    },
    "visual.face_asymmetry": {
        "effect_size_d": 0.30,
        "reliability_tier": 3,
        "family": "visual",
        "region": "mouth",
        "citation": (
            "Facial asymmetry under load — unilateral action intensity differences; "
            "weak-moderate, person-relative baseline required (catalog cue 5/12 family)."
        ),
    },
```

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest tests/overlay/test_visual_cues.py tests/overlay/test_weights.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add blitz_overlay/cues/visual.py blitz_overlay/weights.py tests/overlay/test_visual_cues.py
git commit -m "feat(overlay): duchenne_absence, stress_brow, face_asymmetry cues (Visual 21→24)"
```

---

### Task 3: Temporal cues — head_velocity, head_acceleration, blink_duration

**Files:**
- Modify: `blitz_overlay/cues/visual.py`
- Modify: `blitz_overlay/weights.py`
- Test: `tests/overlay/test_visual_cues.py`

- [ ] **Step 1: Write failing tests**

```python
def test_head_velocity_measures_rotation_speed():
    from blitz_overlay.cues.visual import HeadVelocity
    d = HeadVelocity()
    d.measure(_frame(0, g_yaw=0.0))
    still = d.measure(_frame(100, g_yaw=0.0))
    d2 = HeadVelocity()
    d2.measure(_frame(0, g_yaw=0.0))
    moving = d2.measure(_frame(100, g_yaw=8.0))   # 8° in 100 ms = 80°/s
    assert moving > still
    assert d.measure(FeatureFrame.from_dict({"ts": 200, "face_present": True})) is None


def test_head_acceleration_spikes_on_sudden_onset():
    from blitz_overlay.cues.visual import HeadAcceleration
    d = HeadAcceleration()
    # steady rotation: constant velocity → low acceleration
    for i, yaw in enumerate([0.0, 2.0, 4.0, 6.0]):
        steady = d.measure(_frame(i * 100, g_yaw=yaw))
    d2 = HeadAcceleration()
    # sudden onset: still, still, then a jerk
    for i, yaw in enumerate([0.0, 0.0, 0.0, 9.0]):
        sudden = d2.measure(_frame(i * 100, g_yaw=yaw))
    assert sudden > steady


def test_blink_duration_reports_last_completed_blink():
    from blitz_overlay.cues.visual import BlinkDuration
    d = BlinkDuration()
    d.measure(_frame(0, eyeBlinkLeft=0.1))          # open
    d.measure(_frame(100, eyeBlinkLeft=0.9))        # closes
    d.measure(_frame(500, eyeBlinkLeft=0.9))        # held closed
    val = d.measure(_frame(600, eyeBlinkLeft=0.1))  # reopens → blink took ~500 ms
    assert abs(val - 0.5) < 0.05
    assert d.measure(_frame(700, eyeBlinkLeft=0.1)) > 0.0   # remembered within window
    assert d.measure(_frame(20_000, eyeBlinkLeft=0.1)) == 0.0  # decayed after window
```

Note: the `_frame` helper routes `g_yaw` into `head_pose.yaw` already (see helper at top of file).

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/overlay/test_visual_cues.py -q`
Expected: FAIL — `ImportError: cannot import name 'HeadVelocity'`

- [ ] **Step 3: Implement** (append to `visual.py`)

```python
HEAD_KIN_WINDOW_MS = 800         # smoothing window for velocity/acceleration
BLINK_MEMORY_MS = 5_000          # a completed blink stays reportable this long


class _HeadKinematics(CueDetector):
    """Shared pose-history buffer for velocity/acceleration cues."""

    direction = 1

    def __init__(self) -> None:
        super().__init__()
        self._hist: deque[tuple[int, float, float, float]] = deque()

    def _push(self, frame: FeatureFrame) -> list[tuple[int, float, float, float]] | None:
        hp = frame.head_pose
        if not hp or not any(k in hp for k in ("yaw", "pitch", "roll")):
            return None
        now = frame.ts
        self._hist.append((now, float(hp.get("yaw", 0.0)),
                           float(hp.get("pitch", 0.0)), float(hp.get("roll", 0.0))))
        while self._hist and self._hist[0][0] < now - HEAD_KIN_WINDOW_MS:
            self._hist.popleft()
        return list(self._hist)

    @staticmethod
    def _velocities(hist) -> list[float]:
        out = []
        for (t0, y0, p0, r0), (t1, y1, p1, r1) in zip(hist, hist[1:], strict=False):
            dt = max(1, t1 - t0) / 1000.0
            dist = ((y1 - y0) ** 2 + (p1 - p0) ** 2 + (r1 - r0) ** 2) ** 0.5
            out.append(dist / dt)
        return out


class HeadVelocity(_HeadKinematics):
    """Head rotation speed (°/s) — nods/shakes/tilts as tension or emphasis."""

    cue_id = "visual.head_velocity"

    def measure(self, frame: FeatureFrame) -> float | None:
        hist = self._push(frame)
        if hist is None:
            return None
        v = self._velocities(hist)
        return sum(v) / len(v) if v else 0.0


class HeadAcceleration(_HeadKinematics):
    """Sudden head-movement onsets (°/s²) — jerky motion vs steady rotation."""

    cue_id = "visual.head_acceleration"

    def measure(self, frame: FeatureFrame) -> float | None:
        hist = self._push(frame)
        if hist is None:
            return None
        v = self._velocities(hist)
        if len(v) < 2:
            return 0.0
        acc = []
        for (t0, *_), v0, v1 in zip(hist, v, v[1:], strict=False):
            acc.append(abs(v1 - v0))
        return max(acc)


class BlinkDuration(CueDetector):
    """Duration of the most recent completed blink (s) — long blinks and slow rebound."""

    cue_id = "visual.blink_duration"
    direction = 1

    def __init__(self) -> None:
        super().__init__()
        self._closed_since: int | None = None
        self._last_blink: tuple[int, float] | None = None   # (ended_ts, duration_s)

    def measure(self, frame: FeatureFrame) -> float | None:
        bs = frame.blendshapes
        if "eyeBlinkLeft" not in bs and "eyeBlinkRight" not in bs:
            return None
        closed = max(bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0)) >= BLINK_CLOSED_THRESHOLD
        now = frame.ts
        if closed and self._closed_since is None:
            self._closed_since = now
        elif not closed and self._closed_since is not None:
            self._last_blink = (now, (now - self._closed_since) / 1000.0)
            self._closed_since = None
        if self._last_blink and now - self._last_blink[0] <= BLINK_MEMORY_MS:
            return self._last_blink[1]
        return 0.0
```

Registry grows:

```python
    DuchenneAbsence, StressBrow, FaceAsymmetry,
    HeadVelocity, HeadAcceleration, BlinkDuration,
]
```

- [ ] **Step 4: Add weights**

```python
    "visual.head_velocity": {
        "effect_size_d": 0.25,
        "reliability_tier": 3,
        "family": "visual",
        "region": "head",
        "citation": (
            "Catalog cue 14 family — head movement dynamics; velocity component, weak "
            "single-cue diagnosticity, person-relative."
        ),
    },
    "visual.head_acceleration": {
        "effect_size_d": 0.25,
        "reliability_tier": 3,
        "family": "visual",
        "region": "head",
        "citation": (
            "Catalog cue 14 family — sudden movement onsets (jerk) distinct from sustained "
            "restlessness; weak, person-relative."
        ),
    },
    "visual.blink_duration": {
        "effect_size_d": 0.30,
        "reliability_tier": 3,
        "family": "visual",
        "region": "eyes",
        "citation": (
            "Catalog cue 60 family — blink duration/rebound timing complements rate; "
            "long closures under load."
        ),
    },
```

- [ ] **Step 5: Run tests** — `python3 -m pytest tests/overlay/test_visual_cues.py tests/overlay/test_weights.py -q` → PASS

- [ ] **Step 6: Commit**

```bash
git add blitz_overlay/cues/visual.py blitz_overlay/weights.py tests/overlay/test_visual_cues.py
git commit -m "feat(overlay): head velocity/acceleration + blink duration cues (Visual 24→27)"
```

---

### Task 4: Windowed cues — facial_rigidity, microexpression_burst

**Files:**
- Modify: `blitz_overlay/cues/visual.py`
- Modify: `blitz_overlay/weights.py`
- Test: `tests/overlay/test_visual_cues.py`

- [ ] **Step 1: Write failing tests**

```python
def test_facial_rigidity_low_variance_scores_high():
    from blitz_overlay.cues.visual import FacialRigidity
    frozen = FacialRigidity()
    lively = FacialRigidity()
    for i in range(30):
        frozen.measure(_frame(i * 100, browInnerUp=0.30, mouthSmileLeft=0.20))
        wob = 0.3 * (i % 2)
        lively.measure(_frame(i * 100, browInnerUp=0.30 + wob, mouthSmileLeft=0.20 + wob))
    v_frozen = frozen.measure(_frame(3000, browInnerUp=0.30, mouthSmileLeft=0.20))
    v_lively = lively.measure(_frame(3000, browInnerUp=0.30, mouthSmileLeft=0.20))
    # direction = -1: LOW expressivity variance is the suspicious pole, so the raw
    # measure (variance) must be LOWER for the frozen face.
    assert v_frozen < v_lively
    assert frozen.direction == -1


def test_microexpression_burst_spikes_on_fast_onset():
    from blitz_overlay.cues.visual import MicroexpressionBurst
    slow = MicroexpressionBurst()
    fast = MicroexpressionBurst()
    for i in range(10):
        slow.measure(_frame(i * 100, mouthFrownLeft=i * 0.01))       # creeping change
        fast.measure(_frame(i * 100, mouthFrownLeft=0.6 if i == 5 else 0.0))  # 1-frame flash
    v_slow = slow.measure(_frame(1000, mouthFrownLeft=0.1))
    v_fast = fast.measure(_frame(1000, mouthFrownLeft=0.0))
    assert v_fast > v_slow
    assert MicroexpressionBurst().measure(
        FeatureFrame.from_dict({"ts": 0, "face_present": True})) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/overlay/test_visual_cues.py -q`
Expected: FAIL — `ImportError: cannot import name 'FacialRigidity'`

- [ ] **Step 3: Implement** (append to `visual.py`)

```python
RIGIDITY_WINDOW_MS = 4_000       # window for expressivity variance
MICRO_WINDOW_MS = 1_000          # window for onset-velocity spikes


class _BlendshapeWindowCue(CueDetector):
    """Shared rolling buffer of the full blendshape vector."""

    direction = 1
    window_ms = 1_000

    def __init__(self) -> None:
        super().__init__()
        self._hist: deque[tuple[int, dict]] = deque()

    def _push(self, frame: FeatureFrame) -> list[tuple[int, dict]] | None:
        bs = frame.blendshapes
        if not bs:
            return None
        now = frame.ts
        self._hist.append((now, dict(bs)))
        while self._hist and self._hist[0][0] < now - self.window_ms:
            self._hist.popleft()
        return list(self._hist)


class FacialRigidity(_BlendshapeWindowCue):
    """Expressivity collapse — LOW blendshape variance over ~4 s = freezing under load.

    Raw measure = mean per-key standard deviation; direction = -1 (low is suspicious).
    """

    cue_id = "visual.facial_rigidity"
    direction = -1
    window_ms = RIGIDITY_WINDOW_MS

    def measure(self, frame: FeatureFrame) -> float | None:
        hist = self._push(frame)
        if hist is None:
            return None
        if len(hist) < 5:
            return None   # not enough window yet — abstain, don't fake stillness
        keys = set()
        for _, bs in hist:
            keys.update(bs)
        stds = []
        for k in keys:
            vals = [bs.get(k, 0.0) for _, bs in hist]
            mean = sum(vals) / len(vals)
            stds.append((sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5)
        return sum(stds) / len(stds) if stds else None


class MicroexpressionBurst(_BlendshapeWindowCue):
    """Fast-onset proxy: peak frame-to-frame blendshape delta in ~1 s.

    HONEST caveat (governance): micro-expressions have low base rates and modest
    real-world effect sizes — weighted low (tier 4) by design.
    """

    cue_id = "visual.microexpression_burst"
    direction = 1
    window_ms = MICRO_WINDOW_MS

    def measure(self, frame: FeatureFrame) -> float | None:
        hist = self._push(frame)
        if hist is None:
            return None
        if len(hist) < 2:
            return 0.0
        peak = 0.0
        for (_, a), (_, b) in zip(hist, hist[1:], strict=False):
            keys = set(a) | set(b)
            delta = sum(abs(b.get(k, 0.0) - a.get(k, 0.0)) for k in keys) / max(1, len(keys))
            peak = max(peak, delta)
        return peak
```

Registry grows (final — 29 visual cues):

```python
    HeadVelocity, HeadAcceleration, BlinkDuration,
    FacialRigidity, MicroexpressionBurst,
]
```

- [ ] **Step 4: Add weights**

```python
    "visual.facial_rigidity": {
        "effect_size_d": 0.35,
        "reliability_tier": 3,
        "family": "visual",
        "region": "head",
        "citation": (
            "Reduced expressivity/illustrators under cognitive load — overall rigidity vs "
            "personal baseline (DePaulo et al. 2003 cue family: decreased movement)."
        ),
    },
    "visual.microexpression_burst": {
        "effect_size_d": 0.20,
        "reliability_tier": 4,
        "family": "visual",
        "region": "head",
        "citation": (
            "Micro-expression onset proxy — HONEST caveat: low base rates and modest "
            "real-world effect sizes (Porter & ten Brinke 2008); weighted low by design."
        ),
    },
```

- [ ] **Step 5: Add a registry-count regression test**

```python
def test_registry_has_29_visual_cues():
    assert len(VISUAL_DETECTORS) == 29
    assert len({d().cue_id for d in VISUAL_DETECTORS}) == 29
```

- [ ] **Step 6: Run the full suite** — `python3 -m pytest -q` → PASS, `python3 -m ruff check .` → clean

- [ ] **Step 7: Commit**

```bash
git add blitz_overlay/cues/visual.py blitz_overlay/weights.py tests/overlay/test_visual_cues.py
git commit -m "feat(overlay): facial rigidity + microexpression burst cues (Visual 27→29)"
```

---

### Task 5: Question-timed gaze weighting in content fusion

**Files:**
- Modify: `blitz_overlay/content/fusion.py`
- Test: `tests/overlay/test_content_fusion.py`

- [ ] **Step 1: Write failing test** (append; see existing tests in the file for the verdict stub pattern)

```python
def test_gaze_aligned_answer_window_adds_confidence():
    from blitz_overlay.content.fusion import fuse_turn

    class V:
        available = True
        risk = 0.5
        def to_dict(self): return {"risk": self.risk}

    base_window = {"n_frames": 10, "cue_ids": [], "families": [],
                   "peak_z": 0.0, "max_families_synchronous": 0}
    gaze_window = dict(base_window, cue_ids=["visual.gaze_aversion"])

    plain = fuse_turn(V(), base_window)
    gazey = fuse_turn(V(), gaze_window)
    assert gazey["combined"] > plain["combined"]
    assert gazey["gaze_aligned"] is True
    assert plain["gaze_aligned"] is False

    # Below the WATCH floor, gaze alone must NOT move the verdict
    class Low(V):
        risk = 0.1
    assert fuse_turn(Low(), gaze_window)["combined"] == fuse_turn(Low(), base_window)["combined"]
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/overlay/test_content_fusion.py -q`
Expected: FAIL — `KeyError: 'gaze_aligned'`

- [ ] **Step 3: Implement in `fusion.py`**

Add near the top:

```python
GAZE_CUES = {"visual.gaze_aversion", "visual.gaze_fixation"}
GAZE_BOOST = 0.05
```

In `fuse_turn`, after `cue_confirms` is computed:

```python
    gaze_aligned = bool(GAZE_CUES & set(cue_window.get("cue_ids", [])))
```

In the content-available branch, extend the combine line and result:

```python
    combined = content_risk + (0.15 if cue_confirms and content_risk >= WATCH else 0.0)
    combined += GAZE_BOOST if gaze_aligned and content_risk >= WATCH else 0.0
    combined = max(0.0, min(1.0, combined))
```

Add `"gaze_aligned": gaze_aligned,` to BOTH returned dicts (offline fallback gets `"gaze_aligned": False` — no content window to align to... actually compute it there too, it is still honest cue information: use the same `gaze_aligned` value in both).

- [ ] **Step 4: Run tests** — `python3 -m pytest tests/overlay/test_content_fusion.py -q` → PASS

- [ ] **Step 5: Commit**

```bash
git add blitz_overlay/content/fusion.py tests/overlay/test_content_fusion.py
git commit -m "feat(content): question-timed gaze weighting — gaze cues in the answer window add confidence"
```

---

### Task 6: Governance note, full verification, push

**Files:**
- Modify: `docs/OVERLAY_README.md` (micro-expression caveat)

- [ ] **Step 1: Add the caveat to `docs/OVERLAY_README.md`** (after the audio caveat blockquote)

```markdown
> **Micro-expression caveat:** the `microexpression_burst` cue is an onset-velocity *proxy*,
> not true micro-expression recognition. Micro-expressions have low base rates and modest
> real-world effect sizes in the literature — the cue is deliberately weighted low (tier 4)
> and can never drive a FLAG on its own (two-gate still requires a second family).
```

- [ ] **Step 2: Full verification**

Run: `python3 -m ruff check . && python3 -m pytest -q`
Expected: clean + all tests pass (159 + ~10 new)

- [ ] **Step 3: Update cue counts in docs** — README.md + OVERLAY_README.md say "32 cues … 21 visual"; update to "40 cues … 29 visual" everywhere those counts appear (`grep -rn "32 cues\|21 visual" README.md docs/`).

- [ ] **Step 4: Commit + push**

```bash
git add docs/ README.md
git commit -m "docs: micro-expression caveat + cue counts 32→40 (Visual 29)"
git push origin main
```

- [ ] **Step 5: In-browser verification with the user** — run `BLITZ_OVERLAY_BASELINE_SECONDS=20 python3 -m blitz_overlay`; confirm the new enneagram and that calibration completes with the 8 new cues producing.
