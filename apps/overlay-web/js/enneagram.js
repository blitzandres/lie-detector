// Enneagram visualization — live deforming signal fingerprint
// STATUS_COLORS mirrors overlay-renderer.js (not exported there, so defined locally)
const STATUS_COLORS = {
  CALIBRATING: "#5b8def",
  CLEAR: "#28c76f",
  WATCH: "#ff9f43",
  FLAG: "#ea5455",
};

// 9 cue slots in fixed order — slots 7-8 are placeholders for future families
const CUE_SLOTS = [
  "visual.gaze_aversion",
  "visual.blink_rate",
  "visual.brow_flash",
  "visual.lip_press",
  "visual.jaw_tension",
  "physio.heart_rate",
  "audio.tremor",
  "linguistic.verbal",
  "cbca.content",
];

// Classic enneagram inner lines: the 1-4-2-8-5-7 hexad and 3-6-9 triangle.
// Using 0-based indices (slot 0 = point 1, etc.)
const INNER_LINES = [
  [0, 3], [3, 1], [1, 7], [7, 4], [4, 6], [6, 0], // hexad: 1-4-2-8-5-7
  [2, 5], [5, 8], [8, 2],                           // triangle: 3-6-9
];

// Short display name for each slot (shown on highest-pull point)
const CUE_LABELS = [
  "gaze_aversion",
  "blink_rate",
  "brow_flash",
  "lip_press",
  "jaw_tension",
  "heart_rate",
  "tremor",
  "verbal",
  "content",
];

export class Enneagram {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this._consensus = null;
    this._running = false;

    // Per-point interpolated pull (0..1)
    this._pull = new Float32Array(9);
    // Per-point interpolated glow alpha (0..1)
    this._glow = new Float32Array(9);
    // Pulse state for FLAG
    this._pulsePhase = 0;
    // Agreement brightening per inner line (0..1 interpolated)
    this._lineLight = new Float32Array(INNER_LINES.length);

    // Global risk amplitude interpolated
    this._riskCur = 0;
    this._statusCur = "CALIBRATING";
    this._flagCur = false;
  }

  /** Called by main.js each time a consensus message arrives (~10 Hz). */
  setConsensus(consensus) {
    this._consensus = consensus;
  }

  /** Kick off the 60fps rAF loop. Call once after construction. */
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

  stop() {
    this._running = false;
  }

  // ── Interpolation ──────────────────────────────────────────────────────────

  _interpolate() {
    const c = this._consensus;
    const EASE = 0.15;

    // Risk amplitude
    const targetRisk = c ? Math.max(0, Math.min(1, c.risk)) : 0;
    this._riskCur += (targetRisk - this._riskCur) * EASE;

    // Status / flag
    if (c) {
      this._statusCur = c.status || "CALIBRATING";
      this._flagCur = !!c.flag;
    }

    // Per-point pull from active_cues z-scores
    const cues = c ? c.active_cues || [] : [];
    for (let i = 0; i < 9; i++) {
      const slotId = CUE_SLOTS[i];
      const hit = cues.find((cu) => cu.cue_id === slotId);
      const targetPull = hit ? Math.max(0, Math.min(1, hit.z / 6)) : 0;
      this._pull[i] += (targetPull - this._pull[i]) * EASE;
      // Glow slightly faster than pull for snappier visual feedback
      this._glow[i] += (targetPull - this._glow[i]) * (EASE * 1.5);
    }

    // Inner line brightening: lit when both endpoint families agree
    // We check n_agree > 0 and map each line's pair of endpoints to their family votes
    const families = c ? c.families || [] : [];
    // Build family vote map keyed by name
    const fvote = {};
    for (const f of families) {
      fvote[f.name] = f.wired && f.fresh && f.vote;
    }
    // Each slot belongs to a family: visual(0-4), physio(5), audio(6), linguistic(7), cbca(8)
    const slotFamily = [
      "visual", "visual", "visual", "visual", "visual",
      "physio", "audio", "linguistic", "cbca",
    ];
    for (let li = 0; li < INNER_LINES.length; li++) {
      const [a, b] = INNER_LINES[li];
      const bothAgree =
        (fvote[slotFamily[a]] || false) && (fvote[slotFamily[b]] || false);
      const targetLine = bothAgree ? 1 : 0;
      this._lineLight[li] += (targetLine - this._lineLight[li]) * EASE;
    }

    // Pulse phase for FLAG glow (continuous even if flag drops, fades naturally)
    this._pulsePhase += 0.06; // ~3.6 Hz at 60fps
  }

  // ── Drawing ────────────────────────────────────────────────────────────────

  _draw() {
    const canvas = this.canvas;
    const ctx = this.ctx;
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Background fill — deep navy matching overlay.css body bg
    ctx.fillStyle = "#0b0f14";
    ctx.fillRect(0, 0, W, H);

    const cx = W / 2;
    const cy = H / 2 - 14; // slight upward offset to leave label space below
    const baseR = Math.min(W, H) * 0.36; // base radius of the ring
    const color = STATUS_COLORS[this._statusCur] || "#5b8def";
    const risk = this._riskCur;

    // Compute the 9 deformed points
    const pts = this._computePoints(cx, cy, baseR, risk);

    // ── FLAG pulse glow halo ───────────────────────────────────────────────
    if (this._flagCur) {
      const pulse = 0.5 + 0.5 * Math.sin(this._pulsePhase);
      const grad = ctx.createRadialGradient(cx, cy, baseR * 0.3, cx, cy, baseR * 1.45);
      grad.addColorStop(0, "rgba(234,84,85,0)");
      grad.addColorStop(0.6, `rgba(234,84,85,${0.08 * pulse})`);
      grad.addColorStop(1, `rgba(234,84,85,${0.3 * pulse})`);
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(cx, cy, baseR * 1.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // ── Inner connection lines ─────────────────────────────────────────────
    for (let li = 0; li < INNER_LINES.length; li++) {
      const [a, b] = INNER_LINES[li];
      const lit = this._lineLight[li];
      const alpha = 0.12 + lit * 0.45;
      ctx.beginPath();
      ctx.moveTo(pts[a].x, pts[a].y);
      ctx.lineTo(pts[b].x, pts[b].y);
      // Lit lines use the status color; dim lines use muted steel
      const lineColor = lit > 0.1 ? color : "#3a4a5c";
      ctx.strokeStyle = this._hexAlpha(lineColor, alpha);
      ctx.lineWidth = 1 + lit * 1.5;
      ctx.stroke();
    }

    // ── Outer ring (closed polygon through the 9 deformed points) ─────────
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < 9; i++) ctx.lineTo(pts[i].x, pts[i].y);
    ctx.closePath();
    // Filled region — very subtle fill
    const fillAlpha = 0.04 + risk * 0.08;
    ctx.fillStyle = this._hexAlpha(color, fillAlpha);
    ctx.fill();
    // Stroke
    ctx.strokeStyle = this._hexAlpha(color, 0.7);
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // ── Per-point glows and dots ───────────────────────────────────────────
    let maxPull = 0;
    let maxIdx = -1;
    for (let i = 0; i < 9; i++) {
      if (this._pull[i] > maxPull) { maxPull = this._pull[i]; maxIdx = i; }
    }

    for (let i = 0; i < 9; i++) {
      const p = pts[i];
      const pull = this._pull[i];
      const glow = this._glow[i];

      if (glow > 0.02) {
        // Radial glow proportional to pull
        const glowR = 6 + glow * 22;
        const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, glowR);
        grad.addColorStop(0, this._hexAlpha(color, 0.6 * glow));
        grad.addColorStop(1, this._hexAlpha(color, 0));
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(p.x, p.y, glowR, 0, Math.PI * 2);
        ctx.fill();
      }

      // Dot at each vertex
      const dotR = pull > 0.05 ? 3 + pull * 4 : 2.5;
      const dotAlpha = pull > 0.05 ? 0.9 : 0.35;
      ctx.beginPath();
      ctx.arc(p.x, p.y, dotR, 0, Math.PI * 2);
      ctx.fillStyle = this._hexAlpha(color, dotAlpha);
      ctx.fill();

      // Point index label (small, muted) — only for active or hovered
      if (pull > 0.1) {
        ctx.font = "8px monospace";
        ctx.fillStyle = this._hexAlpha(color, 0.5);
        ctx.fillText(String(i + 1), p.x + dotR + 2, p.y + 3);
      }
    }

    // ── Label on highest-pull point ────────────────────────────────────────
    if (maxIdx >= 0 && maxPull > 0.08) {
      const p = pts[maxIdx];
      const label = CUE_LABELS[maxIdx];
      ctx.font = "bold 10px monospace";
      ctx.fillStyle = color;
      // Position label away from center
      const angle = this._slotAngle(maxIdx);
      const lx = p.x + Math.cos(angle) * 14;
      const ly = p.y + Math.sin(angle) * 14;
      ctx.textAlign = "center";
      ctx.fillText(label, lx, ly);
      ctx.textAlign = "left";
    }

    // ── Risk % readout ─────────────────────────────────────────────────────
    const riskPct = Math.round(risk * 100);
    ctx.font = "11px monospace";
    ctx.fillStyle = "#7d8da3";
    ctx.textAlign = "center";
    ctx.fillText(`RISK  ${riskPct}%`, cx, H - 10);
    ctx.textAlign = "left";

    // ── Center dot ────────────────────────────────────────────────────────
    ctx.beginPath();
    ctx.arc(cx, cy, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = this._hexAlpha(color, 0.3);
    ctx.fill();
  }

  // ── Geometry helpers ───────────────────────────────────────────────────────

  /** Angle of slot i on the circle, starting from top (−π/2), going clockwise. */
  _slotAngle(i) {
    return -Math.PI / 2 + (i / 9) * Math.PI * 2;
  }

  /**
   * Compute deformed points:
   * base radius + pull[i] * risk * maxDeform pushed outward.
   */
  _computePoints(cx, cy, baseR, risk) {
    const maxDeform = baseR * 0.45; // max spike = 45% of base radius
    const pts = [];
    for (let i = 0; i < 9; i++) {
      const angle = this._slotAngle(i);
      const r = baseR + this._pull[i] * risk * maxDeform;
      pts.push({
        x: cx + Math.cos(angle) * r,
        y: cy + Math.sin(angle) * r,
      });
    }
    return pts;
  }

  /**
   * Convert a hex color + alpha (0..1) to rgba() string.
   * Accepts 6-digit hex like "#28c76f".
   */
  _hexAlpha(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha.toFixed(3)})`;
  }
}
