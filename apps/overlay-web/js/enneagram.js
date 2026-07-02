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
