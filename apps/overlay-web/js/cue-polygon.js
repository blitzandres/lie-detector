/**
 * CuePolygon — the radial "Detailed Polygon" cue view (replaces the linear Cue Mixer).
 *
 * Every cue is a vertex on a static N-sided polygon (grouped by channel into arcs). When a cue
 * is lit, a light shoots from its vertex toward the centre. The centre brightens with how many
 * cues are firing together this moment (convergence.n_lit) — the synchrony signal — and flares
 * red + (via the existing bell) on an earned burst. Pure consumer of the consensus payload:
 * cue_rows {cue_id, family, z, lit} · convergence {n_lit, n_families, burst} · bell {ringing}.
 * No engine changes; scales from today's 22 cues toward ~300 (near-circle, labels drop off).
 */
const STATUS_COLORS = { CALIBRATING: "#5b8def", CLEAR: "#28c76f", WATCH: "#ff9f43", FLAG: "#ea5455" };
const FAMILY_ORDER = ["visual", "audio", "linguistic", "physio"];
const FAMILY_COLORS = { visual: "#5b8def", audio: "#28c76f", linguistic: "#ff9f43", physio: "#ea5455" };
const FAMILY_ABBR = { visual: "V", audio: "A", linguistic: "L", physio: "P", cbca: "C", body: "B" };
const SYNC_BRIGHT = 5;   // centre is fully bright at >= this many cues lit together
const LABEL_MAX = 40;    // show vertex labels only when there are at most this many cues

export class CuePolygon {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this._c = null;
    this._order = null;     // [{cue_id, family, label}] grouped by family
    this._lit = {};         // cue_id -> interpolated lit intensity 0..1
    this._beam = {};        // cue_id -> beam progress 0(vertex)..1(centre)
    this._centerGlow = 0;
    this._pulse = 0;
    this._running = false;
    this._verts = [];       // cached vertex screen positions for hover hit-testing

    // hover tooltip (abbreviated vertex labels → full cue id on hover)
    this._tip = document.createElement("div");
    this._tip.className = "polygon-tip";
    this._tip.style.display = "none";
    (canvas.parentElement || document.body).appendChild(this._tip);
    canvas.addEventListener("mousemove", (e) => this._onMove(e));
    canvas.addEventListener("mouseleave", () => { this._tip.style.display = "none"; });
  }

  _onMove(e) {
    const rect = this.canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    let best = null;
    let bd = 16;
    for (const v of this._verts) {
      const d = Math.hypot(v.x - mx, v.y - my);
      if (d < bd) { bd = d; best = v; }
    }
    if (!best) { this._tip.style.display = "none"; return; }
    this._tip.textContent = `${best.o.abbr} · ${best.o.cue_id}`;
    this._tip.style.display = "";
    this._tip.style.left = `${this.canvas.offsetLeft + mx + 12}px`;
    this._tip.style.top = `${this.canvas.offsetTop + my + 8}px`;
  }

  setConsensus(c) {
    this._c = c;
    if (c && c.cue_rows) this._ensureOrder(c.cue_rows);
  }

  start() {
    if (this._running) return;
    this._running = true;
    const tick = () => { if (!this._running) return; this._interp(); this._draw(); requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  }

  stop() { this._running = false; }

  // ── internal ──────────────────────────────────────────────────────────────

  _ensureOrder(rows) {
    if (this._order && this._order.length === rows.length) return;
    const byFam = {};
    for (const r of rows) (byFam[r.family] = byFam[r.family] || []).push(r);
    const fams = [...FAMILY_ORDER, ...Object.keys(byFam).filter((f) => !FAMILY_ORDER.includes(f))];
    const order = [];
    const famCount = {};
    for (const fam of fams) {
      for (const r of (byFam[fam] || [])) {
        famCount[fam] = (famCount[fam] || 0) + 1;
        const abbr = (FAMILY_ABBR[fam] || fam[0].toUpperCase()) + famCount[fam];
        order.push({ cue_id: r.cue_id, family: fam, label: r.label, abbr });
      }
    }
    this._order = order;
  }

  _interp() {
    const c = this._c;
    const EASE = 0.18;
    const conv = (c && c.convergence) ? c.convergence : {};
    const nLit = conv.n_lit || 0;
    const nFam = conv.n_families || 0;
    // Honest: the centre brightens mainly with independent CHANNELS (real convergence) — raw
    // cue count only nudges it, so many correlated face cues can't light the centre on their own.
    const targetGlow = Math.max(0, Math.min(1,
      0.8 * Math.min(1, nFam / 2) + 0.2 * Math.min(1, nLit / SYNC_BRIGHT)));
    this._centerGlow += (targetGlow - this._centerGlow) * 0.15;
    this._pulse += 0.06;

    const rows = (c && c.cue_rows) ? c.cue_rows : [];
    const litMap = {};
    const zMap = {};
    for (const r of rows) { litMap[r.cue_id] = r.lit; zMap[r.cue_id] = Math.abs(r.z || 0); }

    for (const o of (this._order || [])) {
      const lit = !!litMap[o.cue_id];
      const targetLit = lit ? Math.max(0.3, Math.min(1, (zMap[o.cue_id] || 0) / 6)) : 0;
      this._lit[o.cue_id] = (this._lit[o.cue_id] || 0) + (targetLit - (this._lit[o.cue_id] || 0)) * EASE;
      let p = this._beam[o.cue_id] || 0;
      p = lit ? (p + 0.045) % 1 : 0;       // shoot inward repeatedly while lit
      this._beam[o.cue_id] = p;
    }
  }

  _draw() {
    const cv = this.canvas;
    const cssW = cv.clientWidth || 300;
    const cssH = cv.clientHeight || 300;
    if (cv.width !== cssW) cv.width = cssW;
    if (cv.height !== cssH) cv.height = cssH;
    const ctx = this.ctx;
    const W = cssW;
    const H = cssH;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0b0f14";
    ctx.fillRect(0, 0, W, H);

    const order = this._order;
    const cx = W / 2;
    const cy = H / 2 + 6;
    const R = Math.min(W, H) * 0.40;
    const status = this._c ? this._c.status : "CALIBRATING";
    const color = STATUS_COLORS[status] || "#5b8def";

    if (!order || !order.length) {
      ctx.fillStyle = "#5b6675";
      ctx.font = "11px monospace";
      ctx.textAlign = "center";
      ctx.fillText("waiting for cue stream…", cx, cy);
      ctx.textAlign = "left";
      return;
    }

    const N = order.length;
    const ang = (i) => -Math.PI / 2 + (i / N) * Math.PI * 2;
    const vert = (i) => ({ x: cx + Math.cos(ang(i)) * R, y: cy + Math.sin(ang(i)) * R });

    // faint static polygon outline
    ctx.beginPath();
    for (let i = 0; i < N; i++) { const p = vert(i); i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y); }
    ctx.closePath();
    ctx.strokeStyle = "rgba(58,74,92,0.5)";
    ctx.lineWidth = 1;
    ctx.stroke();

    // beams: lit cues shoot light to the centre
    for (let i = 0; i < N; i++) {
      const o = order[i];
      const lit = this._lit[o.cue_id] || 0;
      if (lit < 0.04) continue;
      const v = vert(i);
      const fam = FAMILY_COLORS[o.family] || "#888";
      ctx.strokeStyle = this._a(fam, 0.10 + 0.25 * lit);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(v.x, v.y);
      ctx.lineTo(cx, cy);
      ctx.stroke();
      const p = this._beam[o.cue_id] || 0;
      const hx = v.x + (cx - v.x) * p;
      const hy = v.y + (cy - v.y) * p;
      const rad = 5 + 8 * lit;
      const g = ctx.createRadialGradient(hx, hy, 0, hx, hy, rad);
      g.addColorStop(0, this._a(fam, 0.9 * lit));
      g.addColorStop(1, this._a(fam, 0));
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(hx, hy, rad, 0, Math.PI * 2);
      ctx.fill();
    }

    // vertices (+ compact abbreviations like V1/A2/L3; full id on hover). Cache positions.
    this._verts = [];
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (let i = 0; i < N; i++) {
      const o = order[i];
      const v = vert(i);
      this._verts.push({ x: v.x, y: v.y, o });
      const lit = this._lit[o.cue_id] || 0;
      const fam = FAMILY_COLORS[o.family] || "#888";
      ctx.beginPath();
      ctx.arc(v.x, v.y, lit > 0.05 ? 2.5 + lit * 3 : 2, 0, Math.PI * 2);
      ctx.fillStyle = this._a(fam, lit > 0.05 ? 0.95 : 0.4);
      ctx.fill();
      if (N <= LABEL_MAX) {
        const la = ang(i);
        ctx.font = "8px monospace";
        ctx.fillStyle = this._a(fam, 0.4 + 0.5 * lit);
        ctx.fillText(o.abbr, v.x + Math.cos(la) * 12, v.y + Math.sin(la) * 12);
      }
    }
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";

    // centre synchrony glow
    const glow = this._centerGlow;
    const conv = this._c && this._c.convergence ? this._c.convergence : {};
    const ringing = this._c && this._c.bell && this._c.bell.ringing;
    const hot = conv.burst || ringing;
    const cCol = hot ? "#ea5455" : color;
    const pulse = ringing ? (0.6 + 0.4 * Math.sin(this._pulse)) : 1;
    const cr = R * 0.10 + glow * R * 0.34 * pulse;
    if (glow > 0.02 || ringing) {
      const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, cr);
      g.addColorStop(0, this._a(cCol, 0.5 + 0.5 * glow));
      g.addColorStop(0.6, this._a(cCol, 0.15 * glow));
      g.addColorStop(1, this._a(cCol, 0));
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(cx, cy, cr, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.beginPath();
    ctx.arc(cx, cy, 3 + glow * 3, 0, Math.PI * 2);
    ctx.fillStyle = this._a(cCol, 0.5 + 0.5 * glow);
    ctx.fill();

    // top readout: how many cues / channels firing together right now
    ctx.font = "bold 11px monospace";
    ctx.fillStyle = this._a(cCol, 0.6 + 0.4 * glow);
    ctx.textAlign = "center";
    ctx.fillText(`${conv.n_families || 0} channels · ${conv.n_lit || 0} cues`, cx, 13);
    ctx.textAlign = "left";
  }

  _a(hex, al) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${Math.max(0, Math.min(1, al)).toFixed(3)})`;
  }
}
