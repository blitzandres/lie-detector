/**
 * CueTimeline — the "Cue Mixer": a scrolling multi-track timeline of cue firings.
 *
 * Each cue is a horizontal lane (grouped by channel, like mixer tracks). Time runs left→right
 * with a NOW playhead at the right edge; every moment a cue is lit, a note is painted in its
 * lane at that time position. When many lanes light in the SAME vertical column across ≥2
 * channels (a synchrony burst), that column is highlighted red — you literally see the
 * cues "hit the same beat together." Engine-agnostic: it only buffers consensus.cue_rows +
 * consensus.convergence frames the engine already sends.
 */
const STATUS_COLORS = {
  CALIBRATING: "#5b8def", CLEAR: "#28c76f", WATCH: "#ff9f43", FLAG: "#ea5455",
};
const FAMILY_ORDER = ["visual", "audio", "linguistic", "physio"];
const FAMILY_COLORS = {
  visual: "#5b8def", audio: "#28c76f", linguistic: "#ff9f43", physio: "#ea5455",
};
const WINDOW_MS = 12000;   // visible time span
const GUTTER = 118;        // left label column
const FAM_TAG = { visual: "VIS", audio: "AUD", linguistic: "LNG", physio: "PHY" };

export class CueTimeline {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this._frames = [];     // [{t, rows:[{cue_id,z,lit}], burst}]
    this._lanes = null;    // [{cue_id, family, label}]
    this._laneIndex = {};  // cue_id -> lane row
    this._running = false;
    this._statusColor = "#888";
  }

  /** Buffer one consensus frame (call from the WS callback, ~10 Hz). */
  push(c) {
    const now = Date.now();
    if (!this._lanes && c.cue_rows && c.cue_rows.length) this._buildLanes(c.cue_rows);
    const rows = (c.cue_rows || []).map((r) => ({ cue_id: r.cue_id, z: r.z, lit: r.lit }));
    this._frames.push({ t: now, rows, burst: !!(c.convergence && c.convergence.burst) });
    const cutoff = now - WINDOW_MS;
    while (this._frames.length && this._frames[0].t < cutoff) this._frames.shift();
    this._statusColor = STATUS_COLORS[c.status] || "#888";
  }

  start() {
    if (this._running) return;
    this._running = true;
    const tick = () => { if (!this._running) return; this._draw(); requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  }

  stop() { this._running = false; }

  // ── internal ────────────────────────────────────────────────────────────────

  _buildLanes(rows) {
    const byFam = {};
    for (const r of rows) (byFam[r.family] = byFam[r.family] || []).push(r);
    const lanes = [];
    const fams = [...FAMILY_ORDER, ...Object.keys(byFam).filter((f) => !FAMILY_ORDER.includes(f))];
    for (const fam of fams) {
      for (const r of (byFam[fam] || [])) lanes.push({ cue_id: r.cue_id, family: fam, label: r.label });
    }
    this._lanes = lanes;
    this._laneIndex = {};
    lanes.forEach((l, i) => { this._laneIndex[l.cue_id] = i; });
  }

  _xForTime(t, now, W) {
    const age = Math.max(0, now - t);                 // 0..WINDOW
    return W - (age / WINDOW_MS) * (W - GUTTER);       // now→right edge, oldest→gutter
  }

  _draw() {
    const cssW = this.canvas.clientWidth || 800;
    const cssH = this.canvas.clientHeight || 190;
    if (this.canvas.width !== cssW) this.canvas.width = cssW;
    if (this.canvas.height !== cssH) this.canvas.height = cssH;
    const ctx = this.ctx;
    const W = cssW;
    const H = cssH;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0b0f14";
    ctx.fillRect(0, 0, W, H);

    if (!this._lanes || !this._lanes.length) {
      ctx.fillStyle = "#5b6675";
      ctx.font = "11px monospace";
      ctx.fillText("waiting for cue stream…", GUTTER, H / 2);
      return;
    }

    const now = Date.now();
    const lanes = this._lanes;
    const n = lanes.length;
    const top = 4;
    const laneH = (H - top - 2) / n;

    // Lane backgrounds + channel/cue labels (clean gutter: color band · 3-letter tag · clipped name)
    const fs = Math.max(7, Math.min(10, Math.floor(laneH - 2)));
    ctx.textBaseline = "middle";
    let lastFam = null;
    for (let i = 0; i < n; i++) {
      const y = top + i * laneH;
      const yc = y + laneH / 2;
      const lane = lanes[i];
      const famColor = FAMILY_COLORS[lane.family] || "#888";

      // lane background (zebra) + family color band on the far left
      ctx.fillStyle = (i % 2) ? "#0e141b" : "#11161d";
      ctx.fillRect(GUTTER, y, W - GUTTER, laneH);
      ctx.fillStyle = famColor;
      ctx.fillRect(0, y, 3, laneH);

      // group separator + 3-letter channel tag at each family's first lane
      if (lane.family !== lastFam) {
        if (i > 0) {
          ctx.strokeStyle = "#1f2733";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(0, y);
          ctx.lineTo(W, y);
          ctx.stroke();
        }
        ctx.font = `bold ${fs}px monospace`;
        ctx.fillStyle = famColor;
        ctx.fillText(FAM_TAG[lane.family] || lane.family.slice(0, 3).toUpperCase(), 8, yc);
        lastFam = lane.family;
      }

      // cue name, clipped so it never bleeds past the gutter into the note area
      ctx.font = `${fs}px monospace`;
      ctx.fillStyle = "#8595a8";
      this._fillClipped(ctx, lane.label, 36, yc, GUTTER - 40);
    }
    ctx.textBaseline = "alphabetic";

    // Per-frame column width (≈ one consensus tick)
    const colW = Math.max(2, (W - GUTTER) / (WINDOW_MS / 100));

    // Synchrony burst columns (drawn under the notes)
    for (const f of this._frames) {
      if (!f.burst) continue;
      const x = this._xForTime(f.t, now, W);
      ctx.fillStyle = "rgba(234,84,85,0.16)";
      ctx.fillRect(x - colW / 2, 0, colW, H);
    }

    // Notes: a lit cell per lit cue per frame
    for (const f of this._frames) {
      const x = this._xForTime(f.t, now, W);
      for (const r of f.rows) {
        if (!r.lit) continue;
        const i = this._laneIndex[r.cue_id];
        if (i === undefined) continue;
        const y = top + i * laneH;
        const intensity = Math.max(0, Math.min(1, Math.abs(r.z) / 6));
        const fam = lanes[i].family;
        ctx.fillStyle = this._alpha(FAMILY_COLORS[fam] || "#888", 0.35 + 0.65 * intensity);
        ctx.fillRect(x - colW / 2, y + 1, colW, laneH - 2);
      }
    }

    // NOW playhead
    ctx.strokeStyle = this._statusColor;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(W - 1, 0);
    ctx.lineTo(W - 1, H);
    ctx.stroke();
    ctx.fillStyle = this._statusColor;
    ctx.font = "8px monospace";
    ctx.fillText("NOW", W - 26, 9);
  }

  _fillClipped(ctx, text, x, y, maxW) {
    if (ctx.measureText(text).width <= maxW) { ctx.fillText(text, x, y); return; }
    let t = text;
    while (t.length > 1 && ctx.measureText(t + "…").width > maxW) t = t.slice(0, -1);
    ctx.fillText(t + "…", x, y);
  }

  _alpha(hex, a) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${a.toFixed(3)})`;
  }
}
