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
    if (c.flag) this._redPulse();
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
