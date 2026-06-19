/**
 * Calibration — the active-calibration card. Shown while consensus.calibration.active is true.
 *
 * Hard gate (option 2): calibration won't complete until every channel that is producing signal
 * has a full baseline. This card shows the per-channel checklist + live guidance so the user
 * knows exactly which channels still need to be fed (talk, hold still, face the light).
 */
const FAM_ORDER = ["visual", "audio", "linguistic", "physio"];
const FAM_LABEL = { visual: "VISUAL", audio: "AUDIO", linguistic: "LINGUISTIC", physio: "PHYSIO" };
const HINTS = {
  visual: "look at the camera",
  audio: "speak to calibrate",
  linguistic: "keep talking",
  physio: "rPPG (camera, not a sensor) — hold still, good light; may not lock",
};

export class Calibration {
  constructor(el) {
    this.el = el;
    this._built = false;
    this._rows = {};
  }

  setConsensus(c) {
    const cal = c.calibration;
    if (!cal || !cal.active) { this.el.style.display = "none"; return; }
    this.el.style.display = "";
    if (!this._built) this._build();

    const pct = Math.round((cal.progress || 0) * 100);
    this._pct.textContent = `${pct}%`;
    this._bar.style.width = `${pct}%`;
    this._guid.textContent = cal.guidance || "";

    const fams = cal.families || {};
    for (const name of FAM_ORDER) {
      const row = this._rows[name];
      const f = fams[name];
      if (!row) continue;
      if (!f) { row.li.className = "calib-ch"; row.icon.textContent = "·"; row.count.textContent = "—"; row.hint.textContent = ""; continue; }
      const ready = f.status === "ready";
      row.li.className = `calib-ch ${f.status}`;
      row.icon.textContent = ready ? "✓" : "◌";
      row.count.textContent = `${f.ready}/${f.total}`;
      row.hint.textContent = ready ? "ready" : (HINTS[name] || "feed this channel");
    }
  }

  _build() {
    this.el.innerHTML = "";
    const head = document.createElement("div");
    head.className = "calib-head";
    head.innerHTML = 'CALIBRATING BASELINE <span class="calib-pct">0%</span>';
    this._pct = head.querySelector(".calib-pct");

    const barWrap = document.createElement("div");
    barWrap.className = "calib-barwrap";
    this._bar = document.createElement("div");
    this._bar.className = "calib-bar";
    barWrap.appendChild(this._bar);

    this._guid = document.createElement("div");
    this._guid.className = "calib-guidance";

    const ul = document.createElement("ul");
    ul.className = "calib-channels";
    for (const name of FAM_ORDER) {
      const li = document.createElement("li");
      li.className = "calib-ch";
      const icon = document.createElement("span"); icon.className = "calib-icon"; icon.textContent = "◌";
      const lab = document.createElement("span"); lab.className = "calib-label"; lab.textContent = FAM_LABEL[name];
      const count = document.createElement("span"); count.className = "calib-count"; count.textContent = "0/0";
      const hint = document.createElement("span"); hint.className = "calib-hint"; hint.textContent = "";
      li.append(icon, lab, count, hint);
      ul.appendChild(li);
      this._rows[name] = { li, icon, count, hint };
    }

    this.el.append(head, barWrap, this._guid, ul);
    this._built = true;
  }
}
