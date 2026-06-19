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
    name.textContent = row.cue_id === "physio.heart_rate" ? "rPPG·cam" : row.label;
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
