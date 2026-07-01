/**
 * CueVerifier — the honest verdict line + convergence counter. The per-cue text checklist was
 * retired in favour of the Cue Polygon (+ hover); this just shows the verdict and the
 * convergence summary. It leads with CHANNELS (independent families) — the honest convergence
 * signal — not the raw cue count, which inflates with correlated face cues.
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
    this.verdict = els.verdict;
    this.convergence = els.convergence;
  }

  setConsensus(c) {
    const color = STATUS_COLORS[c.status] || "#888";
    this.verdict.textContent = VERDICT_TEXT[c.status] || c.status;
    this.verdict.style.color = color;

    const cv = c.convergence || {};
    const burst = cv.burst ? " · BURST" : "";
    // Channels first (honest convergence); raw cue count is secondary.
    this.convergence.textContent =
      `${cv.n_families || 0} channels · ${cv.n_lit || 0} cues firing${burst}`;
    this.convergence.style.color = cv.burst ? "#ea5455" : "#7d8da3";
  }
}
