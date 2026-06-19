/**
 * QaPanel — the content-first Q&A interface. Operator sets a question, captures the answer
 * (live transcript or a dev script), and asks the engine to judge the turn. Renders the
 * content-primary verdict (scores + flagged phrases + content↔cue convergence).
 */
import { CALIBRATION_READING, TRUE_SCRIPT, FALSE_SCRIPT } from "./dev-scripts.js";

export class QaPanel {
  constructor(els, ws, getTranscript, getClock) {
    this.els = els;
    this.ws = ws;                  // WsClient (has .send)
    this.getTranscript = getTranscript;  // () => latest transcript text
    this.getClock = getClock;      // () => performance.now() rounded (same ts as frames)
    this._answerStartTs = null;

    els.start.addEventListener("click", () => this._startAnswer());
    els.judge.addEventListener("click", () => this._judge());
    els.fillTrue.addEventListener("click", () => { els.answer.value = TRUE_SCRIPT; });
    els.fillFalse.addEventListener("click", () => { els.answer.value = FALSE_SCRIPT; });
    els.fillRead.addEventListener("click", () => { els.answer.value = CALIBRATION_READING; });
  }

  _startAnswer() {
    this._answerStartTs = this.getClock();
    this.els.start.textContent = "● answering…";
  }

  _judge() {
    const t0 = this._answerStartTs ?? (this.getClock() - 8000);
    const t1 = this.getClock();
    const answer = this.els.answer.value.trim() || this.getTranscript();
    this.ws.send({ type: "turn", question: this.els.question.value, answer, t0, t1 });
    this.els.verdict.textContent = "judging…";
    this.els.start.textContent = "start answer";
    this._answerStartTs = null;
  }

  /** Called when a turn_result arrives over the WS. */
  showResult(r) {
    const s = (r.content && r.content.scores) ? r.content.scores : {};
    const flagged = ((r.content && r.content.flagged_phrases) || [])
      .map((f) => `“${f.text}” — ${f.reason}`).join("; ");
    const pct = Math.round((r.combined || 0) * 100);
    this.els.verdict.innerHTML =
      `<b>${r.label}</b> · ${pct}%` +
      (r.content_available
        ? `<br><span class="qa-scores">consistency ${fmt(s.consistency)} · richness ${fmt(s.richness_rm)} · ` +
          `verifiable ${fmt(s.verifiability)} · relevant ${fmt(s.relevance)}</span>` +
          (flagged ? `<br><span class="qa-flagged">${flagged}</span>` : "")
        : `<br><span class="qa-flagged">content layer offline — install/run Ollama for content analysis</span>`);
  }
}

function fmt(v) { return v == null ? "—" : Math.round(v * 100) + "%"; }
