/**
 * Transcriber — live transcript source for the Linguistic family.
 *
 * Interface (the seam): start(), stop(), latest() -> { text, seq }, available, supported.
 * `seq` increments only when the rolling window text changes, so the engine can sample
 * linguistic cues per-utterance instead of per video frame.
 *
 * WebSpeechTranscriber wraps Chrome's webkitSpeechRecognition. HONEST FRAMING: Chrome
 * streams mic audio to Google for transcription — the only path by which audio leaves the
 * device. A fully-local LocalWhisperTranscriber can drop in behind this same interface
 * later with no engine change.
 */

const WINDOW_WORDS = 40;       // rolling window cap (~last 12s of speech)
const WINDOW_MS = 12000;       // drop words older than this

export class WebSpeechTranscriber {
  constructor() {
    this.supported = !!(window.SpeechRecognition || window.webkitSpeechRecognition);
    this.available = false;
    this._recog = null;
    this._words = [];           // [{ word, ts }]
    this._seq = 0;
    this._text = "";
  }

  start() {
    if (!this.supported) {
      console.warn("[Transcriber] Web Speech API unsupported — linguistic family disabled.");
      return;
    }
    try {
      const Ctor = window.SpeechRecognition || window.webkitSpeechRecognition;
      const r = new Ctor();
      r.continuous = true;
      r.interimResults = true;
      r.lang = "en-US";
      r.onresult = (e) => this._onResult(e);
      r.onerror = (e) => console.warn("[Transcriber] error:", e.error);
      r.onend = () => { if (this.available) { try { r.start(); } catch { /* already starting */ } } };
      r.start();
      this._recog = r;
      this.available = true;
    } catch (err) {
      console.warn("[Transcriber] start failed (non-fatal):", err.message);
      this.available = false;
    }
  }

  stop() {
    this.available = false;
    if (this._recog) { try { this._recog.stop(); } catch { /* noop */ } }
  }

  /** Latest rolling-window snapshot. seq advances only when the window text changes. */
  latest() {
    return { text: this._text, seq: this._seq };
  }

  _onResult(event) {
    const now = Date.now();
    // Collect the newest transcript fragment (interim or final) and split into words.
    let fragment = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      fragment += event.results[i][0].transcript + " ";
    }
    const newWords = fragment.trim().split(/\s+/).filter(Boolean);
    if (newWords.length === 0) return;

    // Append words and let the window cap bound growth (interim re-fires are bounded by the cap).
    for (const w of newWords) this._words.push({ word: w, ts: now });

    // Trim by age and count
    const cutoff = now - WINDOW_MS;
    this._words = this._words.filter((x) => x.ts >= cutoff);
    if (this._words.length > WINDOW_WORDS) {
      this._words = this._words.slice(this._words.length - WINDOW_WORDS);
    }

    const text = this._words.map((x) => x.word).join(" ");
    if (text !== this._text) {
      this._text = text;
      this._seq += 1;
    }
  }
}
