/**
 * BellPlayer — plays an earned WebAudio chime on bell.just_rang and tracks a trust meter.
 *
 * Honest framing: the chime means "strong deception-pattern convergence," not "lie".
 * The trust meter = recent bell frequency (more bells in the window -> lower trust reading).
 */
const TRUST_WINDOW_MS = 60000;  // bells in the last minute drive the trust meter

export class BellPlayer {
  constructor() {
    this._ctx = null;
    this._bellTimes = [];   // timestamps (ms) of recent rings
  }

  /** Call every consensus frame with consensus.bell. */
  handle(bell) {
    if (bell && bell.just_rang) this._ring();
  }

  _ring() {
    try {
      if (!this._ctx) this._ctx = new (window.AudioContext || window.webkitAudioContext)();
      const ctx = this._ctx;
      const now = ctx.currentTime;
      // Two-tone chime (G5 -> C6), short decay.
      [784, 1047].forEach((freq, i) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = "sine";
        osc.frequency.value = freq;
        const t0 = now + i * 0.12;
        gain.gain.setValueAtTime(0.0001, t0);
        gain.gain.exponentialRampToValueAtTime(0.25, t0 + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.5);
        osc.connect(gain).connect(ctx.destination);
        osc.start(t0);
        osc.stop(t0 + 0.55);
      });
    } catch (err) {
      console.warn("[BellPlayer] chime failed (non-fatal):", err.message);
    }
    this._bellTimes.push(Date.now());
  }

  /** 0..1 trust reading: 1 = no recent bells, decreasing as bells accumulate. */
  trust() {
    const cutoff = Date.now() - TRUST_WINDOW_MS;
    this._bellTimes = this._bellTimes.filter((t) => t >= cutoff);
    // Each bell in the window knocks 20% off trust, floored at 0.
    return Math.max(0, 1 - this._bellTimes.length * 0.2);
  }

  bellCount() {
    const cutoff = Date.now() - TRUST_WINDOW_MS;
    return this._bellTimes.filter((t) => t >= cutoff).length;
  }
}
