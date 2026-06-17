/**
 * AudioCapture — mic capture + lightweight Web Audio DSP features.
 *
 * Computes per-window scalars entirely in the browser (no server round-trip,
 * no audio data transmitted — consistent with the project's browser-does-signal
 * philosophy, spec §3 / READINESS #15).
 *
 * Features returned by latest():
 *   f0          — fundamental pitch (Hz) via autocorrelation; 0 if unvoiced/silent
 *   energy      — RMS of the time-domain buffer
 *   pause_ratio — fraction of recent ~2 s of energy frames below silence threshold
 *   tremor      — short-term f0 variability (normalized stddev of last N voiced f0
 *                 values); rough vocal-instability proxy
 *
 * Honest framing: these are real-time browser approximations computed from a
 * single AnalyserNode, not clinical-grade (Parselmouth / WhisperX come later).
 */

const FFT_SIZE = 2048;
const TICK_INTERVAL_MS = 50;     // analysis tick rate (~20 Hz)
const SILENCE_THRESHOLD = 0.01;  // RMS below this = silence frame
const ENERGY_HISTORY_MS = 2000;  // rolling window for pause_ratio
const F0_HISTORY_LEN = 20;       // voiced f0 values kept for tremor stddev

export class AudioCapture {
  constructor() {
    this.available = false;
    this._ctx = null;
    this._analyser = null;
    this._timeBuf = null;
    this._freqBuf = null;
    this._tickId = null;

    // Rolling energy history for pause_ratio: [{ts, energy}, ...]
    this._energyHistory = [];
    // Recent voiced f0 values for tremor
    this._f0History = [];

    // Latest computed features
    this._latest = { f0: 0, energy: 0, pause_ratio: 0, tremor: 0 };
  }

  /**
   * Request mic permission and start analysis.
   * Does NOT throw on denial — sets available=false and returns gracefully.
   */
  async start() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      this._ctx = new (window.AudioContext || window.webkitAudioContext)();
      this._analyser = this._ctx.createAnalyser();
      this._analyser.fftSize = FFT_SIZE;
      this._analyser.smoothingTimeConstant = 0.0; // no smoothing — we want raw frames

      const source = this._ctx.createMediaStreamSource(stream);
      source.connect(this._analyser);

      this._timeBuf = new Float32Array(this._analyser.fftSize);
      this._freqBuf = new Float32Array(this._analyser.frequencyBinCount);

      this.available = true;
      this._startTick();
    } catch (err) {
      // Permission denied or device unavailable — degrade silently
      console.warn("[AudioCapture] Mic unavailable:", err.message);
      this.available = false;
    }
  }

  /** Return latest feature snapshot. All values are 0 when unavailable. */
  latest() {
    if (!this.available) return { f0: null, energy: null, pause_ratio: null, tremor: null };
    return { ...this._latest };
  }

  // ── Internal ──────────────────────────────────────────────────────────────

  _startTick() {
    this._tickId = setInterval(() => this._tick(), TICK_INTERVAL_MS);
  }

  _tick() {
    if (!this._analyser) return;
    this._analyser.getFloatTimeDomainData(this._timeBuf);

    const energy = this._rms(this._timeBuf);
    const now = Date.now();

    // Rolling energy history for pause_ratio
    this._energyHistory.push({ ts: now, energy });
    const cutoff = now - ENERGY_HISTORY_MS;
    while (this._energyHistory.length && this._energyHistory[0].ts < cutoff) {
      this._energyHistory.shift();
    }
    const silentFrames = this._energyHistory.filter((e) => e.energy < SILENCE_THRESHOLD).length;
    const pauseRatio = this._energyHistory.length > 0
      ? silentFrames / this._energyHistory.length
      : 0;

    // Pitch via autocorrelation (only attempt when energy is above silence)
    let f0 = 0;
    if (energy >= SILENCE_THRESHOLD) {
      f0 = this._autocorrF0(this._timeBuf, this._ctx.sampleRate);
    }

    // f0 history for tremor (voiced frames only)
    if (f0 > 0) {
      this._f0History.push(f0);
      if (this._f0History.length > F0_HISTORY_LEN) this._f0History.shift();
    }
    const tremor = this._f0Stddev(this._f0History);

    this._latest = { f0, energy, pause_ratio: pauseRatio, tremor };
  }

  /** RMS of a Float32Array. */
  _rms(buf) {
    let sum = 0;
    for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
    return Math.sqrt(sum / buf.length);
  }

  /**
   * Estimate fundamental frequency via parabolic-interpolated autocorrelation.
   * Returns Hz or 0 if unvoiced / indeterminate.
   * Range: 80–600 Hz (human voice). Works on a mono Float32 frame at sampleRate.
   */
  _autocorrF0(buf, sampleRate) {
    const n = buf.length;
    const minLag = Math.floor(sampleRate / 600); // 600 Hz upper bound
    const maxLag = Math.floor(sampleRate / 80);  // 80 Hz lower bound

    // Compute unnormalized autocorrelation for lags in [minLag, maxLag]
    let bestCorr = -1;
    let bestLag = -1;
    for (let lag = minLag; lag <= maxLag; lag++) {
      let corr = 0;
      for (let i = 0; i < n - lag; i++) corr += buf[i] * buf[i + lag];
      if (corr > bestCorr) {
        bestCorr = corr;
        bestLag = lag;
      }
    }

    if (bestLag < minLag) return 0;

    // Parabolic interpolation for sub-sample accuracy
    const y0 = bestLag > minLag ? this._acorrAt(buf, n, bestLag - 1) : bestCorr;
    const y1 = bestCorr;
    const y2 = bestLag < maxLag ? this._acorrAt(buf, n, bestLag + 1) : bestCorr;
    const denom = 2 * (2 * y1 - y0 - y2);
    const refinedLag = denom !== 0
      ? bestLag + (y0 - y2) / denom
      : bestLag;

    // Confidence check: peak must be reasonably strong vs lag-0 energy
    const energy0 = this._acorrAt(buf, n, 0);
    if (energy0 < 1e-8 || bestCorr / energy0 < 0.2) return 0;

    return sampleRate / refinedLag;
  }

  _acorrAt(buf, n, lag) {
    let s = 0;
    for (let i = 0; i < n - lag; i++) s += buf[i] * buf[i + lag];
    return s;
  }

  /**
   * Normalized stddev of f0 history — a proxy for vocal tremor/jitter.
   * Normalised by mean so it's scale-invariant (returns 0..~1 range for typical voices).
   */
  _f0Stddev(hist) {
    if (hist.length < 3) return 0;
    const mean = hist.reduce((a, b) => a + b, 0) / hist.length;
    if (mean < 1) return 0;
    const variance = hist.reduce((s, v) => s + (v - mean) ** 2, 0) / hist.length;
    return Math.sqrt(variance) / mean; // coefficient of variation
  }
}
