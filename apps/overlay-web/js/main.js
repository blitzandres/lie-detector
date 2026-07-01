import { WebcamSource } from "./capture.js";
import { MediaPipeExtractor } from "./mediapipe-extractor.js";
import { RppgSampler } from "./rppg-sampler.js";
import { AudioCapture } from "./audio-capture.js";
import { WebSpeechTranscriber } from "./transcriber.js";
import { WsClient } from "./ws-client.js";
import { OverlayRenderer } from "./overlay-renderer.js";
import { Enneagram } from "./enneagram.js";
import { CueVerifier } from "./cue-verifier.js";
import { BellPlayer } from "./bell.js";
import { CuePolygon } from "./cue-polygon.js";
import { Calibration } from "./calibration.js";
import { QaPanel } from "./qa-panel.js";

const video = document.getElementById("cam");
const canvas = document.getElementById("overlay");
const panel = {
  status: document.getElementById("status"),
  risk: document.getElementById("risk-fill"),
  agree: document.getElementById("agree"),
  message: document.getElementById("message"),
  voters: document.getElementById("voters"),
  toggle: document.getElementById("toggle"),
  body: document.getElementById("panel-body"),
};

const extractor = new MediaPipeExtractor();
const sampler = new RppgSampler();
const audio = new AudioCapture();
const transcriber = new WebSpeechTranscriber();
let _lastTranscriptSeq = -1;
const renderer = new OverlayRenderer(canvas, panel);
const enneagram = new Enneagram(document.getElementById("enneagram"));

const caption = document.getElementById("caption");
const sttNotice = document.getElementById("stt-notice");
document.getElementById("stt-toggle").addEventListener("click", () => {
  if (transcriber.available) {
    transcriber.stop();
    sttNotice.style.display = "none";
    caption.textContent = "";
  }
});
const cueVerifier = new CueVerifier({
  verdict: document.getElementById("verdict"),
  convergence: document.getElementById("convergence"),
});
const bellPlayer = new BellPlayer();
const cuePolygon = new CuePolygon(document.getElementById("cue-polygon"));
const calibration = new Calibration(document.getElementById("calibration"));
const trustEl = document.getElementById("trust");
let _sensitivity = 0;
document.getElementById("sens").addEventListener("input", (e) => {
  _sensitivity = parseFloat(e.target.value);
});

const wsUrl = `ws://${location.host}/ws`;
let qaPanel;
const ws = new WsClient(wsUrl, (c) => {
  if (c.type === "turn_result") { qaPanel.showResult(c); return; }
  renderer.setConsensus(c);
  enneagram.setConsensus(c);
  cueVerifier.setConsensus(c);
  cuePolygon.setConsensus(c);
  calibration.setConsensus(c);
  bellPlayer.handle(c.bell);
  trustEl.textContent =
    `trust: ${Math.round(bellPlayer.trust() * 100)}% · bells/min: ${bellPlayer.bellCount()}`;
},
  (s) => { if (s === "engine-offline") panel.message.textContent = "Engine offline — reconnecting…"; });

qaPanel = new QaPanel({
  question: document.getElementById("qa-question"),
  answer: document.getElementById("qa-answer"),
  start: document.getElementById("qa-start"),
  judge: document.getElementById("qa-judge"),
  fillRead: document.getElementById("qa-fill-read"),
  fillTrue: document.getElementById("qa-fill-true"),
  fillFalse: document.getElementById("qa-fill-false"),
  verdict: document.getElementById("qa-verdict"),
}, ws, () => (transcriber.available ? transcriber.latest().text : ""), () => Math.round(performance.now()));

panel.toggle.addEventListener("click", () => panel.body.classList.toggle("collapsed"));

async function start() {
  enneagram.start();
  cuePolygon.start();
  // Mic capture: independent try/catch — a denied mic must never kill the visual overlay.
  try {
    await audio.start();
    transcriber.start();
  } catch (err) {
    console.warn("[main] Mic start failed (non-fatal):", err.message);
  }
  if (!transcriber.available) sttNotice.style.display = "none";
  try {
    const source = new WebcamSource(video);
    await source.start();
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    await extractor.init();
    ws.connect();
    requestAnimationFrame(loop);
  } catch (err) {
    panel.message.textContent = `Camera unavailable: ${err.message}`;
  }
}

function loop() {
  const ts = Math.round(performance.now());
  const frame = extractor.extract(video, ts);
  if (frame.face_present && extractor.lastLandmarks) {
    frame.rppg = sampler.sample(video, extractor.lastLandmarks);
  }
  // Attach audio block when mic is live (nulls omitted so Python side sees absence cleanly)
  if (audio.available) {
    frame.audio = audio.latest();
  }
  // Attach transcript only when the window changed (per-utterance, not per frame).
  if (transcriber.available) {
    const t = transcriber.latest();
    if (t.seq !== _lastTranscriptSeq && t.text) {
      frame.transcript = t;
      _lastTranscriptSeq = t.seq;
    }
    // Display only the most recent ~16 words; CSS clamps the box to 2 lines max.
    caption.textContent = t.text.split(/\s+/).slice(-16).join(" ");
  }
  frame.config = { sensitivity: _sensitivity };
  ws.send(frame);
  renderer.draw(extractor.lastLandmarks);
  requestAnimationFrame(loop);
}

start();
