import { WebcamSource } from "./capture.js";
import { MediaPipeExtractor } from "./mediapipe-extractor.js";
import { RppgSampler } from "./rppg-sampler.js";
import { WsClient } from "./ws-client.js";
import { OverlayRenderer } from "./overlay-renderer.js";
import { Enneagram } from "./enneagram.js";

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
const renderer = new OverlayRenderer(canvas, panel);
const enneagram = new Enneagram(document.getElementById("enneagram"));
const wsUrl = `ws://${location.host}/ws`;
const ws = new WsClient(wsUrl, (c) => { renderer.setConsensus(c); enneagram.setConsensus(c); },
  (s) => { if (s === "engine-offline") panel.message.textContent = "Engine offline — reconnecting…"; });

panel.toggle.addEventListener("click", () => panel.body.classList.toggle("collapsed"));

async function start() {
  enneagram.start();
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
  ws.send(frame);
  renderer.draw(extractor.lastLandmarks);
  requestAnimationFrame(loop);
}

start();
