import { REGION_LANDMARKS } from "./regions.js";

// Samples mean RGB of forehead + cheek ROIs from the live video, off-screen.
// Only the 3-number means are forwarded — never pixels (privacy, spec §3).
export class RppgSampler {
  constructor() {
    this.canvas = document.createElement("canvas");
    this.canvas.width = 64; this.canvas.height = 64;
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
  }

  _roiMean(video, landmarks, idxs) {
    let cx = 0, cy = 0, n = 0;
    for (const i of idxs) { const p = landmarks[i]; if (p) { cx += p.x; cy += p.y; n++; } }
    if (n === 0) return [0, 0, 0];
    cx /= n; cy /= n;
    const vw = video.videoWidth, vh = video.videoHeight;
    const boxW = vw * 0.12, boxH = vh * 0.08;
    const sx = Math.max(0, cx * vw - boxW / 2), sy = Math.max(0, cy * vh - boxH / 2);
    this.ctx.drawImage(video, sx, sy, boxW, boxH, 0, 0, 64, 64);
    const data = this.ctx.getImageData(0, 0, 64, 64).data;
    let r = 0, g = 0, b = 0;
    for (let i = 0; i < data.length; i += 4) { r += data[i]; g += data[i + 1]; b += data[i + 2]; }
    const px = data.length / 4;
    return [r / px, g / px, b / px];
  }

  sample(video, landmarks) {
    if (!landmarks) return null;
    return {
      forehead_rgb: this._roiMean(video, landmarks, REGION_LANDMARKS.forehead),
      cheek_rgb: this._roiMean(video, landmarks, [50, 280]),
    };
  }
}
