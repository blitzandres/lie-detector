import { REGION_LANDMARKS } from "./regions.js";

// Samples mean RGB of forehead + cheek ROIs from the live video, off-screen.
// Only the 3-number means are forwarded — never pixels (privacy, spec §3).
export class RppgSampler {
  constructor() {
    this.canvas = document.createElement("canvas");
    this.canvas.width = 64; this.canvas.height = 64;
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
  }

  // Mean RGB of the ROI using ONLY skin-toned pixels (cheap YCbCr per-pixel mask — the light,
  // model-free version of "semantic skin segmentation"). Excludes hair/glasses/beard/shadow so
  // the rPPG pulse is cleaner. Returns { rgb, skin } where skin = fraction of pixels that were skin.
  _roiMean(video, landmarks, idxs) {
    let cx = 0, cy = 0, n = 0;
    for (const i of idxs) { const p = landmarks[i]; if (p) { cx += p.x; cy += p.y; n++; } }
    if (n === 0) return { rgb: [0, 0, 0], skin: 0 };
    cx /= n; cy /= n;
    const vw = video.videoWidth, vh = video.videoHeight;
    const boxW = vw * 0.12, boxH = vh * 0.08;
    const sx = Math.max(0, cx * vw - boxW / 2), sy = Math.max(0, cy * vh - boxH / 2);
    this.ctx.drawImage(video, sx, sy, boxW, boxH, 0, 0, 64, 64);
    const data = this.ctx.getImageData(0, 0, 64, 64).data;
    const px = data.length / 4;

    let sr = 0, sg = 0, sb = 0, skin = 0;   // skin-only accumulators
    let ar = 0, ag = 0, ab = 0;             // all-pixel fallback
    for (let i = 0; i < data.length; i += 4) {
      const R = data[i], G = data[i + 1], B = data[i + 2];
      ar += R; ag += G; ab += B;
      // YCbCr skin test (classic Cb∈[77,127], Cr∈[133,173]) — tone-robust, not perfect at extremes.
      const cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B;
      const cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B;
      if (cb >= 77 && cb <= 127 && cr >= 133 && cr <= 173) { sr += R; sg += G; sb += B; skin++; }
    }
    const frac = skin / px;
    if (skin >= px * 0.15) return { rgb: [sr / skin, sg / skin, sb / skin], skin: frac };
    return { rgb: [ar / px, ag / px, ab / px], skin: frac };  // too little skin → fall back, report low frac
  }

  sample(video, landmarks) {
    if (!landmarks) return null;
    const f = this._roiMean(video, landmarks, REGION_LANDMARKS.forehead);
    const c = this._roiMean(video, landmarks, [50, 280]);
    return {
      forehead_rgb: f.rgb,
      cheek_rgb: c.rgb,
      skin_fraction: (f.skin + c.skin) / 2,
    };
  }
}
