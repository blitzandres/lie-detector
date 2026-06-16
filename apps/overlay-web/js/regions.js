// Region -> representative FaceMesh landmark indices (478-point model). Used to draw
// telestrator circles where a cue fired. Indices are stable across the canonical mesh.
export const REGION_LANDMARKS = {
  eyes: [33, 133, 362, 263],     // outer/inner corners L & R
  brow: [105, 334, 70, 300],     // brow ridge L & R
  mouth: [61, 291, 13, 14],      // mouth corners + lip center
  jaw: [172, 397, 152],          // gonial L/R + chin
  forehead: [10, 67, 297],       // forehead center + sides (rPPG ROI)
  head: [1],                     // nose tip
  body: [],                      // pose-driven (unused in v1 telestrator)
};

// Average a set of landmark {x,y} (normalized) into one canvas point.
export function regionCenter(region, landmarks, w, h) {
  const idxs = REGION_LANDMARKS[region] || [1];
  let sx = 0, sy = 0, n = 0;
  for (const i of idxs) {
    const p = landmarks[i];
    if (!p) continue;
    sx += p.x; sy += p.y; n++;
  }
  if (n === 0) return null;
  return { x: (sx / n) * w, y: (sy / n) * h };
}
