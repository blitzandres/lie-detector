// Mirror of blitz_overlay/schemas.py — keep SCHEMA_VERSION in sync.
export const SCHEMA_VERSION = "1.0";

// Blendshape coefficient names we forward (MediaPipe FaceLandmarker, 52 categories).
export const USED_BLENDSHAPES = [
  "eyeBlinkLeft", "eyeBlinkRight", "eyeWideLeft", "eyeWideRight",
  "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeLookDownLeft",
  "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeLookDownRight",
  "browInnerUp", "browDownLeft", "browDownRight",
  "mouthPressLeft", "mouthPressRight", "mouthPucker",
];
