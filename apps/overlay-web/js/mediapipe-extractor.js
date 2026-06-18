import { SCHEMA_VERSION, USED_BLENDSHAPES } from "./schema.js";
import { FaceLandmarker, FilesetResolver } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

export class MediaPipeExtractor {
  constructor() { this.landmarker = null; this.lastLandmarks = null; }

  async init() {
    const fileset = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm");
    this.landmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    });
  }

  // Returns a feature frame (the WS payload) or a face_present:false frame.
  extract(video, tsMs) {
    const result = this.landmarker.detectForVideo(video, tsMs);
    const faces = result.faceLandmarks;
    if (!faces || faces.length === 0) {
      this.lastLandmarks = null;
      return { schema_version: SCHEMA_VERSION, ts: tsMs, face_present: false, confidence: 0 };
    }
    const landmarks = faces[0];
    this.lastLandmarks = landmarks;

    const blendshapes = {};
    const cats = (result.faceBlendshapes?.[0]?.categories) || [];
    for (const c of cats) {
      if (USED_BLENDSHAPES.includes(c.categoryName)) blendshapes[c.categoryName] = c.score;
    }

    const headPose = this._headPose(result.facialTransformationMatrixes?.[0]);
    const geometry = this._geometry(landmarks, blendshapes);

    return {
      schema_version: SCHEMA_VERSION,
      ts: tsMs,
      face_present: true,
      confidence: 0.9,
      blendshapes,
      head_pose: headPose,
      geometry,
    };
  }

  _headPose(matrix) {
    if (!matrix || !matrix.data) return { yaw: 0, pitch: 0, roll: 0 };
    const m = matrix.data; // column-major 4x4
    const yaw = Math.atan2(m[8], m[10]) * 180 / Math.PI;
    const pitch = Math.atan2(-m[9], Math.hypot(m[8], m[10])) * 180 / Math.PI;
    const roll = Math.atan2(m[1], m[5]) * 180 / Math.PI;
    return { yaw, pitch, roll };
  }

  _dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

  _geometry(lm, bs) {
    const eyeSpan = this._dist(lm[33], lm[263]) || 1e-6;
    const jawWidthRatio = this._dist(lm[172], lm[397]) / eyeSpan;
    const gx = ((bs.eyeLookOutLeft || 0) + (bs.eyeLookInRight || 0))
             - ((bs.eyeLookInLeft || 0) + (bs.eyeLookOutRight || 0));
    const gy = ((bs.eyeLookUpLeft || 0) + (bs.eyeLookUpRight || 0))
             - ((bs.eyeLookDownLeft || 0) + (bs.eyeLookDownRight || 0));
    return {
      jaw_width_ratio: jawWidthRatio,
      gaze_x: gx / 2,
      gaze_y: gy / 2,
      iris_ratio: this._irisRatio(lm),   // pupil-dilation proxy (null if iris landmarks absent)
    };
  }

  /**
   * Iris diameter ÷ eye width, averaged over both eyes — a pupil-dilation proxy.
   * Needs the 478-landmark (iris-refined) FaceLandmarker output (indices 468-477).
   * Returns null when iris landmarks are unavailable so the cue simply abstains.
   */
  _irisRatio(lm) {
    if (!lm || lm.length < 478) return null;
    const ringMean = (c, ring) => ring.reduce((s, i) => s + this._dist(lm[c], lm[i]), 0) / ring.length;
    const lIris = ringMean(468, [469, 470, 471, 472]) * 2;
    const rIris = ringMean(473, [474, 475, 476, 477]) * 2;
    const lEye = this._dist(lm[33], lm[133]) || 1e-6;   // left eye outer→inner corner
    const rEye = this._dist(lm[362], lm[263]) || 1e-6;  // right eye inner→outer corner
    return ((lIris / lEye) + (rIris / rEye)) / 2;
  }
}
