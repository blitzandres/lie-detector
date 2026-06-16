# Lie Detector — Implementation Research
> Completed March 30, 2026 | Consolidation update April 1, 2026

---

## Summary of Findings

All 6 implementation gaps resolved. **2 blockers discovered** that require decisions before coding.

Localhost remains the active deployment target for research use. Any Railway notes below are retained only as optional archival reference so the research context is not lost.

---

## BLOCKER 1: CrisperWhisper — RESOLVED → WhisperX (June 15, 2026)

**CrisperWhisper is CC-BY-NC-4.0 (non-commercial only) AND ~6–7 GB RAM (1.55B params).**

Two independent reasons rule it out for the default path: licensing *and* memory — on the 8 GB M1 it does
not fit the working budget (see `EXECUTION_ARCHITECTURE.md` Part A). The RAM constraint, not just licensing,
is decisive.

Options considered:
- **Option A (chosen):** Use **WhisperX / faster-whisper (int8)** (BSD-2, ~2.5–3 GB) — word timestamps +
  pause gaps. In the browser path use **whisper-tiny/base** (transformers.js / whisper.cpp-WASM).
- Option D (fold in): get filler detection on top of WhisperX via short-hesitation pause + filler-word
  text matching (recovers most of what CrisperWhisper's `[UH]/[UM]` gave us).
- Options B/C (contact Nyra for a commercial license / use CrisperWhisper temporarily) — **rejected**: even
  setting licensing aside, it does not fit 8 GB.

**DECISION:** WhisperX is the transcriber for local + cloud; whisper-tiny/base for browser. Filler detection
reimplemented on WhisperX (Option D). CrisperWhisper may only be used on the 24 GB cloud box if a specific
cue ever requires its verbatim fillers and accuracy justifies it.

> Note (pre-existing doc error to fix): `LIE_DETECTOR_BLUEPRINT.md` and `docs/CUE_CATALOG.md` library tables
> list CrisperWhisper as "Apache 2.0" — that is wrong; it is **CC-BY-NC-4.0**. Correct on next pass.

**RESOLVED (2026-06-16):** WhisperX (BSD-2) is the transcription path for later audio stages (also a RAM decision per EXECUTION_ARCHITECTURE §C.1). Stage 1 (Live Consensus Overlay) wires NO live transcription, so this does not block the current build.

---

## BLOCKER 2: AU28 (Jaw Tension) Not in OpenGraphAU

OpenGraphAU has 41 AUs but **does not include AU28** (lip suck/jaw tension). The model skips from AU27 to AU32.

Fallback for jaw tension detection: use MediaPipe Face Mesh landmark distances (jaw width ratio over time) — already in the blueprint as a custom approach.

**RESOLVED (2026-06-16):** Implemented as the `visual.jaw_tension` cue via a MediaPipe Face Mesh landmark-distance ratio (gonial landmarks 172↔397 normalized by the outer-eye-corner span 33↔263) in `blitz_overlay/cues/visual.py`. The browser computes the ratio; the engine scores its deviation from the rolling baseline. No OpenGraphAU dependency needed.

---

## Gap 1: CrisperWhisper Install

**Install method:** Clone from GitHub + install custom transformers fork

```bash
git clone https://github.com/nyrahealth/CrisperWhisper.git
cd CrisperWhisper
pip install -r requirements.txt
pip install git+https://github.com/nyrahealth/transformers.git@crisper_whisper
huggingface-cli login   # must accept CC-BY-NC-4.0 license on HF page first
```

**Key facts:**
- Only ONE model size: large-v3 equivalent (~1.55B params)
- RAM on CPU: ~6-7 GB (tight on 8 GB-class CPU boxes — use `batch_size=1`)
- Filler tokens output: `[UH]` and `[UM]` in transcript
- `utils.py` is NOT a package — must be on sys.path at runtime (copy into your project)
- Custom transformers fork is MANDATORY for correct word timestamps
- HuggingFace gated model: need `huggingface-cli login` or `HF_TOKEN` env var

**Minimal usage:**
```python
import sys
sys.path.insert(0, "/app/CrisperWhisper")   # utils.py lives here

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from utils import adjust_pauses_for_hf_pipeline_output

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "nyrahealth/CrisperWhisper",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)
processor = AutoProcessor.from_pretrained("nyrahealth/CrisperWhisper")
pipe = pipeline("automatic-speech-recognition", model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                chunk_length_s=30, batch_size=1,
                return_timestamps="word", torch_dtype=torch.float32)

raw = pipe("audio.wav")
result = adjust_pauses_for_hf_pipeline_output(raw)

# result["text"] — full transcript with [UH] / [UM] tokens
# result["chunks"] — list of {"text": str, "timestamp": (start_sec, end_sec)}
fillers = [c for c in result["chunks"] if c["text"].strip() in ("[UH]", "[UM]")]
```

**Known bugs:**
- `yaml.load()` error on PyYAML >= 6.0 — not relevant (this is the CrisperWhisper repo issue, not the library itself)
- Duplicate timestamps at 30-second chunk boundaries (open bug #41)
- Tensor errors on very short/silent clips
- No support for Nvidia 50-series GPUs yet

---

## Gap 2: OpenGraphAU Model Weights + Inference

**Install method:** Clone from GitHub (NOT on PyPI)

```bash
git clone https://github.com/lingjivoo/OpenGraphAU.git
cd OpenGraphAU
pip install -r requirements.txt
```

**Download weights:** Links in the README table on GitHub → Google Drive → put `.pth` files in `checkpoints/`

Recommended model: `OpenGprahAU-ResNet50_second_stage.pth` (good F1/speed balance)

**Required code patches (must apply before running):**

1. `conf.py` — fix PyYAML crash:
   ```python
   # Change: yaml.load(f)
   # To:
   yaml.load(f, Loader=yaml.FullLoader)
   ```

2. `ANFL.py` line 148 — fix Swin backbone download:
   ```python
   # Change: self.backbone = swin_transformer_base()
   # To:
   self.backbone = swin_transformer_base(pretrained=False)
   ```

**Minimal CPU inference:**
```python
import torch
from PIL import Image
from torchvision import transforms
import sys
sys.path.insert(0, '/app/OpenGraphAU')

from model.MEFARG import MEFARG
from utils import hybrid_prediction_infolist

device = torch.device('cpu')

net = MEFARG(num_main_classes=27, num_sub_classes=14,
             backbone='resnet50', neighbor_num=4, metric='dots')
ckpt = torch.load('checkpoints/OpenGprahAU-ResNet50_second_stage.pth', map_location='cpu')
net.load_state_dict(ckpt)
net.eval().to(device)

transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

img = Image.open('face.jpg').convert('RGB')
x = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    pred = net(x)

pred_np = pred.squeeze().cpu().numpy()   # 41-dim sigmoid vector
au_labels, au_probs = hybrid_prediction_infolist(pred_np, thresh=0.5)
```

**AU index map (the ones that matter for deception):**
| Index | AU | Meaning |
|---|---|---|
| 0 | AU1 | Inner brow raiser |
| 1 | AU2 | Outer brow raiser |
| 2 | AU4 | Brow lowerer |
| 4 | AU6 | Cheek raiser (genuine smile) |
| 6 | AU9 | Nose wrinkler (disgust) |
| 9 | AU12 | Lip corner puller (smile) |
| 17 | AU20 | Lip stretcher |
| 19 | AU23 | Lip tightener |
| 27–40 | AUL/R | Lateralized (left/right asymmetry) |
| — | **AU28** | **NOT PRESENT** — use landmark distance fallback |

**CPU performance:** ~1-5 FPS on ResNet-50. For video analysis this is fine (we process clips, not real-time frames).

---

## Gap 3: vitallens-python Usage

**Install:** `pip install vitallens` (simple PyPI package)

**System dependency:** `ffmpeg` must be on PATH

**Input format:**
- Video file path: `vl("clip.mp4")`
- Numpy array: `vl(array, fps=30.0)` where array is `(n_frames, H, W, 3)` RGB uint8
- Minimum: 16 frames; practical minimum for HRV: 20-30 seconds

**Methods available (no API key needed for local):**
- `method="pos"` — classical POS algorithm, fastest, CPU-only
- `method="chrom"` — CHROM algorithm, slightly more accurate
- `method=Method.VITALLENS` — neural model via API (best accuracy, requires free API key from rouast.com)

**Output:**
```python
from vitallens import VitalLens

vl = VitalLens(method="pos")          # or "chrom" for slightly better accuracy
results = vl("clip.mp4")

face = results[0]['vital_signs']
hr       = face['heart_rate']['value']        # BPM float
hr_conf  = face['heart_rate']['confidence']   # 0.0-1.0
ppg      = face['ppg_waveform']['value']      # numpy array — raw rPPG signal
hrv_sdnn = face['hrv_sdnn']['value']          # ms (may need API plan)
```

**Accuracy on compressed H.264 (YouTube-quality):**
- POS/CHROM: expect 3-8 bpm MAE degradation vs lab conditions
- VitalLens API method: MAE ~1.57 bpm (SOTA, tested on 422 subjects)
- Recommendation: use API method for production (free API key available), POS for offline/dev

**CPU speed:** ~2-5 seconds for a 30-second clip using POS/CHROM (classical signal processing, very fast)

---

## Gap 4: MMPose 133-Keypoint Setup

**Recommended approach: rtmlib** (lightweight ONNX wrapper, no mmcv/mmdet needed)

```bash
pip install rtmlib onnxruntime opencv-python
```

**Usage:**
```python
import cv2
from rtmlib import Wholebody

# Downloads ONNX model automatically on first run (~100 MB to ~/.cache/rtmlib/)
model = Wholebody(mode='balanced', backend='onnxruntime', device='cpu')
# mode: 'performance' (best accuracy), 'balanced' (recommended), 'lightweight' (fastest)

frame = cv2.imread('frame.jpg')
keypoints, scores = model(frame)
# keypoints: (N_persons, 133, 2) — x,y pixel coords
# scores:    (N_persons, 133)    — confidence 0..1
```

**Keypoint index map:**
```python
# Body (0-16)
NOSE           = 0
LEFT_SHOULDER  = 5    # shoulder tension / posture
RIGHT_SHOULDER = 6
LEFT_WRIST     = 9    # hand movement / gestures
RIGHT_WRIST    = 10
LEFT_HIP       = 11   # center of mass
RIGHT_HIP      = 12
LEFT_EAR       = 3    # head movement tracking
RIGHT_EAR      = 4

# Face landmarks (23-90) — 68 points
FACE_START     = 23
FACE_END       = 91   # keypoints[0][23:91]

# Hands (91-132)
LEFT_HAND      = slice(91, 112)   # 21 keypoints
RIGHT_HAND     = slice(112, 133)  # 21 keypoints

# Center of mass approximation
def center_of_mass(kpts):  # kpts shape (133, 2)
    torso = kpts[[5, 6, 11, 12]]  # both shoulders + both hips
    return torso.mean(axis=0)
```

**CPU performance:**
| Mode | Speed on CPU |
|---|---|
| lightweight | ~20-30 FPS |
| balanced (RTMW-m) | ~10-15 FPS |
| performance (RTMW-l) | ~3-5 FPS |

**Full MMPose path** (if you need more control): install via `pip install openmim && mim install mmengine mmcv mmdet mmpose` — but rtmlib is sufficient for this project.

---

## Gap 5: Optional Deployment Reference (Local-first plan)

**Active plan:** Run the backend locally first for research use.

**Optional free remote fallback:** Oracle Cloud Always Free or Hugging Face Spaces CPU.

**Archived paid reference:** Railway Hobby notes are preserved below in case the project scope changes later.

**Build approach: custom Dockerfile** (nixpacks is deprecated as of March 2026; railpack doesn't handle ML sys deps reliably)

**`Dockerfile`:**
```dockerfile
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg libsndfile1 libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

**`railway.toml`:**
```toml
[build]
builder = "DOCKERFILE"
dockerfilePath = "Dockerfile"

[deploy]
startCommand = "uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 75"
healthcheckPath = "/health"
healthcheckTimeout = 300
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3
```

**Model weights strategy for a managed host:** Download to a persistent volume instead of baking weights into the image
1. Dashboard → Service → Volumes → Add Volume, mount at `/app/models`
2. In FastAPI lifespan: check if weights exist before downloading
3. Never bake weights into Docker image (makes image 2-10 GB)

**Secrets:** Use env vars such as `CLAUDE_API_KEY`, `HF_TOKEN`, etc.

**Cold start mitigation:**
- `healthcheckTimeout = 300` (5 min) — models take time to load
- `--workers 1` — prevents OOM from multiplied model copies
- Load all models in FastAPI `lifespan` context on startup, not lazily

---

## Gap 6: Chrome Extension MV3 Video Capture

**API to use:** `chrome.tabCapture.getMediaStreamId()` (Chrome 116+)

**Architecture:**
```
User clicks extension icon
→ background.js (service worker): gets stream ID via tabCapture
→ sends stream ID to offscreen document via chrome.runtime.sendMessage()
→ offscreen.js: getUserMedia({chromeMediaSource: 'tab'}) → MediaRecorder
→ records 15s → onstop fires → base64 encode → POST to local FastAPI endpoint
  (or configured remote endpoint) → start next clip
```

**`manifest.json` permissions:**
```json
{
  "manifest_version": 3,
  "permissions": ["tabCapture", "activeTab", "offscreen"],
  "minimum_chrome_version": "116"
}
```

**Video output format:**
- Container: WebM
- Codec: VP9+Opus (primary), VP8+Opus (fallback)
- Chrome does NOT support MP4 from MediaRecorder — backend must accept WebM

**Critical gotchas:**
| Issue | Fix |
|---|---|
| Stream ID expires in seconds | Call `getUserMedia()` immediately after receiving message in offscreen.js |
| Stream ID is one-use only | Reuse the `MediaStream` object across clips — do NOT get a new stream ID per clip |
| Service worker sleeps | Keep all recording logic in offscreen.js (stays alive while recording) |
| One offscreen doc limit | Check `chrome.runtime.getContexts()` before creating |
| User gesture required | Must trigger from `chrome.action.onClicked` — cannot auto-start |
| Tab navigation kills stream | Monitor `mediaStream.oninactive` to detect and handle |

**Loop pattern (in offscreen.js):**
```javascript
function recordClip() {
  const chunks = [];
  const mimeType = 'video/webm;codecs=vp9,opus';
  mediaRecorder = new MediaRecorder(mediaStream, { mimeType });

  mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
  mediaRecorder.onstop = async () => {
    const blob = new Blob(chunks, { type: mimeType });
    await sendToServer(blob);   // base64 encode + POST
    recordClip();               // immediately start next clip
  };

  mediaRecorder.start();
  setTimeout(() => mediaRecorder.stop(), 15000);  // 15s clips
}
```

---

## Build Order (Confirmed — Matches Blueprint)

```
1. Linguistic module  — spaCy + VADER + NRCLex + TextDescriptives (all pip, no video)
2. Audio module       — librosa + Parselmouth + CrisperWhisper (testable on audio files)
3. Visual module      — MediaPipe blink + OpenGraphAU (+ AU28 landmark fallback) + rtmlib/MMPose
4. rPPG               — vitallens-python POS method (needs clip architecture first)
5. Integration        — assemble features, build Claude scoring prompt, wire VHS bars
6. Chrome Extension   — Phase 2, wrap backend in widget
```

---

## Dependency Install Summary

```bash
# Linguistic
pip install spacy vaderSentiment nrclex textdescriptives

# Audio
pip install librosa praat-parselmouth
git clone https://github.com/nyrahealth/CrisperWhisper.git
pip install git+https://github.com/nyrahealth/transformers.git@crisper_whisper

# Visual (body pose — easy path)
pip install rtmlib onnxruntime opencv-python

# Visual (face AUs)
git clone https://github.com/lingjivoo/OpenGraphAU.git
# + apply 2 patches (yaml.load fix + pretrained=False fix)
# + download ResNet50 Stage 2 weights from README Google Drive links

# rPPG
pip install vitallens  # + ffmpeg on PATH

# Backend
pip install fastapi uvicorn anthropic

# System deps (in Dockerfile)
# libgl1 libglib2.0-0 libsm6 libxext6 ffmpeg libsndfile1
```

---

*Research complete — ready to begin Phase 1 (linguistic module) from the consolidated repo plan*
