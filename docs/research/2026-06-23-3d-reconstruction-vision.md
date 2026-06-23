# Chapter — 3D Reconstruction ("Tesla-style vision") for Deception Detection

> Research + feasibility · 2026-06-23 · status: PLANNING / exploration (project on hold)
> The idea: give the computer a richer **3D "view"** of the subject — like Tesla turns camera
> feeds into a 3D world — so micro-movement and expression are captured in depth, not just 2D.

## The big honest takeaway first

**We already turn the video into a real-time 3D model of the face — we just haven't used it as 3D.**
MediaPipe FaceLandmarker gives us **478 landmarks with depth (x, y, z)**, a **facial transformation
matrix** (true head pose), and **52 blendshapes** — and those blendshapes ARE a learned 3D expression
*vector space* (a 3D-morphable-model-style decomposition). So the "computer sees the face in 3D" part
is **done**; today we only *read it as 2D* (we use blendshape scalars + a couple of pose angles).

**And the honest limit:** a better 3D "view" sharpens the *cues*, but it **cannot make a lie certain**.
Deception detection is fundamentally probabilistic — the science ceiling (~70–75%) is set by human
behaviour, not by perception fidelity. Even a perfect 3D twin of someone's face reads *cues*, which are
inherently noisy. So "with 3D it's impossible to hide a lie" is **not** true — 3D improves cue *quality*,
not certainty. (Locked honest framing: never a binary "LIE".) The strongest signal stays **content**
(the words), which the content engine already targets.

## What Tesla actually does (and why our regime differs)

Tesla FSD builds a 3D "vector space" from cameras: featurizer nets extract features → a **transformer
with spatial attention** projects every camera into one shared 3D frame and fuses them → **temporal
alignment** across frames → **voxel / occupancy / Signed-Distance-Field** prediction of the scene.
Vision-only, no LiDAR. ([thinkautonomous — occupancy networks](https://www.thinkautonomous.ai/blog/occupancy-networks/),
[notateslaapp — 3D world from pixels](https://www.notateslaapp.com/news/2773/how-teslas-fsd-builds-a-3d-world-from-pixels-part-3))

**Why we can't just copy it:** Tesla uses **8+ synchronized cameras + custom silicon (HW3/HW4) +
fleet-scale training**. We have **one webcam + an 8 GB M1**. The *technique* (2D → learned 3D) maps to a
face, but the *scale* doesn't. For a single face the right analogues are face-specific 3D models, not a
car-scale occupancy net.

## The face-3D reconstruction ladder (cheap → heavy)

| Tier | Tech | What it gives | Cost / on M1-8GB |
|---|---|---|---|
| **0 — HAVE IT** | **MediaPipe** face mesh (478 pts **+ z**), head-pose matrix, 52 blendshapes | a real-time 3D face + expression vector space, in-browser | ✅ already running, cheap |
| **1 — cheap next** | *use the 3D we already have*: per-landmark **depth** motion, true **3D head-pose dynamics** (we only use yaw/pitch/roll for one cue), **3D facial asymmetry**, depth-based micro-tremor | richer 3D cues, no new model | ✅ browser-side math only |
| **2 — FLAME** | statistical 3D head model (identity + expression + pose params) — the academic standard ([FLAME-Universe](https://github.com/TimoBolkart/FLAME-Universe)) | a clean parametric 3D face to fit | Python model, GPU-leaning |
| **3 — DECA / EMOCA / SPARK** | monocular → **detailed 3D face + expression**; **EMOCA** is emotion-aware (reconstructs emotional expression in 3D, CVPR 2022); **SPARK** (SIGGRAPH Asia 2024) pushes this to **real-time** ([EMOCA](https://github.com/radekd91/emoca), [SPARK](https://arxiv.org/html/2409.07984v1)) | the genuinely *better* 3D expression/AU signal vs 2D blendshapes | heavy Python, "after more RAM" |
| **4 — Gaussian Splatting avatars** | photoreal animatable head avatars from a **monocular webcam**, **>45 FPS** on consumer GPUs (explicit 3DGS, faster than NeRF) — Mono-Splat, SEGA, VOODOO-XP ([Mono-Splat](https://sciety.org/articles/activity/10.20944/preprints202512.2774.v1), [SEGA](https://arxiv.org/pdf/2504.14373)) | the literal "virtual-reality" photoreal twin | needs a real GPU + per-person fit; marginal on 8 GB |

## Does a fancier 3D view actually help detect lies?

- **Yes, modestly — it improves cue *quality*:** depth-aware micro-expressions, true 3D head pose,
  region motion in real 3D (not a flattened projection), robustness to camera angle. EMOCA-style
  emotion-aware 3D expression would give **more reliable AU/expression cues** than our 2D blendshapes —
  a real Tier-3 upgrade to the *visual* channel.
- **No, it doesn't change the ceiling:** the limiting factor is that behavioural cues are weak and
  probabilistic (Bond & DePaulo: humans ~54%; best automated ~60–75%). Better eyes → cleaner cues →
  maybe a few points; **not** certainty. And **content > expression** in the literature, so a photoreal
  3D face is *cool but low-ROI* for deception specifically.

## How it would slot into our engine

A "3D reconstruction engine" is **not a new architecture** — it's an **upgrade to the visual front-end**.
It would emit richer per-frame features (FLAME expression coefficients, 3D AU intensities, 3D
region-motion, depth tremor) that feed the **existing cue engine** through the same cue interface — i.e.,
swap MediaPipe-blendshape extraction for (or augment it with) a 3D-reconstruction extractor, behind the
same `CueDetector` contract. Everything downstream (synchrony, polygon, content fusion, honest framing)
is unchanged. This mirrors the swappable seams we already use (`Transcriber`, `ContentJudge`).

## Recommendation / roadmap placement

1. **Tier-1.5 (cheap, do first if pursued):** mine the **3D we already have** — depth-aware landmark
   motion, full 3D head-pose dynamics, 3D asymmetry. New cues, **no new model, no RAM**. This is the
   honest first step toward the user's "3D vision" and it directly fills the polygon.
2. **Tier-3 (after more RAM):** **EMOCA / SPARK** as the real 3D-expression upgrade to the visual channel
   — the version that actually improves cue quality.
3. **Tier-4 / optional:** **Gaussian-Splatting** photoreal avatar — impressive, but **low-ROI for lie
   detection** and needs a real GPU; pursue only if the 3D twin is a *product* goal, not an accuracy goal.

**Bottom line:** we already have the 3D model; the cheap win is *using* it more; the real upgrade
(EMOCA/SPARK) waits for more RAM; photoreal "VR" is a separate, low-ROI ambition. None of it changes the
honest rule — we surface deception-pattern *risk*, never a guaranteed lie.

## Sources
- Tesla vector space / occupancy: [thinkautonomous](https://www.thinkautonomous.ai/blog/occupancy-networks/) · [notateslaapp](https://www.notateslaapp.com/news/2773/how-teslas-fsd-builds-a-3d-world-from-pixels-part-3) · [opentools](https://opentools.ai/news/teslas-fsd-revolutionizes-autonomous-driving-with-vision-only-3d-worlds)
- Face 3D: [FLAME-Universe](https://github.com/TimoBolkart/FLAME-Universe) · [EMOCA (CVPR 2022)](https://github.com/radekd91/emoca) · [SPARK (SIGGRAPH Asia 2024)](https://arxiv.org/html/2409.07984v1)
- Gaussian splatting avatars: [Mono-Splat](https://sciety.org/articles/activity/10.20944/preprints202512.2774.v1) · [SEGA](https://arxiv.org/pdf/2504.14373) · [VOODOO-XP](https://arxiv.org/pdf/2405.16204)
