# Future Vision Engine

Reference-based visual inspection for fixtures and assembly stations. Pick one
golden frame (frame 0, frame N, or an external image) and every other frame or
picture is compared against it. The engine reports which components are
**missing**, **shifted** (deflection, in pixels), **changed** (damaged, wrong
part, flipped) and which **extra** objects appeared. Output uses the same folders,
CSV and JSON as `wisedo_fixture_vision_cpp`, plus extra columns.

Written in C++20 on OpenCV, with an optional DINOv2 backend through ONNX Runtime.
There is no Python at runtime.

```text
frame ─► register to reference ─► anomaly map ─────────────► per-component verdict ─► HUD / CSV / JSON / video
         (ORB + RANSAC,           classical: colour + SSIM     MISSING  board ratio / background test
          residual-verified)      dino:      DINOv2 patches    SHIFTED  template match, sub-pixel dx/dy
                                  hybrid:    both, fused       CHANGED  anomaly inside the slot
                                  calibrated per pixel         EXTRA    unexplained anomaly regions
```

---

## 1. Quick start

```bash
# macOS
brew install cmake opencv onnxruntime
# Ubuntu / Debian (ONNX Runtime: download the release tarball and pass -DCMAKE_PREFIX_PATH=<dir>)
sudo apt-get install -y build-essential cmake libopencv-dev

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure        # unit + end-to-end tests

# Inspect a video against its first frame (model-free, fastest)
./build/fve_inspect --video video.mp4 --backend classical --side-by-side
```

Results go to `inspection_results/`. Open `annotated_results.mp4`, or browse the
`corrected/` and `deviated/` folders.

### Best accuracy: add the DINOv2 model (one-time, offline)

```bash
python3 -m venv .venv && .venv/bin/pip install onnx
.venv/bin/python tools/prepare_model.py --variant reg-small --layers 5 8 --out models/dinov2_ml.onnx
```

This downloads DINOv2-with-registers ViT-S/14 (Apache-2.0, about 85 MB) and
exposes encoder layers 5 and 8 next to the last layer. Python is used only for
this step. Once `models/dinov2_ml.onnx` exists, `fve_inspect` picks it up
automatically and runs the **hybrid** backend. On Apple Silicon it uses CoreML;
elsewhere it runs on the CPU (`--provider cuda` for NVIDIA).

```bash
./build/fve_inspect --video video.mp4 --side-by-side --heatmap
```

### Docker (no local toolchain needed)

```bash
docker build -t future-vision-engine .                              # compiles, runs the tests, adds the DINOv2 model
docker build -t future-vision-engine --build-arg WITH_MODEL=0 .     # classical-only, smaller image

docker run --rm -v "$PWD:/work" future-vision-engine \
    --video /work/video.mp4 --output-dir /work/inspection_results --side-by-side --heatmap

docker compose run --rm inspect                                     # same, using docker-compose.yml
```

The image is multi-stage: the build stage compiles and runs `ctest`, so a failing
test fails the build. Python exists only in the model-export stage. The runtime
image holds just the binaries, the OpenCV and ONNX Runtime shared libraries and
the model, and runs as a non-root user. It builds for `linux/amd64` and
`linux/arm64`. Inside Docker the model runs on CPU (CoreML is macOS-only); use
the native build on a Mac for the fastest hybrid runs.

---

## 2. Generate output, step by step

### Option A: native binary (fastest; uses CoreML on Apple Silicon)

```bash
cd future-vision-engine

# 1. Build once (see Quick start for installing dependencies)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# 2. Optional, one-time: prepare the DINOv2 model for hybrid mode (best accuracy)
python3 -m venv .venv && .venv/bin/pip install onnx
.venv/bin/python tools/prepare_model.py --variant reg-small --layers 5 8 --out models/dinov2_ml.onnx

# 3. Put your video in the folder (any name), then run
./build/fve_inspect --video video.mp4 --output-dir inspection_results/run1 --side-by-side --heatmap

# 4. Open the results
open inspection_results/run1/annotated_results.mp4        # macOS; use xdg-open on Linux
open inspection_results/run1/deviated/                    # every frame that differs from the reference
```

### Option B: Docker (no compiler or OpenCV needed on the host)

```bash
cd future-vision-engine
docker build -t future-vision-engine .                    # once; compiles, runs the tests, adds the model

docker run --rm -v "$PWD:/work" future-vision-engine \
    --video /work/video.mp4 --output-dir /work/inspection_results/run1 --side-by-side --heatmap
```

Inside the container the folder is mounted at `/work`, so input and output paths
must start with `/work/`. Results appear on the host in
`inspection_results/run1/`. `docker compose run --rm inspect` does the same for
`./video.mp4`.

### What the console shows while it runs

```text
Engine            : backend=hybrid [dinov2 ... on coreml] segmentation=fixture components=24 ...
Frame 0024 (t=  1.00s) [CORRECT]  | 24/24 Correct | score 0.29 | All match reference
Frame 0025 (t=  1.04s) [DEVIATED] | 20/24 Correct | score 1.36 | 4 missing, 0 shifted, 0 changed, 3 extra
...
SUMMARY OF INSPECTION RESULTS ...
  [CORRECTED] Frame Ranges    : 0-24
  [DEVIATED]  Frame Ranges    : 25-239
```

`score` is the image-level anomaly score: below 1.0 means within normal
variation, 1.0 is the decision threshold, and higher means a stronger deviation.

### Reading the results

| Look at | To answer |
|---|---|
| Summary `Frame Ranges` | Which frames are correct vs deviated |
| `annotated_results.mp4` / `deviated/*.jpg` | Which parts, where: green OK, red MISSING, magenta SHIFTED (arrow = direction), orange CHANGED, amber EXTRA |
| `side_by_side/` | Reference and inspected frame next to each other |
| `heatmaps/` | Where the engine saw a difference (white outline = above threshold) |
| `frame_deviations.csv` | One row per frame for Excel: counts, missing / shifted / changed slot IDs, `max_shift_px`, `anomaly_score` |
| `frame_deviations.json` → `slot_deviations` | Per part: `dx`, `dy` (deflection in px), `ncc` (match quality), `anomaly`, `board_ratio` |
| `deviated_frames.json` → `frame_ranges` | Compact list such as `["25-239"]` for downstream systems |

### Choosing options

| Situation | Add |
|---|---|
| Reference is not the first frame | `--reference-frame 120` |
| You have a separate golden photo | `--reference-image golden.jpg` |
| Inspect photos instead of a video | `--input photos/` (first image = reference) or `--input part.jpg --reference-image golden.jpg` |
| No model, or maximum speed | `--backend classical` |
| Lighting changes during the video | keep the default hybrid (needs the model) |
| Too many / too few alarms | `--sensitivity 0.8` / `--sensitivity 1.3` |
| Small movements should be allowed | `--shift-tolerance 8` (pixels) |
| Parts are not on a tan fixture board | `--segmentation foreground`, or `none` for pure anomaly regions |
| Long static stretches | `--deduplicate` (one record per unchanged state, with its frame range) |
| Only numbers, no media | `--no-images --no-video` |
| Fixed slot positions, ignored areas | `--config configs/example.json` (see section 7) |

---

## 3. Common commands

| Goal | Command |
|---|---|
| First frame as reference | `fve_inspect --video line.mp4` |
| Frame 120 as reference | `fve_inspect --video line.mp4 --reference-frame 120` |
| External golden image | `fve_inspect --video line.mp4 --reference-image golden.jpg` |
| Compare one picture | `fve_inspect --input part.jpg --reference-image golden.jpg` |
| Folder of pictures (first image is reference) | `fve_inspect --input photos/` |
| Live camera / RTSP | `fve_inspect --input 0` or `--input rtsp://...` |
| Side-by-side + heat-maps | `--side-by-side --heatmap` |
| Collapse identical states | `--deduplicate` |
| Fixed slot geometry | `--config configs/example.json` |
| No predefined components (pure anomaly) | `--segmentation none` |
| Fewer false alarms / more sensitive | `--sensitivity 0.8` / `--sensitivity 1.3` |
| Learn real camera noise | `--calibration-frames 10` (the 10 frames after the reference must be good) |
| Analysis only, no media | `--no-images --no-video` |

Run `fve_inspect --help` for every flag. CLI flags override the config file.

---

## 4. Outputs (same layout as `wisedo_fixture_vision_cpp`)

```text
inspection_results/
├── corrected/frame_0000_CORRECT.jpg ...       annotated frames that match the reference
├── deviated/frame_0025_DEVIATED.jpg ...       annotated frames with deviations
├── side_by_side/{corrected,deviated}/         [Reference | Inspected]   (--side-by-side)
├── heatmaps/heat_frame_XXXX_DEVIATED.jpg      anomaly heat-map overlays (--heatmap)
├── annotated_results.mp4                      full annotated video
├── annotated_side_by_side.mp4                 (--side-by-side)
├── frame_deviations.csv / .json               every frame
├── corrected_frames.csv / .json               correct frames + frame ranges
└── deviated_frames.csv / .json                deviated frames + frame ranges
```

The first 14 CSV columns and the original JSON keys are unchanged. The added
fields are: `shifted_slots`, `changed_slots`, `max_shift_px`, `anomaly_score`
(1.0 is the decision threshold), `mean_diff`, `changed_pct`, `alignment_ok`,
`processing_ms`, and in JSON `slot_deviations` (per slot: `dx`, `dy`, `ncc`,
`anomaly`, `board_ratio`) and `extra_boxes` (`[x1, y1, x2, y2]`).

HUD colours: **green** OK · **red with cross** MISSING · **magenta** SHIFTED (a
dashed box marks the expected position, an arrow points to the actual one) ·
**orange** CHANGED · **amber** EXTRA.

---

## 5. How it works, and what came from where

| Stage | What it does | Source of the idea |
|---|---|---|
| Component discovery | Tan-board and white-part detection (ported unchanged, so `SLOT_xx` IDs match), generic foreground segmentation, or slots from config | wisedo_fixture_vision_cpp; UniVAD (component-level reasoning) |
| Registration | ORB + RANSAC similarity transform. The warp is applied only if it lowers the photometric residual, so camera shake or a re-seated fixture is not a defect | — |
| Classical map | Lightness normalisation, ±1 px shift-tolerant CIE76 ΔE, local SSIM | — |
| DINOv2 map | Patch features from several layers, each L2-normalised and fused | VisionAD (multi-layer integration), Dinomaly2 (layer averaging) |
| | Reference memory bank plus half-patch-shifted copies | VisionAD / AnomalyDINO (support augmentation) |
| | 1-NN cosine distance, positional neighbourhood (`--search-radius`, `-1` = global) | AnomalyDINO; UniVAD's component-aware matching |
| | Bilinear upsampling + Gaussian smoothing; image score = mean of top 1 % | Dinomaly2; AnomalyDINO |
| | Model runs on the board crop only (fewer tokens at the same detail) | — |
| Calibration | Per-pixel thresholds from the scorer's response to nuisance variants of the reference (sub-pixel shift, rotation, gain, noise, JPEG, blur) and optional good frames, so edges get higher thresholds than flat areas | Few-shot "normal-only" setting of all four papers |
| Hybrid fusion | Geometric mean of the classical and DINO normalised maps: classical gives pixel precision, DINO vetoes lighting changes | — |
| Verdicts | MISSING (board ratio or background similarity), SHIFTED (NCC template match with context padding, sub-pixel dx/dy, search radius below half the neighbour distance), CHANGED (anomaly inside slot), EXTRA (anomaly regions not explained by any slot) | wisedo_fixture_vision_cpp; UniVAD (logical/component anomalies) |

### How the four reference repositories are used

The engine re-implements ideas from four few-shot anomaly-detection papers in
C++. No code is copied: the originals are Python/PyTorch research code built to
evaluate on datasets like MVTec. Each idea was adapted to one reference frame,
a fixed camera and real-time video.

**[AnomalyDINO](https://github.com/dammsi/AnomalyDINO): the core detection method**
- *Idea:* DINOv2 patch features of the good image form a memory bank. Each test
  patch is scored by its cosine distance to the nearest bank patch. The image
  score is the mean of the top 1% of patch distances.
- *Here:* same backbone (DINOv2 ViT-S/14), the same 1-NN cosine distance
  (`src/core/dino.cpp`, `extract()` and `distanceMap()`) and the same top-1%
  image score (`topPercentMean()` in `src/core/anomaly.cpp`).
- *Adapted:* AnomalyDINO searches the whole image. Here each patch is compared
  only with nearby reference positions (`--search-radius 2`), which is faster
  and stops a missing part being "matched" to an identical neighbour.
  `--search-radius -1` restores the global search.

**[VisionAD](https://github.com/Qiqigeww/VisionAD): multi-layer features and support augmentation**
- *Idea:* fusing features from several transformer layers, and augmenting the
  few support (reference) images, makes plain nearest-neighbour search very strong.
- *Here:* `tools/prepare_model.py` exposes encoder layers 5 and 8 next to the
  last layer. `dino.cpp` L2-normalises each layer and concatenates them.
  The memory bank holds the reference plus four copies shifted by half a patch,
  so sub-patch misalignment is not mistaken for a defect.
- *Not used:* rotation and flip augmentation. With a fixed camera a rotated part
  is a defect, not a nuisance.

**[Dinomaly2](https://github.com/guojiajeremy/Dinomaly2): anomaly-map construction**
- *Idea:* per-layer maps of `1 - cosine` similarity, averaged across layers,
  bilinearly upsampled and Gaussian-smoothed.
- *Here:* the concatenated unit vectors are scaled by `1/sqrt(L)`, so a single
  dot product equals the mean per-layer cosine similarity. That is Dinomaly2's
  layer average at the cost of one comparison. The patch map is upsampled and
  smoothed the same way, which produces the images in `heatmaps/`.
- *Not used:* the trained reconstruction decoder. Training needs many good
  images per product, and this engine works from one reference.

**[UniVAD](https://github.com/FantasticGNU/UniVAD): component-aware reasoning**
- *Idea:* split the image into components, match patches within components,
  and model component relationships to catch *logical* anomalies (a missing,
  extra or misplaced part), not only texture defects.
- *Here:* `src/core/inspector.cpp` gives every component its own verdict:
  MISSING, SHIFTED with sub-pixel `dx/dy`, or CHANGED. Anomaly regions that no
  component explains become EXTRA. Positional patch matching is the
  fixed-camera form of UniVAD's component-aware matching.
- *Not used:* GroundingDINO + SAM segmentation and CLIP prompts, which are far
  too heavy for video. Components come from the fast fixture-board detector
  (ported from `wisedo_fixture_vision_cpp`) or a generic foreground segmentation.

**Added by this engine (not in any of the four):** per-pixel thresholds
calibrated from nuisance variants of the reference (the papers tune thresholds
on labelled test sets); the classical colour + SSIM scorer and its hybrid fusion
with DINO, which keeps in-place parts from being flagged under lighting changes;
camera-shake registration; sub-pixel shift measurement; the change gate; the
hybrid cascade; and the multithreaded video pipeline.

### Speed design

- Reader thread → parallel inspection workers → ordered writer. Image and video
  encoding run on background threads. In-flight frames are bounded, so memory
  stays flat.
- **Change gate:** a frame that is unchanged from the last inspected frame
  reuses its verdict. The gate compares 8×8 blocks, so even one small part
  appearing forces a full inspection.
- **Hybrid cascade:** if the classical map stays below 0.35 everywhere, the
  verdict cannot change, so DINO is skipped.
- Static ONNX shapes, CoreML / CUDA providers, `-O3 -mcpu=native`.

---

## 6. Measured results

On the 1280×720, 24 FPS, 240-frame test video (`video.mp4`), Apple M5, full
output (images + annotated video + side-by-side + heat-maps):

| Engine | Throughput | Frame verdicts |
|---|---|---|
| `wisedo_fixture_vision_cpp` (original) | 71 FPS | 0–24 correct, 25–239 deviated; presence + extra only; extra counts inflated (81 and 106 at frames 80 and 200) |
| this engine, `--backend classical` | about 180 FPS (about 540 FPS analysis-only) | 0–24 correct, 25–239 deviated, with missing / shifted / changed / extra per part |
| this engine, hybrid (default with model, CoreML) | about 57 FPS | same frame verdicts; also robust to the lighting change at frame 180 (classical flags 16 in-place parts as CHANGED there, hybrid 4) |

On the synthetic ground-truth video (`fve_synth`, 8 segments: missing, shifted,
extra, damaged, camera shake, lighting), both classical and hybrid match
154/154 checked frames.

### Known limits

- Pure `--backend dino` cannot resolve parts smaller than about 2 DINO patches
  (about 20 px at 1280 width). Use hybrid, the default, or raise `--dino-size`.
- With a fully rearranged layout, new parts that sit on a missing slot's
  footprint are counted under that slot rather than as separate EXTRAs.
- Per-slot labels close to a threshold can flip between consecutive frames on
  compressed video (for example MISSING ↔ SHIFTED). Frame-level verdicts are stable.
- The `classical` backend is sensitive to strong, uneven lighting changes; hybrid handles them.

---

## 7. Configuration

`configs/example.json` documents every key. Remove `"slots"` to auto-discover
components on the reference. Slot boxes use the original format
(`slot_id`, `type`, `bbox: [x1, y1, x2, y2]`), so existing
`wisedo_fixture_vision_cpp` configs load unchanged. Useful knobs:

- `ignore_regions` — areas never inspected (clocks, conveyors, people zones).
- `components.shift_tolerance_px` — allowed deflection (0 = auto: 10 % of the part's short side, at least 3 px).
- `anomaly.sensitivity`, `anomaly.margin`, `*_floor` — global decision levels.
- `dino.search_radius` — 2 for a fixed camera; `-1` for global matching when parts may legitimately move.

---

## 8. Layout

```text
include/fve/       public API: types, config, fixture, alignment, anomaly, inspector, output, pipeline
src/core/          implementation (one static library: fve_core)
src/app/main.cpp   CLI (fve_inspect)
tests/             behaviour tests on synthetic scenes + end-to-end ground-truth check
tools/             prepare_model.py (offline), fve_synth (test video), fve_probe (tuning aid)
configs/           example configuration
models/            ONNX models (git-ignored)
```

To use the engine as a library: link `fve_core`, create an `fve::Inspector`,
call `setReference(golden)`, then `inspect(frame, index, fps)`. `inspect` is
thread-safe.
