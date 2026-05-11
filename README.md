# Portrait Splat Forge

Four-view portrait-to-GS3D prototype.

Japanese README: [README.ja.md](README.ja.md)

Third-party backend and model license notes: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

## Requirements

- Node.js 20.19+ or 22.x. The checked development runtime is Node.js 22.
- npm 10+
- Python 3.9+ for the bundled `rembg` mask helper
- Optional for video input: `ffmpeg` and `ffprobe` on `PATH`
- Optional for production 3DGS backends: CUDA, conda, model checkpoints, and
  backend-specific setup

## Quick Start

```bash
npm install
npm run setup
npm run start
```

Open:

```text
http://127.0.0.1:5173/
```

The app starts in local/mock mode. No OpenAI key or external 3DGS model is
required for the basic UI and lightweight preview.

## Environment

Create `.env` only when you need API keys or optional backend configuration:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Leave optional backend commands blank until each backend is installed locally.
Copying `.env.example` should not download or enable Apple SHARP, InstantSplat,
VGGT, gsplat, or GaussianObject by itself.

## Setup Details

JavaScript dependencies are locked by `package-lock.json`.

To do a first-time local setup and pre-download the two rembg segmentation
models used by the app:

```bash
npm run setup:all
```

`setup:all` installs npm packages, creates `.venv-rembg`, installs
`requirements-rembg.txt`, and warms up `u2net_human_seg` plus
`isnet-general-use`.

Python 3.9+ works for the bundled `rembg` environment. Python 3.10 or 3.11 is
recommended for newer external research backends.

External research backends such as Apple SHARP, InstantSplat, VGGT, gsplat, and
GaussianObject are intentionally not downloaded automatically. Their source
code, checkpoints, and model weights can have separate licenses, hardware
requirements, and CUDA/Python constraints. Configure them explicitly after
reading [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Current Pipeline

1. Upload one upper-body portrait.
2. Generate four staged views: front, left 45, right 45, back.
3. Build a splat on the server with either the local lightweight preview or a configured production GS3D backend.
4. Preview the point cloud in Three.js.
5. Export:
   - `.ply`
   - `.splat.json`
   - `.splat`
6. Persist generated artifacts under `outputs/splats`.

Video intake is wired for Quest Link / Air Link with PC GPU generation, not standalone Quest. Upload a video, analyze it with `ffprobe`, extract frames with `ffmpeg`, assign sampled frames to the four-view slots, and launch a staged build. This is a frame-extraction bridge into the existing static GS backends, not a full dynamic video-GS trainer yet.

Video presets:

- Video Draft: 1080p/1920 width, 60 frames, preview Auto, roughly 10-30 minutes after extraction.
- Video Standard: 1080p/1920 width, 120 frames, preview Dense, roughly 30-60+ minutes.
- Video High: 1440p/2560 width, 180 frames, preview Dense, roughly 1-2+ hours.
- Video Ultra: source width, 240 frames, preview Full, hours depending on the backend and GPU.

## Lightweight Splat

The lightweight splat is not a trained 3D Gaussian Splatting model yet. It samples the generated view images with `sharp`, estimates rough foreground/depth, rotates samples by the view yaw, and writes a colored Gaussian-style cloud with alpha and scale fields.

The viewer renders this as GPU Gaussian sprites using a custom Three.js shader, not plain square points.

This is useful for fast visual judgment before running a heavier backend such as GaussianObject, OpenSplat, gsplat, nerfstudio/splatfacto, or a human-specific reconstruction model.

## Production GS3D Backends

`/api/build-splat` now stages the four views into a reproducible job folder before calling an external backend:

```text
outputs/jobs/{jobId}/input/images/front.png
outputs/jobs/{jobId}/input/images/left45.png
outputs/jobs/{jobId}/input/images/right45.png
outputs/jobs/{jobId}/input/images/back.png
outputs/jobs/{jobId}/input/manifest.json
outputs/jobs/{jobId}/output/
```

Enable a backend by setting one of these commands before starting the server:

```bash
SHARP_COMMAND='node scripts/backends/run-sharp.mjs'
SHARP_CONDA_ENV=sharp
GAUSSIAN_OBJECT_COMMAND='python /path/to/GaussianObject/run_portrait_four_view.py --input "{inputDir}" --output "{outputDir}" --quality "{quality}"'
INSTANTSPLAT_COMMAND='node scripts/backends/run-instantsplat.mjs'
INSTANTSPLAT_ROOT=H:/InstantSplat
INSTANTSPLAT_CONDA_ENV=instantsplat
INSTANTSPLAT_ITERATIONS=
INSTANTSPLAT_RENDER_VIDEO=0
GS3D_FLIP_EXTERNAL_Y=1
FOUR_VIEW_FUSION_COMMAND='node scripts/backends/run-four-view-fusion.mjs'
OPEN_SPLAT_COMMAND='opensplat --input "{inputDir}" --output "{outputDir}/output.ply"'
GSPLAT_COMMAND='python /path/to/gsplat/train_four_view.py --manifest "{manifest}" --output "{outputDir}"'
CUSTOM_GS3D_COMMAND='your-command --input "{inputDir}" --output "{outputDir}"'
```

The command may use `{inputDir}`, `{outputDir}`, `{manifest}`, `{quality}`, and `{jobId}` placeholders. The same values are also provided as `GS3D_INPUT_DIR`, `GS3D_OUTPUT_DIR`, `GS3D_MANIFEST`, `GS3D_QUALITY`, and `GS3D_JOB_ID` environment variables.

The InstantSplat wrapper runs `init_geo.py` and `train.py` directly against the staged app images. It defaults to 500, 2000, or 7000 training iterations for draft, balanced, and high quality. Set `INSTANTSPLAT_ITERATIONS` to override that. Video rendering is skipped by default because the app needs the optimized `point_cloud.ply`; set `INSTANTSPLAT_RENDER_VIDEO=1` only when you also want InstantSplat's diagnostic render pass. External PLY previews are converted to the app's Y-up viewer convention by default; set `GS3D_FLIP_EXTERNAL_Y=0` only if a backend already exports in the viewer coordinate system.

On Windows with recent Visual Studio Build Tools, InstantSplat's CUDA extensions need a newer nvcc than the system CUDA 12.1 toolchain. The local setup uses conda `cuda-nvcc=12.8` and the helper `scripts/backends/build-instantsplat-extension.cmd` to compile:

```cmd
conda install -n instantsplat -c nvidia cuda-nvcc=12.8.93 cuda-cudart-dev=12.8 -y
scripts\backends\build-instantsplat-extension.cmd submodules/simple-knn
scripts\backends\build-instantsplat-extension.cmd submodules/fused-ssim
scripts\backends\build-instantsplat-extension.cmd submodules/diff-gaussian-rasterization
```

The backend should write one of:

```text
output.splat.json
output.ply
output.splat
last.ply
point_cloud.ply
```

Binary little-endian and ASCII PLY are both accepted for preview import. If the production PLY contains millions of Gaussians, the app preserves the original PLY in `outputs/splats` and down-samples only the browser preview. Tune that with `GS3D_PREVIEW_POINTS`.

Recommended research path for high-quality four-image output:

- Apple SHARP: best first replacement for the current fake/lightweight path when the source is a single portrait. It directly predicts a real 3D Gaussian PLY from one image and is much more honest than generated side-view fusion.
- InstantSplat: best next step when the four uploaded/generated views are genuinely high quality and mutually consistent. It estimates camera/geometry without a traditional SfM dependency, then optimizes 3DGS.
- VGGT + gsplat: highest-accuracy four-view path currently wired into this app. VGGT estimates cameras, depth, and COLMAP-compatible sparse geometry from the four masked views; gsplat then optimizes a real 3D Gaussian field and exports PLY.
- 4-view Visual Hull GS: best fallback for clean orthographic front/side/back character sheets when VGGT camera estimation collapses. It does not infer cameras; it fuses the four silhouettes and projects source colors, prioritizing recognizable shape over learned novel-view hallucination.
- GaussianObject: best match for exactly four object views. It uses visual hull initialization, floater elimination, and a diffusion-based repair stage.
- MVSplat/AnySplat: stronger feed-forward sparse-view direction when camera poses or unconstrained view sets are available, but they usually need model-specific setup and may be less portrait-object-specific out of the box.

### VGGT + gsplat backend

This backend uses two separate Python environments because VGGT and the gsplat example trainer require incompatible `pycolmap` APIs:

```bash
VGGT_GSPLAT_COMMAND=node scripts/backends/run-vggt-gsplat.mjs
VGGT_ROOT=/path/to/VGGT
VGGT_CONDA_ENV=vggt
GSPLAT_ROOT=/path/to/gsplat
GSPLAT_CONDA_ENV=gsplat-train
VGGT_USE_BA=0
VGGT_GSPLAT_SKIP_TRAIN=0
VGGT_GSPLAT_STEPS=
VGGT_POINT_FILTER=1
VGGT_FILTER_QUANTILE=
VGGT_FILTER_MAX_POINTS=
VGGT_GSPLAT_PROFILE=portrait
VGGT_GSPLAT_EXTRA_ARGS=
GS3D_APPLY_INPUT_MASK=1
GS3D_MASK_MODE=refined
GS3D_AUGMENT_VIEWS=1
GS3D_AUGMENT_VIEW_COUNT=4
```

Default optimization steps are `draft=1000`, `balanced=3000`, and `high=7000`. Set `VGGT_GSPLAT_STEPS` to override. The runner writes live phase and iteration progress to each job's `output/progress.json`, which is surfaced by the UI progress bar.

Input staging now writes `input/masks/*.png`, masked `input/images/*.png`, and four intermediate helper views under `input/augmented/images/*.png`. The helper views are deterministic blends between adjacent high-quality inputs, so VGGT receives eight views without asking the user for more files. Set `GS3D_AUGMENT_VIEWS=0` to disable this.

`GS3D_MASK_MODE=refined` keeps the largest central foreground component, closes small holes, removes loose background islands, and feathers the alpha before darkening RGB where the mask is low. Use `heuristic` for the older border-color mask.

`VGGT_POINT_FILTER=1` filters VGGT's COLMAP `points3D.txt` after export by robust radius/error statistics and rewrites image tracks that referenced removed points. This reduces floaters before gsplat sees the reconstruction. `VGGT_FILTER_QUANTILE` and `VGGT_FILTER_MAX_POINTS` override the quality-specific defaults.

`VGGT_GSPLAT_PROFILE=portrait` applies stricter sparse-view training defaults: pose optimization, random background, AbsGS-style densification for balanced/high, and small opacity/scale regularization. `VGGT_GSPLAT_EXTRA_ARGS` can append any raw `simple_trainer.py` options.

## Server API

```text
POST /api/upload
POST /api/upload-video
POST /api/analyze-video
POST /api/extract-video-frames
POST /api/generate-views
POST /api/build-splat
POST /api/save-splat
GET  /api/health
```

`/api/build-splat` is now the main handoff point for production backends. It receives generated view metadata, builds the selected backend output, saves `.ply`, `.splat.json`, `.splat`, and `.manifest.json`, and returns a preview splat for the browser.

Saved artifacts include:

```text
outputs/splats/{id}.ply
outputs/splats/{id}.splat.json
outputs/splats/{id}.splat
outputs/splats/{id}.manifest.json
```

## QA

With the app running:

```bash
npm run verify:render
```

This uploads `samples/mock-portrait.svg`, generates views, builds the splat, captures desktop/mobile screenshots under `outputs/qa`, and checks that the WebGL canvas is nonblank.

## Backend Selector

The UI lists available backends from `/api/health`:

```text
lightweight     -> current fast point cloud
sharp           -> Apple SHARP single-image Gaussian PLY
gaussianobject  -> external four-view GaussianObject-compatible command
instantsplat    -> InstantSplat sparse-view SfM-free 3DGS
vggtgsplat      -> VGGT camera/depth initialization plus gsplat optimization
fourviewfusion  -> camera-free four-view visual hull fusion
opensplat       -> external OpenSplat-compatible command
gsplat          -> external Python/torch training command
custom          -> any command matching the staged job contract
```

Backends are disabled until their command environment variable is set.

## OpenAI Image Mode

Set these environment variables before running the server:

```bash
OPENAI_API_KEY=
OPENAI_IMAGE_MODEL=gpt-image-2
```

The app defaults to Local mode. OpenAI mode must be selected in the UI before uploaded portraits are sent for image generation.

## License

Portrait Splat Forge is MIT licensed. Optional external backends and model
weights are governed by their own licenses. In particular, Apple SHARP model
weights are distributed under Apple's research-only model license and are not
covered by this repository's MIT License. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
