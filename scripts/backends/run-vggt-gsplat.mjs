import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";

const inputDir = process.env.GS3D_INPUT_DIR || process.argv[2];
const outputDir = process.env.GS3D_OUTPUT_DIR || process.argv[3];
const jobId = process.env.GS3D_JOB_ID || `portrait-${Date.now()}`;
const quality = process.env.GS3D_QUALITY || "balanced";
const vggtRoot = process.env.VGGT_ROOT;
const gsplatRoot = process.env.GSPLAT_ROOT;
const vggtCondaEnv = process.env.VGGT_CONDA_ENV || "vggt";
const gsplatCondaEnv = process.env.GSPLAT_CONDA_ENV || "gsplat-train";
const gpuId = process.env.VGGT_GPU_ID || process.env.CUDA_VISIBLE_DEVICES || "0";
const manifestPath = process.env.GS3D_MANIFEST || path.join(inputDir || "", "manifest.json");

const defaultStepsByQuality = {
  draft: 1000,
  balanced: 3000,
  high: 7000
};
const gsplatSteps = Number(process.env.VGGT_GSPLAT_STEPS || defaultStepsByQuality[quality] || 3000);
const useBundleAdjustment = process.env.VGGT_USE_BA === "1" || process.env.VGGT_USE_BA === "true";
const skipGsplat = process.env.VGGT_GSPLAT_SKIP_TRAIN === "1" || process.env.VGGT_GSPLAT_SKIP_TRAIN === "true";
const pointFilterEnabled = process.env.VGGT_POINT_FILTER !== "0";
const gsplatProfile = process.env.VGGT_GSPLAT_PROFILE || "portrait";
const totalSteps = skipGsplat ? (pointFilterEnabled ? 4 : 3) : (pointFilterEnabled ? 5 : 4);

if (!inputDir || !outputDir) {
  throw new Error("run-vggt-gsplat requires GS3D_INPUT_DIR and GS3D_OUTPUT_DIR.");
}
if (!vggtRoot || !fs.existsSync(vggtRoot)) {
  throw new Error("Set VGGT_ROOT to the local facebookresearch/vggt checkout.");
}
if (!skipGsplat && (!gsplatRoot || !fs.existsSync(gsplatRoot))) {
  throw new Error("Set GSPLAT_ROOT to the local nerfstudio-project/gsplat checkout, or set VGGT_GSPLAT_SKIP_TRAIN=1.");
}

fs.mkdirSync(outputDir, { recursive: true });

function writeProgress(patch) {
  const progressPath = path.join(outputDir, "progress.json");
  let previous = {};
  if (fs.existsSync(progressPath)) {
    try {
      previous = JSON.parse(fs.readFileSync(progressPath, "utf8"));
    } catch {
      previous = {};
    }
  }
  fs.writeFileSync(progressPath, JSON.stringify({
    ...previous,
    backend: "vggtgsplat",
    status: "running",
    quality,
    totalIterations: skipGsplat ? 0 : gsplatSteps,
    ...patch,
    updatedAt: new Date().toISOString()
  }, null, 2), "utf8");
}

function listFilesRecursive(dir) {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const resolved = path.join(dir, entry.name);
    return entry.isDirectory() ? listFilesRecursive(resolved) : [resolved];
  });
}

function newestFile(files) {
  return files
    .filter((file) => fs.existsSync(file))
    .sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs)[0];
}

function readManifest() {
  if (!manifestPath || !fs.existsSync(manifestPath)) return {};
  try {
    return JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  } catch {
    return {};
  }
}

function parseExtraArgs(value) {
  if (!value) return [];
  const trimmed = value.trim();
  if (!trimmed) return [];
  if (trimmed.startsWith("[")) {
    const parsed = JSON.parse(trimmed);
    return Array.isArray(parsed) ? parsed.map(String) : [];
  }
  return trimmed.match(/"[^"]+"|'[^']+'|\S+/g)?.map((part) => part.replace(/^['"]|['"]$/g, "")) || [];
}

function gsplatTuningArgs() {
  const args = [];
  if (quality !== "draft") {
    args.push("--pose-opt", "--random-bkgd", "--strategy.absgrad", "--ssim-lambda", quality === "high" ? "0.24" : "0.20");
  }
  if (quality === "high") {
    args.push(
      "--strategy.grow-grad2d",
      "0.0008",
      "--strategy.prune-opa",
      "0.004",
      "--strategy.refine-stop-iter",
      String(Math.max(900, Math.min(gsplatSteps, 6500)))
    );
  }
  if (gsplatProfile === "portrait") {
    args.push("--opacity-reg", quality === "high" ? "0.0008" : "0.0005", "--scale-reg", quality === "high" ? "0.0006" : "0.0004");
  }
  args.push(...parseExtraArgs(process.env.VGGT_GSPLAT_EXTRA_ARGS));
  return args;
}

function run(command, args, logPrefix, progressConfig = {}, cwd = process.cwd()) {
  return new Promise((resolve, reject) => {
    const stdoutPath = path.join(outputDir, `${logPrefix}.stdout.log`);
    const stderrPath = path.join(outputDir, `${logPrefix}.stderr.log`);
    fs.writeFileSync(stdoutPath, "", "utf8");
    fs.writeFileSync(stderrPath, "", "utf8");
    const child = spawn(command, args, {
      cwd,
      shell: true,
      windowsHide: true,
      env: {
        ...process.env,
        CUDA_VISIBLE_DEVICES: gpuId,
        PYTHONIOENCODING: "utf-8",
        PYTHONUTF8: "1"
      }
    });
    let stdout = "";
    let stderr = "";
    const handleText = (text, logPath) => {
      fs.appendFileSync(logPath, text, "utf8");
      progressConfig.onData?.(text);
    };
    child.stdout.on("data", (chunk) => {
      const text = chunk.toString();
      stdout += text;
      handleText(text, stdoutPath);
    });
    child.stderr.on("data", (chunk) => {
      const text = chunk.toString();
      stderr += text;
      handleText(text, stderrPath);
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve({ stdout, stderr });
        return;
      }
      reject(new Error(`VGGT+gsplat ${logPrefix} exited with code ${code}.\n${stderr || stdout}`.trim()));
    });
  });
}

function pythonForEnv(envName, override) {
  if (override) return override;
  if (!envName || process.platform !== "win32") return "python";
  const condaRoot = process.env.CONDA_ROOT || process.env.CONDA_PREFIX?.replace(/\\envs\\[^\\]+$/i, "") || "C:\\Users\\shin\\anaconda3";
  const candidate = path.join(condaRoot, "envs", envName, "python.exe");
  return fs.existsSync(candidate) ? candidate : "python";
}

async function runPython(script, args, logPrefix, progressConfig, cwd, envName = vggtCondaEnv) {
  const override = envName === gsplatCondaEnv ? process.env.GSPLAT_PYTHON : process.env.VGGT_PYTHON;
  await run(pythonForEnv(envName, override), [script, ...args], logPrefix, progressConfig, cwd);
}

writeProgress({
  phase: "preparing",
  message: "Preparing masked and augmented scene for VGGT",
  percent: 10,
  step: 1,
  totalSteps
});

const sceneDir = path.join(outputDir, "vggt_scene");
const sceneImageDir = path.join(sceneDir, "images");
fs.mkdirSync(sceneImageDir, { recursive: true });

const manifest = readManifest();
const orderedInputs = [
  ...(manifest.views || []).map((view) => ({ id: view.id, file: view.file || `images/${view.id}.png` })),
  ...(manifest.augmentedViews || []).map((view) => ({ id: view.id, file: view.file }))
].filter((view) => view.id && view.file);
const inputs = orderedInputs.length
  ? orderedInputs
  : ["front", "left45", "right45", "back"].map((id) => ({ id, file: `images/${id}.png` }));

inputs.forEach((view, index) => {
  const src = path.join(inputDir, view.file);
  if (!fs.existsSync(src)) {
    throw new Error(`Missing VGGT input image: ${src}`);
  }
  fs.copyFileSync(src, path.join(sceneImageDir, `${String(index).padStart(2, "0")}_${view.id}.png`));
});

writeProgress({
  phase: "vggt",
  message: useBundleAdjustment ? "VGGT is estimating cameras, depth, and BA tracks" : "VGGT is estimating cameras and dense depth",
  percent: 18,
  step: 2,
  totalSteps
});

const vggtArgs = ["--scene_dir", sceneDir];
if (useBundleAdjustment) {
  vggtArgs.push("--use_ba", "--shared_camera", "--query_frame_num", "4", "--max_query_pts", "4096");
}
await runPython(
  path.join(vggtRoot, "demo_colmap.py"),
  vggtArgs,
  "vggt.01_colmap",
  {
    onData(text) {
      if (/Model loaded/i.test(text)) {
        writeProgress({ percent: 28, message: "VGGT model loaded; running inference" });
      }
      if (/Loaded .* images/i.test(text)) {
        writeProgress({ percent: 38, message: `VGGT loaded ${inputs.length} input views` });
      }
      if (/Converting to COLMAP format/i.test(text)) {
        writeProgress({ percent: 58, message: "VGGT is exporting COLMAP cameras and points" });
      }
      if (/Saving reconstruction/i.test(text)) {
        writeProgress({ percent: 68, message: "VGGT reconstruction is being saved" });
      }
    }
  },
  vggtRoot
);

const vggtPly = path.join(sceneDir, "sparse", "points.ply");
if (!fs.existsSync(vggtPly)) {
  throw new Error(`VGGT completed but did not create ${vggtPly}`);
}
fs.copyFileSync(vggtPly, path.join(outputDir, "vggt_points.ply"));

const sparseDir = path.join(sceneDir, "sparse");
await runPython(
  path.join(process.cwd(), "scripts", "backends", "write-colmap-text.py"),
  [sparseDir],
  "vggt.01b_colmap_text",
  {},
  vggtRoot
);

if (pointFilterEnabled) {
  writeProgress({
    phase: "filtering",
    message: "Filtering VGGT camera tracks and point-cloud outliers",
    percent: 69,
    step: 2,
    totalSteps
  });
  const filterArgs = [sparseDir, "--quality", quality, "--output-stats", path.join(outputDir, "vggt_filter_stats.json")];
  if (process.env.VGGT_FILTER_QUANTILE) {
    filterArgs.push("--quantile", process.env.VGGT_FILTER_QUANTILE);
  }
  if (process.env.VGGT_FILTER_MAX_POINTS) {
    filterArgs.push("--max-points", process.env.VGGT_FILTER_MAX_POINTS);
  }
  await runPython(
    path.join(process.cwd(), "scripts", "backends", "filter-colmap-points.py"),
    filterArgs,
    "vggt.01c_filter_points",
    {},
    vggtRoot
  );
}
const binarySparseDir = path.join(sparseDir, "_binary");
fs.mkdirSync(binarySparseDir, { recursive: true });
for (const name of ["cameras.bin", "images.bin", "points3D.bin"]) {
  const file = path.join(sparseDir, name);
  if (fs.existsSync(file)) {
    fs.renameSync(file, path.join(binarySparseDir, name));
  }
}

if (!skipGsplat) {
  writeProgress({
    phase: "training",
    message: `gsplat optimization: 0/${gsplatSteps}`,
    percent: 70,
    iteration: 0,
    totalIterations: gsplatSteps,
    step: pointFilterEnabled ? 4 : 3,
    totalSteps
  });

  const resultDir = path.join(outputDir, "gsplat_result");
  fs.mkdirSync(resultDir, { recursive: true });
  const trainerPath = path.join(gsplatRoot, "examples", "simple_trainer.py");
  const trainerArgs = [
    "default",
    "--disable-viewer",
    "--disable-video",
    "--data-dir",
    sceneDir,
    "--data-factor",
    "1",
    "--result-dir",
    resultDir,
    "--test-every",
    "99",
    "--max-steps",
    String(gsplatSteps),
    "--save-ply",
    "--ply-steps",
    String(gsplatSteps),
    "--eval-steps",
    String(gsplatSteps),
    "--save-steps",
    String(gsplatSteps),
    ...gsplatTuningArgs()
  ];

  await runPython(
    trainerPath,
    trainerArgs,
    "vggt.02_gsplat",
    {
      onData(text) {
        const matches = [...text.matchAll(/(\d+)\s*\/\s*(\d+)/g)];
        const last = matches[matches.length - 1];
        if (!last) return;
        const iteration = Number(last[1]);
        const total = Number(last[2]) || gsplatSteps;
        if (!Number.isFinite(iteration) || !Number.isFinite(total) || total < 1 || iteration > total) return;
        const ratio = iteration / total;
        writeProgress({
          phase: "training",
          message: `gsplat optimization: ${iteration}/${total}`,
          percent: Math.min(96, Math.max(70, Math.round(70 + ratio * 26))),
          iteration,
          totalIterations: total,
          step: pointFilterEnabled ? 4 : 3,
          totalSteps
        });
      }
    },
    gsplatRoot,
    gsplatCondaEnv
  );

  const trainedPly = newestFile(listFilesRecursive(path.join(resultDir, "ply")).filter((file) => file.toLowerCase().endsWith(".ply")));
  if (trainedPly) {
    fs.copyFileSync(trainedPly, path.join(outputDir, "output.ply"));
  } else {
    fs.copyFileSync(vggtPly, path.join(outputDir, "output.ply"));
  }
} else {
  fs.copyFileSync(vggtPly, path.join(outputDir, "output.ply"));
}

writeProgress({
  status: "completed",
  phase: "completed",
  message: skipGsplat ? "VGGT COLMAP point cloud is ready" : "VGGT + gsplat PLY is ready",
  percent: 100,
  step: totalSteps,
  totalSteps
});
