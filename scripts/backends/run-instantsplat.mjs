import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";

const inputDir = process.env.GS3D_INPUT_DIR || process.argv[2];
const outputDir = process.env.GS3D_OUTPUT_DIR || process.argv[3];
const jobId = process.env.GS3D_JOB_ID || `portrait-${Date.now()}`;
const instantSplatRoot = process.env.INSTANTSPLAT_ROOT;
const sceneName = process.env.INSTANTSPLAT_SCENE_NAME || `portrait_splat_forge_${jobId}`;
const quality = process.env.GS3D_QUALITY || "balanced";
const defaultIterationsByQuality = {
  draft: "500",
  balanced: "2000",
  high: "7000"
};
const iterations = process.env.INSTANTSPLAT_ITERATIONS || defaultIterationsByQuality[quality] || "1000";
const gpuId = process.env.INSTANTSPLAT_GPU_ID || "0";

if (!inputDir || !outputDir) {
  throw new Error("run-instantsplat requires GS3D_INPUT_DIR and GS3D_OUTPUT_DIR.");
}
if (!instantSplatRoot) {
  throw new Error("Set INSTANTSPLAT_ROOT to the local NVlabs/InstantSplat checkout.");
}
if (!fs.existsSync(instantSplatRoot)) {
  throw new Error(`INSTANTSPLAT_ROOT does not exist: ${instantSplatRoot}`);
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
    backend: "instantsplat",
    status: "running",
    quality,
    totalIterations: Number(iterations),
    ...patch,
    updatedAt: new Date().toISOString()
  }, null, 2), "utf8");
}

writeProgress({
  phase: "preparing",
  message: "Copying four views into InstantSplat",
  percent: 10,
  step: 1,
  totalSteps: 4
});

const sourceImageDir = path.join(inputDir, "images");
const datasetName = process.env.INSTANTSPLAT_DATASET || "examples";
const sceneRoot = path.join(instantSplatRoot, "assets", datasetName, sceneName);
const sceneImageDir = path.join(sceneRoot, "images");
const modelDir = path.join(instantSplatRoot, "output_infer", datasetName, sceneName, "4_views");
fs.mkdirSync(sceneImageDir, { recursive: true });
fs.mkdirSync(modelDir, { recursive: true });

for (const id of ["front", "left45", "right45", "back"]) {
  const src = path.join(sourceImageDir, `${id}.png`);
  if (!fs.existsSync(src)) {
    throw new Error(`Missing InstantSplat input image: ${src}`);
  }
  fs.copyFileSync(src, path.join(sceneImageDir, `${id}.png`));
}

function run(command, args, logPrefix, progressConfig = {}, cwd = instantSplatRoot) {
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
        INSTANTSPLAT_SCENE_NAME: sceneName
      }
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      const text = chunk.toString();
      stdout += text;
      fs.appendFileSync(stdoutPath, text, "utf8");
      progressConfig.onData?.(text);
    });
    child.stderr.on("data", (chunk) => {
      const text = chunk.toString();
      stderr += text;
      fs.appendFileSync(stderrPath, text, "utf8");
      progressConfig.onData?.(text);
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve();
        return;
      }
      reject(new Error(`InstantSplat ${logPrefix} exited with code ${code}.\n${stderr || stdout}`.trim()));
    });
  });
}

function pythonArgs(script, args) {
  const base = ["python"];
  if (process.env.INSTANTSPLAT_CONDA_ENV) {
    return ["run", "-n", process.env.INSTANTSPLAT_CONDA_ENV, ...base, script, ...args];
  }
  return [script, ...args];
}

async function runPython(script, args, logPrefix) {
  const progressConfig = {};
  if (logPrefix.includes("01_init_geo")) {
    writeProgress({
      phase: "initializing-geometry",
      message: "InstantSplat is estimating co-visible geometry and cameras",
      percent: 18,
      step: 2,
      totalSteps: 4
    });
    progressConfig.onData = (text) => {
      if (/finished|completed|done/i.test(text)) {
        writeProgress({ percent: 42, message: "Geometry initialization is finishing" });
      }
    };
  }
  if (logPrefix.includes("02_train")) {
    writeProgress({
      phase: "training",
      message: `Training 3D Gaussian Splatting: 0/${iterations}`,
      percent: 45,
      iteration: 0,
      step: 3,
      totalSteps: 4
    });
    progressConfig.onData = (text) => {
      if (!text.includes("Training progress")) return;
      const expectedTotal = Number(iterations);
      const matches = [...text.matchAll(/Training progress:[\s\S]*?(\d+)\/(\d+)/g)];
      const last = matches[matches.length - 1];
      if (!last) return;
      const iteration = Number(last[1]);
      const total = Number(last[2]) || Number(iterations);
      if (total !== expectedTotal || iteration > total) return;
      const ratio = total ? iteration / total : 0;
      writeProgress({
        phase: "training",
        message: `Training 3D Gaussian Splatting: ${iteration}/${total}`,
        percent: Math.min(94, Math.max(45, Math.round(45 + ratio * 49))),
        iteration,
        totalIterations: total,
        step: 3,
        totalSteps: 4
      });
    };
  }
  if (logPrefix.includes("03_render")) {
    writeProgress({
      phase: "rendering",
      message: "Rendering InstantSplat diagnostic views",
      percent: 95,
      step: 4,
      totalSteps: 4
    });
  }
  const envPrefix = process.platform === "win32" ? ["cmd", ["/c", "set", `CUDA_VISIBLE_DEVICES=${gpuId}`, "&&"]] : null;
  if (process.env.INSTANTSPLAT_CONDA_ENV) {
    await run("conda", ["run", "--no-capture-output", "-n", process.env.INSTANTSPLAT_CONDA_ENV, "python", script, ...args], logPrefix, progressConfig);
    return;
  }
  if (envPrefix) {
    await run(envPrefix[0], [...envPrefix[1], "python", script, ...args], logPrefix, progressConfig);
    return;
  }
  await run("python", [script, ...args], logPrefix, progressConfig);
}

const useLegacyScript = process.env.INSTANTSPLAT_USE_LEGACY_SCRIPT === "1";
const script = process.env.INSTANTSPLAT_SCRIPT || "scripts/run_infer.sh";
if (process.env.INSTANTSPLAT_CONDA_ENV) {
  process.env.CUDA_VISIBLE_DEVICES = gpuId;
}

if (useLegacyScript) {
  if (process.env.INSTANTSPLAT_CONDA_ENV) {
    await run("conda", ["run", "--no-capture-output", "-n", process.env.INSTANTSPLAT_CONDA_ENV, "bash", script], "instantsplat");
  } else {
    await run("bash", [script], "instantsplat");
  }
} else {
  await runPython(
    "init_geo.py",
    [
      "-s", sceneRoot,
      "-m", modelDir,
      "--n_views", "4",
      "--focal_avg",
      "--co_vis_dsp",
      "--conf_aware_ranking",
      "--infer_video"
    ],
    "instantsplat.01_init_geo"
  );
  await runPython(
    "train.py",
    [
      "-s", sceneRoot,
      "-m", modelDir,
      "-r", "1",
      "--n_views", "4",
      "--iterations", iterations,
      "--pp_optimizer",
      "--optim_pose"
    ],
    "instantsplat.02_train"
  );
  if (process.env.INSTANTSPLAT_RENDER_VIDEO === "1") {
    await runPython(
      "render.py",
      [
        "-s", sceneRoot,
        "-m", modelDir,
        "-r", "1",
        "--n_views", "4",
        "--iterations", iterations,
        "--infer_video"
      ],
      "instantsplat.03_render"
    );
  }
}

function listFilesRecursive(dir) {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const resolved = path.join(dir, entry.name);
    return entry.isDirectory() ? listFilesRecursive(resolved) : [resolved];
  });
}

const candidates = [
  modelDir,
  path.join(instantSplatRoot, "output"),
  path.join(instantSplatRoot, "outputs"),
  path.join(instantSplatRoot, "output_model"),
  path.join(instantSplatRoot, "model_output"),
  sceneRoot
].flatMap(listFilesRecursive);

const ply = candidates
  .filter((file) => file.toLowerCase().endsWith(".ply"))
  .sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs)[0];

if (!ply) {
  throw new Error("InstantSplat completed but no .ply file was found.");
}

fs.copyFileSync(ply, path.join(outputDir, "output.ply"));
writeProgress({
  status: "completed",
  phase: "completed",
  message: "InstantSplat PLY is ready",
  percent: 100,
  step: 4,
  totalSteps: 4
});
