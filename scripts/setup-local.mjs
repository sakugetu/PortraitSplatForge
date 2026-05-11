import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = new Set(process.argv.slice(2));
const shouldInstallNode = args.has("--npm-install");
const shouldDownloadRembgModels = args.has("--download-rembg-models");
const venvDir = path.join(root, ".venv-rembg");
const isWindows = process.platform === "win32";
const venvPython = isWindows
  ? path.join(venvDir, "Scripts", "python.exe")
  : path.join(venvDir, "bin", "python");

function run(command, commandArgs, options = {}) {
  const executable = isWindows && command === "npm" ? "cmd" : command;
  const args = isWindows && command === "npm" ? ["/c", "npm", ...commandArgs] : commandArgs;
  console.log(`\n> ${command} ${commandArgs.join(" ")}`);
  const result = spawnSync(executable, args, {
    cwd: root,
    stdio: "inherit",
    shell: false,
    env: process.env,
    ...options
  });
  if (result.error) {
    console.error(result.error.message);
    process.exit(1);
  }
  if (result.status !== 0) {
    process.exit(result.status || 1);
  }
}

function findPython() {
  const candidates = process.env.PYTHON
    ? [[process.env.PYTHON, []]]
    : isWindows
      ? [["py", ["-3"]], ["python", []], ["python3", []]]
      : [["python3", []], ["python", []]];

  for (const [command, prefix] of candidates) {
    const result = spawnSync(command, [...prefix, "--version"], {
      cwd: root,
      shell: false,
      stdio: "ignore"
    });
    if (result.status === 0) {
      return { command, prefix };
    }
  }
  throw new Error("Python 3 was not found. Install Python 3.10+ and retry.");
}

if (shouldInstallNode) {
  run("npm", ["install"]);
}

const python = findPython();
if (!fs.existsSync(venvPython)) {
  run(python.command, [...python.prefix, "-m", "venv", venvDir]);
}

run(venvPython, ["-m", "pip", "install", "--upgrade", "pip"]);
run(venvPython, ["-m", "pip", "install", "-r", "requirements-rembg.txt"]);

if (shouldDownloadRembgModels) {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "portrait-splat-forge-"));
  const warmupImage = path.join(tmpDir, "rembg-warmup.png");
  const warmupMask = path.join(tmpDir, "rembg-warmup-mask.png");
  const createImage = [
    "from PIL import Image, ImageDraw",
    `img = Image.new("RGB", (96, 128), "white")`,
    "draw = ImageDraw.Draw(img)",
    "draw.ellipse((30, 12, 66, 48), fill=(30, 30, 30))",
    "draw.rectangle((24, 48, 72, 112), fill=(40, 160, 120))",
    `img.save(${JSON.stringify(warmupImage)})`
  ].join("; ");
  run(venvPython, ["-c", createImage]);
  for (const model of ["u2net_human_seg", "isnet-general-use"]) {
    run(venvPython, [
      "scripts/backends/remove-bg.py",
      warmupImage,
      warmupMask,
      "--model",
      model
    ]);
  }
  fs.rmSync(tmpDir, { recursive: true, force: true });
}

console.log("\nSetup complete.");
console.log(`Python mask environment: ${path.relative(root, venvPython)}`);
console.log("Research-only backends such as Apple SHARP are not downloaded automatically.");
console.log("Read THIRD_PARTY_NOTICES.md before downloading external model weights.");
