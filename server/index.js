import express from "express";
import multer from "multer";
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { spawn } from "node:child_process";
import OpenAI from "openai";
import sharp from "sharp";

const root = process.cwd();
const outputDir = path.join(root, "outputs");
const uploadDir = path.join(outputDir, "uploads");
const videoDir = path.join(outputDir, "videos");
const viewDir = path.join(outputDir, "views");
const splatDir = path.join(outputDir, "splats");
const jobDir = path.join(outputDir, "jobs");
const sharpSequenceDir = path.join(outputDir, "sharp-sequences");

for (const dir of [outputDir, uploadDir, videoDir, viewDir, splatDir, jobDir, sharpSequenceDir]) {
  fs.mkdirSync(dir, { recursive: true });
}

function loadDotEnv() {
  const envPath = path.join(root, ".env");
  if (!fs.existsSync(envPath)) return;
  for (const rawLine of fs.readFileSync(envPath, "utf8").split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const separator = line.indexOf("=");
    if (separator < 1) continue;
    const key = line.slice(0, separator).trim();
    const value = line.slice(separator + 1).trim().replace(/^['"]|['"]$/g, "");
    if (process.env[key] === undefined) {
      process.env[key] = value;
    }
  }
}

loadDotEnv();

const app = express();
const upload = multer({ dest: uploadDir, limits: { fileSize: 50 * 1024 * 1024 } });
const videoUpload = multer({ dest: videoDir, limits: { fileSize: 1024 * 1024 * 1024 } });
const port = Number(process.env.PORT || 8787);
const buildJobs = new Map();

app.use(express.json({ limit: "10mb" }));
app.use("/outputs", express.static(outputDir));

const viewSpecs = [
  { id: "front", label: "Front", yaw: 0, prompt: "front-facing neutral bust portrait" },
  { id: "left45", label: "Left 45", yaw: -45, prompt: "left 45 degree bust portrait" },
  { id: "right45", label: "Right 45", yaw: 45, prompt: "right 45 degree bust portrait" },
  { id: "back", label: "Back", yaw: 180, prompt: "back view bust portrait, same clothing and hairstyle" }
];

const augmentedViewSpecs = [
  { id: "front_left22", label: "Front/Left 22", yaw: -22, from: "front", to: "left45", blend: 0.5 },
  { id: "left_back112", label: "Left/Back 112", yaw: -112, from: "left45", to: "back", blend: 0.58 },
  { id: "back_right112", label: "Back/Right 112", yaw: 112, from: "back", to: "right45", blend: 0.42 },
  { id: "front_right22", label: "Front/Right 22", yaw: 22, from: "front", to: "right45", blend: 0.5 }
];

const externalBackendDefs = {
  sharp: {
    label: "Apple SHARP",
    env: "SHARP_COMMAND",
    requiredViews: ["front"],
    description: "Single-image photorealistic Gaussian Splatting backend from Apple's SHARP model."
  },
  gaussianobject: {
    label: "GaussianObject four-view",
    env: "GAUSSIAN_OBJECT_COMMAND",
    requiredViews: ["front", "left45", "right45", "back"],
    description: "High-quality four-image object 3DGS backend based on GaussianObject-style visual hull + repair pipelines."
  },
  instantsplat: {
    label: "InstantSplat",
    env: "INSTANTSPLAT_COMMAND",
    requiredViews: ["front", "left45", "right45", "back"],
    description: "Sparse-view SfM-free Gaussian Splatting backend for high-quality real multi-view inputs."
  },
  vggtgsplat: {
    label: "VGGT + gsplat",
    env: "VGGT_GSPLAT_COMMAND",
    requiredViews: ["front", "left45", "right45", "back"],
    description: "VGGT camera/depth/point-map initialization followed by optional gsplat optimization."
  },
  fourviewfusion: {
    label: "4-view Visual Hull GS",
    env: "FOUR_VIEW_FUSION_COMMAND",
    requiredViews: ["front", "left45", "right45", "back"],
    description: "Camera-free four-view silhouette fusion for orthographic character sheets and high-fidelity manual views."
  },
  opensplat: {
    label: "OpenSplat CLI",
    env: "OPEN_SPLAT_COMMAND",
    requiredViews: ["front", "left45", "right45", "back"],
    description: "External OpenSplat or compatible command."
  },
  gsplat: {
    label: "gsplat trainer",
    env: "GSPLAT_COMMAND",
    requiredViews: ["front", "left45", "right45", "back"],
    description: "External gsplat/nerfstudio training command."
  },
  custom: {
    label: "Custom 4-view 3DGS",
    env: "CUSTOM_GS3D_COMMAND",
    requiredViews: ["front", "left45", "right45", "back"],
    description: "Any command that accepts the staged four-view job and writes PLY, .splat, or .splat.json."
  }
};

function availableSplatBackends() {
  return [
    { id: "lightweight", label: "Lightweight PLY", available: true, description: "Fast local Gaussian-style preview." },
    ...Object.entries(externalBackendDefs).map(([id, def]) => ({
      id,
      label: def.label,
      available: Boolean(process.env[def.env]),
      env: def.env,
      requiredViews: def.requiredViews || ["front", "left45", "right45", "back"],
      description: def.description
    }))
  ];
}

function publicPath(filePath) {
  return `/outputs/${path.relative(outputDir, filePath).replaceAll(path.sep, "/")}`;
}

function extensionFromPath(filePath) {
  const ext = path.extname(filePath || "").toLowerCase();
  return ext || ".png";
}

function outputPathFromPublicUrl(url) {
  if (!url?.startsWith("/outputs/")) {
    throw new Error(`Unsupported asset URL: ${url}`);
  }
  const relativePath = url.slice("/outputs/".length).replaceAll("/", path.sep);
  const resolved = path.resolve(outputDir, relativePath);
  if (!resolved.startsWith(path.resolve(outputDir))) {
    throw new Error("Asset path escaped output directory.");
  }
  return resolved;
}

function videoPathFromId(id) {
  if (!id) {
    throw new Error("Missing video id.");
  }
  const resolved = path.resolve(videoDir, path.basename(id));
  if (!resolved.startsWith(path.resolve(videoDir))) {
    throw new Error("Video path escaped output directory.");
  }
  return resolved;
}

function writeDataUrlImage(dataUrl, filePath) {
  if (!dataUrl) return null;
  const match = /^data:image\/(?:png|webp|jpeg);base64,([a-zA-Z0-9+/=]+)$/.exec(dataUrl);
  if (!match) {
    throw new Error("Manual keep mask must be a PNG, WebP, or JPEG data URL.");
  }
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, Buffer.from(match[1], "base64"));
  return filePath;
}

function rotateYaw(x, z, yawDegrees) {
  const yaw = (yawDegrees * Math.PI) / 180;
  const cos = Math.cos(yaw);
  const sin = Math.sin(yaw);
  return {
    x: x * cos + z * sin,
    z: -x * sin + z * cos
  };
}

const qualitySettings = {
  draft: { size: 256, step: 3, maxPoints: 60000 },
  balanced: { size: 384, step: 2, maxPoints: 180000 },
  high: { size: 512, step: 1, maxPoints: 420000 }
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function gs3dPreprocessSize(value = process.env.GS3D_PREPROCESS_SIZE) {
  return Math.round(clamp(Number(value || 1024), 512, 2048));
}

function smoothBounds(bounds) {
  const smoothed = bounds.map((bound) => ({ ...bound }));
  for (let pass = 0; pass < 2; pass += 1) {
    for (let y = 1; y < bounds.length - 1; y += 1) {
      if (!smoothed[y].valid) continue;
      const prev = smoothed[y - 1].valid ? smoothed[y - 1] : smoothed[y];
      const next = smoothed[y + 1].valid ? smoothed[y + 1] : smoothed[y];
      smoothed[y] = {
        valid: true,
        min: Math.round((prev.min + smoothed[y].min * 2 + next.min) / 4),
        max: Math.round((prev.max + smoothed[y].max * 2 + next.max) / 4)
      };
    }
  }
  return smoothed;
}

async function prepareView(view, quality = "balanced") {
  const imagePath = outputPathFromPublicUrl(view.url);
  if (!fs.existsSync(imagePath)) {
    throw new Error(`Missing view asset: ${view.url}`);
  }

  const settings = qualitySettings[quality] || qualitySettings.balanced;
  const size = settings.size;
  const { data } = await sharp(imagePath)
    .resize(size, size, { fit: "contain", background: { r: 0, g: 0, b: 0, alpha: 0 } })
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const corners = [
    [0, 0],
    [size - 1, 0],
    [0, size - 1],
    [size - 1, size - 1]
  ].map(([x, y]) => {
    const i = (y * size + x) * 4;
    return [data[i], data[i + 1], data[i + 2]];
  });
  const bg = corners.reduce((acc, color) => [acc[0] + color[0] / 4, acc[1] + color[1] / 4, acc[2] + color[2] / 4], [0, 0, 0]);

  function colorDistance(r, g, b) {
    return Math.hypot(r - bg[0], g - bg[1], b - bg[2]);
  }

  const mask = new Uint8Array(size * size);
  const bounds = Array.from({ length: size }, () => ({ valid: false, min: size, max: -1 }));

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const i = (y * size + x) * 4;
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const a = data[i + 3];
      const luma = (r + g + b) / 765;
      const differsFromBg = colorDistance(r, g, b) > 22;
      const isNearEmptyBlack = luma < 0.018 && colorDistance(r, g, b) < 16;
      const foreground = a > 20 && !isNearEmptyBlack && differsFromBg;
      if (!foreground) continue;
      mask[y * size + x] = 1;
      bounds[y].valid = true;
      bounds[y].min = Math.min(bounds[y].min, x);
      bounds[y].max = Math.max(bounds[y].max, x);
    }
  }

  return {
    ...view,
    data,
    mask,
    bounds: smoothBounds(bounds),
    size,
    step: settings.step,
    maxPoints: settings.maxPoints
  };
}

function samplePreparedView(prepared, index, envelopes) {
  const { data, mask, bounds, size, step } = prepared;
  const points = [];
  const isFront = prepared.id === "front";
  const isBack = prepared.id === "back";
  const isLeft = prepared.id === "left45";
  const isRight = prepared.id === "right45";

  function alphaAt(x, y) {
    if (x < 0 || x >= size || y < 0 || y >= size) return 0;
    return data[(y * size + x) * 4 + 3];
  }

  function rowEnvelope(y) {
    const frontBound = envelopes.front[y]?.valid ? envelopes.front[y] : bounds[y];
    const sideBound = envelopes.side[y]?.valid ? envelopes.side[y] : bounds[y];
    const halfWidth = Math.max(0.04, ((frontBound.max - frontBound.min + 1) / size) * 1.025);
    const halfDepth = Math.max(0.04, ((sideBound.max - sideBound.min + 1) / size) * 1.025);
    const centerX = ((frontBound.min + frontBound.max + 1) / 2 / size - 0.5) * 2.05;
    const centerZ = ((sideBound.min + sideBound.max + 1) / 2 / size - 0.5) * 2.05;
    return { halfWidth, halfDepth, centerX, centerZ };
  }

  for (let y = 0; y < size; y += step) {
    if (!bounds[y].valid) continue;
    const env = rowEnvelope(y);
    for (let x = 0; x < size; x += step) {
      if (!mask[y * size + x]) continue;
      const i = (y * size + x) * 4;
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const a = data[i + 3];
      const luma = (r + g + b) / 765;
      const nxRaw = (x / size - 0.5) * 2.05;
      const ny = -(y / size - 0.5) * 2.32;
      const edge =
        Math.abs(alphaAt(x + step, y) - alphaAt(x - step, y)) +
        Math.abs(alphaAt(x, y + step) - alphaAt(x, y - step));
      const edgeBoost = Math.min(1, edge / 510);

      let x3 = nxRaw;
      let z3 = 0;
      if (isFront || isBack) {
        const xLocal = clamp((nxRaw - env.centerX) / env.halfWidth, -1, 1);
        const curvedDepth = env.halfDepth * Math.sqrt(Math.max(0, 1 - xLocal * xLocal));
        const relief = (luma - 0.45) * env.halfDepth * 0.18;
        x3 = nxRaw;
        z3 = (isFront ? 1 : -1) * Math.max(0.015, curvedDepth + relief);
      } else {
        const zRaw = nxRaw;
        const zLocal = clamp((zRaw - env.centerZ) / env.halfDepth, -1, 1);
        const curvedWidth = env.halfWidth * Math.sqrt(Math.max(0, 1 - zLocal * zLocal));
        const relief = (luma - 0.45) * env.halfWidth * 0.18;
        x3 = (isRight ? 1 : -1) * Math.max(0.015, curvedWidth + relief) + env.centerX;
        z3 = zRaw;
      }

      const basePoint = {
        x: Number(x3.toFixed(4)),
        y: Number(ny.toFixed(4)),
        z: Number((z3 + index * 0.002).toFixed(4)),
        r,
        g,
        b,
        a,
        scale: Number((0.009 + edgeBoost * 0.006).toFixed(4)),
        view: prepared.id
      };
      points.push(basePoint);

      if (edgeBoost > 0.08) {
        const jitter = isFront || isBack ? { x: 0.0025, z: 0.006 * (isFront ? 1 : -1) } : { x: 0.006 * (isRight ? 1 : -1), z: 0.0025 };
        points.push({
          ...basePoint,
          x: Number((basePoint.x + jitter.x).toFixed(4)),
          z: Number((basePoint.z + jitter.z).toFixed(4)),
          scale: Number((basePoint.scale * 0.72).toFixed(4)),
          a: Math.max(120, Math.round(a * 0.76))
        });
      }
    }
  }

  if (points.length <= prepared.maxPoints) return points;
  const stride = Math.ceil(points.length / prepared.maxPoints);
  return points.filter((_point, pointIndex) => pointIndex % stride === 0);
}

async function buildLightweightSplat(views, quality = "balanced") {
  const usableViews = views.filter((view) => view.url && ["front", "left45", "right45", "back"].includes(view.id));
  if (usableViews.length < 4) {
    throw new Error("Four generated views are required before building a splat.");
  }

  const prepared = await Promise.all(usableViews.map((view) => prepareView(view, quality)));
  const byId = Object.fromEntries(prepared.map((view) => [view.id, view]));
  const envelopes = {
    front: byId.front?.bounds || byId.back?.bounds || prepared[0].bounds,
    side: byId.left45?.bounds || byId.right45?.bounds || byId.front?.bounds || prepared[0].bounds
  };
  const sampled = prepared.map((view, index) => samplePreparedView(view, index, envelopes));
  const points = sampled.flat();
  if (!points.length) {
    throw new Error("No usable foreground points were found.");
  }

  return {
    format: "portrait-splat-forge.lightweight-splat.v1",
    createdAt: new Date().toISOString(),
    quality,
    pointCount: points.length,
    sourceViews: usableViews.map(({ id, label, yaw, status, url }) => ({ id, label, yaw, status, url })),
    points
  };
}

function splatToPly(splat) {
  const header = [
    "ply",
    "format ascii 1.0",
    "comment Portrait Splat Forge lightweight Gaussian-style point cloud",
    `element vertex ${splat.points.length}`,
    "property float x",
    "property float y",
    "property float z",
    "property uchar red",
    "property uchar green",
    "property uchar blue",
    "property uchar alpha",
    "property float scale",
    "end_header"
  ].join("\n");
  const body = splat.points
    .map((point) => `${point.x} ${point.y} ${point.z} ${point.r} ${point.g} ${point.b} ${point.a} ${point.scale}`)
    .join("\n");
  return `${header}\n${body}\n`;
}

function splatToBinary(splat) {
  const rowSize = 32;
  const buffer = Buffer.alloc(splat.points.length * rowSize);
  splat.points.forEach((point, index) => {
    const offset = index * rowSize;
    const scale = Number(point.scale || 0.02);
    buffer.writeFloatLE(Number(point.x || 0), offset);
    buffer.writeFloatLE(Number(point.y || 0), offset + 4);
    buffer.writeFloatLE(Number(point.z || 0), offset + 8);
    buffer.writeFloatLE(scale, offset + 12);
    buffer.writeFloatLE(scale, offset + 16);
    buffer.writeFloatLE(scale, offset + 20);
    buffer.writeUInt8(Math.max(0, Math.min(255, Number(point.r || 0))), offset + 24);
    buffer.writeUInt8(Math.max(0, Math.min(255, Number(point.g || 0))), offset + 25);
    buffer.writeUInt8(Math.max(0, Math.min(255, Number(point.b || 0))), offset + 26);
    buffer.writeUInt8(Math.max(0, Math.min(255, Number(point.a ?? 255))), offset + 27);
    buffer.writeUInt8(255, offset + 28);
    buffer.writeUInt8(0, offset + 29);
    buffer.writeUInt8(0, offset + 30);
    buffer.writeUInt8(0, offset + 31);
  });
  return buffer;
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function clampByte(value) {
  return Math.max(0, Math.min(255, Math.round(value)));
}

function shToByte(value) {
  return clampByte((Number(value || 0) * 0.28209479177387814 + 0.5) * 255);
}

function plyTypeSize(type) {
  return {
    char: 1,
    uchar: 1,
    int8: 1,
    uint8: 1,
    short: 2,
    ushort: 2,
    int16: 2,
    uint16: 2,
    int: 4,
    uint: 4,
    int32: 4,
    uint32: 4,
    float: 4,
    float32: 4,
    double: 8,
    float64: 8
  }[type];
}

function readPlyValue(dataView, offset, type) {
  switch (type) {
    case "char":
    case "int8":
      return dataView.getInt8(offset);
    case "uchar":
    case "uint8":
      return dataView.getUint8(offset);
    case "short":
    case "int16":
      return dataView.getInt16(offset, true);
    case "ushort":
    case "uint16":
      return dataView.getUint16(offset, true);
    case "int":
    case "int32":
      return dataView.getInt32(offset, true);
    case "uint":
    case "uint32":
      return dataView.getUint32(offset, true);
    case "double":
    case "float64":
      return dataView.getFloat64(offset, true);
    case "float":
    case "float32":
    default:
      return dataView.getFloat32(offset, true);
  }
}

function writeJsonFile(filePath, data) {
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2), "utf8");
}

function jobProgressPath(jobPath) {
  return path.join(jobPath, "progress.json");
}

function writeJobProgress(jobPath, patch) {
  const progressPath = jobProgressPath(jobPath);
  let previous = {};
  if (fs.existsSync(progressPath)) {
    try {
      previous = JSON.parse(fs.readFileSync(progressPath, "utf8"));
    } catch {
      previous = {};
    }
  }
  writeJsonFile(progressPath, {
    ...previous,
    ...patch,
    updatedAt: new Date().toISOString()
  });
}

function readJobProgress(jobPath) {
  const progressPath = jobProgressPath(jobPath);
  if (!fs.existsSync(progressPath)) return null;
  try {
    return JSON.parse(fs.readFileSync(progressPath, "utf8"));
  } catch {
    return null;
  }
}

function maskAlphaFromBorder({ data, width, height, channels }) {
  const borderSamples = [];
  const border = Math.max(8, Math.round(Math.min(width, height) * 0.035));
  for (let y = 0; y < height; y += 4) {
    for (let x = 0; x < width; x += 4) {
      if (x > border && x < width - border && y > border && y < height - border) continue;
      const offset = (y * width + x) * channels;
      const alpha = channels === 4 ? data[offset + 3] : 255;
      if (alpha < 200) continue;
      borderSamples.push([data[offset], data[offset + 1], data[offset + 2]]);
    }
  }
  if (borderSamples.length < 64) return null;
  const borderColor = borderSamples.reduce((sum, color) => [sum[0] + color[0], sum[1] + color[1], sum[2] + color[2]], [0, 0, 0]).map((value) => value / borderSamples.length);
  const mask = Buffer.alloc(width * height);
  const distances = [];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const offset = (y * width + x) * channels;
      const alpha = channels === 4 ? data[offset + 3] : 255;
      if (alpha < 20) {
        mask[y * width + x] = 0;
        continue;
      }
      const dr = data[offset] - borderColor[0];
      const dg = data[offset + 1] - borderColor[1];
      const db = data[offset + 2] - borderColor[2];
      const distance = Math.sqrt(dr * dr + dg * dg + db * db);
      distances.push(distance);
      mask[y * width + x] = distance;
    }
  }
  if (!distances.length) return null;
  distances.sort((a, b) => a - b);
  const median = distances[Math.floor(distances.length * 0.5)];
  const high = distances[Math.floor(distances.length * 0.82)];
  const threshold = clamp(Math.max(24, median + 12, high * 0.38), 24, 70);
  const soft = Buffer.alloc(width * height);
  for (let index = 0; index < mask.length; index++) {
    const value = mask[index];
    soft[index] = clamp(Math.round(((value - threshold) / 42) * 255), 0, 255);
  }
  return soft;
}

function morphMask(mask, width, height, mode, radius = 1) {
  const output = Buffer.alloc(mask.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let value = mode === "erode" ? 255 : 0;
      for (let dy = -radius; dy <= radius; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= height) continue;
        for (let dx = -radius; dx <= radius; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= width) continue;
          const sample = mask[yy * width + xx];
          value = mode === "erode" ? Math.min(value, sample) : Math.max(value, sample);
        }
      }
      output[y * width + x] = value;
    }
  }
  return output;
}

function blurMask(mask, width, height, radius = 2) {
  const output = Buffer.alloc(mask.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      let count = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= height) continue;
        for (let dx = -radius; dx <= radius; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= width) continue;
          sum += mask[yy * width + xx];
          count += 1;
        }
      }
      output[y * width + x] = Math.round(sum / Math.max(1, count));
    }
  }
  return output;
}

function keepPrimaryForeground(mask, width, height) {
  return keepForegroundComponents(mask, width, height, { threshold: 72, minArea: 96, keepMultiple: false });
}

function keepForegroundComponents(mask, width, height, options = {}) {
  const threshold = Number(options.threshold ?? 72);
  const minArea = Number(options.minArea ?? 96);
  const keepMultiple = Boolean(options.keepMultiple);
  const visited = new Uint8Array(mask.length);
  const binary = new Uint8Array(mask.length);
  for (let index = 0; index < mask.length; index++) {
    binary[index] = mask[index] > threshold ? 1 : 0;
  }

  const components = [];
  const queue = [];
  for (let start = 0; start < binary.length; start++) {
    if (!binary[start] || visited[start]) continue;
    let head = 0;
    let area = 0;
    let sumX = 0;
    let sumY = 0;
    let touchesBorder = false;
    const pixels = [];
    visited[start] = 1;
    queue.length = 0;
    queue.push(start);
    while (head < queue.length) {
      const index = queue[head++];
      const x = index % width;
      const y = Math.floor(index / width);
      pixels.push(index);
      area += 1;
      sumX += x;
      sumY += y;
      touchesBorder ||= x < 4 || y < 4 || x >= width - 4 || y >= height - 4;
      const neighbors = [index - 1, index + 1, index - width, index + width];
      for (const next of neighbors) {
        if (next < 0 || next >= binary.length || visited[next] || !binary[next]) continue;
        if ((index % width === 0 && next === index - 1) || (index % width === width - 1 && next === index + 1)) continue;
        visited[next] = 1;
        queue.push(next);
      }
    }
    if (area < minArea) continue;
    const cx = sumX / area / width - 0.5;
    const cy = sumY / area / height - 0.52;
    const centrality = 1 - Math.min(1, Math.hypot(cx * 1.35, cy));
    const score = area * (touchesBorder ? 0.2 : 1) * (0.72 + centrality * 0.28);
    components.push({ area, score, touchesBorder, centrality, pixels });
  }

  if (!components.length) return mask;
  components.sort((a, b) => b.score - a.score);
  const best = components[0];
  const selected = keepMultiple
    ? components.filter((component) => {
      if (component === best) return true;
      if (component.touchesBorder && component.area < best.area * 0.85) return false;
      return component.area >= Math.max(minArea, best.area * 0.018) && component.centrality > 0.08;
    })
    : [best];
  const output = Buffer.alloc(mask.length);
  for (const component of selected) {
    for (const index of component.pixels) {
      output[index] = Math.max(output[index], mask[index]);
    }
  }
  return output;
}

function refineAlphaMask(alpha, width, height, maskRefine = "clean") {
  if (!alpha) return null;
  const mode = process.env.GS3D_MASK_MODE || "refined";
  if (mode === "heuristic") return alpha;
  const keepProps = normalizeMaskRefine(maskRefine) === "keepProps";
  const foreground = keepProps
    ? keepForegroundComponents(alpha, width, height, { threshold: 36, minArea: 48, keepMultiple: true })
    : keepPrimaryForeground(alpha, width, height);
  const closed = morphMask(morphMask(foreground, width, height, "dilate", keepProps ? 3 : 2), width, height, "erode", keepProps ? 2 : 2);
  const opened = keepProps ? closed : morphMask(morphMask(closed, width, height, "erode", 1), width, height, "dilate", 1);
  const feathered = blurMask(opened, width, height, keepProps ? 1 : 2);
  const output = Buffer.alloc(alpha.length);
  for (let index = 0; index < alpha.length; index++) {
    output[index] = Math.max(Math.min(alpha[index], 255), Math.min(feathered[index] + (keepProps ? 8 : 16), 255));
  }
  return output;
}

function runProcess(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: options.cwd || root,
      shell: false,
      windowsHide: true,
      env: process.env
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve({ stdout, stderr });
        return;
      }
      reject(new Error(`${command} exited with code ${code}.\n${stderr || stdout}`.trim()));
    });
  });
}

async function rembgAlphaMask(sourcePath, maskPath, width, height, maskMode = "person", maskRefine = "clean") {
  if (process.env.GS3D_SEGMENTER === "heuristic") return null;
  const pythonPath = process.env.GS3D_REMBG_PYTHON || path.join(root, ".venv-rembg", "Scripts", "python.exe");
  const scriptPath = path.join(root, "scripts", "backends", "remove-bg.py");
  if (!fs.existsSync(pythonPath) || !fs.existsSync(scriptPath)) return null;
  const normalizedMaskMode = normalizeMaskMode(maskMode);
  const normalizedMaskRefine = normalizeMaskRefine(maskRefine);
  const models = [rembgModelForMaskMode(normalizedMaskMode)];
  if (normalizedMaskMode === "objectProps" && normalizedMaskRefine === "keepProps") {
    const personModel = rembgModelForMaskMode("person");
    if (!models.includes(personModel)) models.push(personModel);
  }
  const masks = [];
  for (let index = 0; index < models.length; index++) {
    const model = models[index];
    const outputMaskPath = index === 0 ? maskPath : maskPath.replace(/\.png$/i, `.${index}.png`);
    const args = [scriptPath, sourcePath, outputMaskPath, "--model", model];
    if (normalizedMaskRefine === "keepProps") args.push("--no-post-process");
    await runProcess(pythonPath, args);
    const { data } = await sharp(outputMaskPath)
      .resize(width, height, { fit: "fill" })
      .greyscale()
      .raw()
      .toBuffer({ resolveWithObject: true });
    masks.push(Buffer.from(data));
  }
  const merged = Buffer.alloc(width * height);
  for (const mask of masks) {
    for (let index = 0; index < merged.length; index++) {
      merged[index] = Math.max(merged[index], mask[index]);
    }
  }
  return refineAlphaMask(merged, width, height, normalizedMaskRefine);
}

function normalizeMaskMode(mode) {
  return mode === "objectProps" || mode === "object-props" ? "objectProps" : "person";
}

function normalizeMaskRefine(mode) {
  return mode === "keepProps" || mode === "keep-props" || mode === "keepPropsStronger" ? "keepProps" : "clean";
}

function normalizeSceneMode(mode) {
  return mode === "full" || mode === "fullImage" || mode === "scene" ? "full" : "subject";
}

function rembgModelForMaskMode(mode) {
  const normalized = normalizeMaskMode(mode);
  if (normalized === "objectProps") {
    return process.env.GS3D_REMBG_OBJECT_MODEL || "isnet-general-use";
  }
  return process.env.GS3D_REMBG_PERSON_MODEL || process.env.GS3D_REMBG_MODEL || "u2net_human_seg";
}

function maskBounds(alpha, width, height, threshold = 16) {
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (alpha[y * width + x] <= threshold) continue;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }
  }
  if (maxX < minX || maxY < minY) return null;
  const subjectWidth = maxX - minX + 1;
  const subjectHeight = maxY - minY + 1;
  if (subjectWidth < 8 || subjectHeight < 8) return null;
  const padding = Math.round(Math.max(subjectWidth, subjectHeight) * Number(process.env.GS3D_MASK_CROP_PADDING || 0.08));
  const left = clamp(minX - padding, 0, width - 1);
  const top = clamp(minY - padding, 0, height - 1);
  const right = clamp(maxX + padding, left, width - 1);
  const bottom = clamp(maxY + padding, top, height - 1);
  return {
    left,
    top,
    width: right - left + 1,
    height: bottom - top + 1
  };
}

function safeExtractArea(crop, width, height) {
  if (!crop) return null;
  const left = Math.max(0, Math.min(width - 1, Math.floor(Number(crop.left) || 0)));
  const top = Math.max(0, Math.min(height - 1, Math.floor(Number(crop.top) || 0)));
  const right = Math.max(left, Math.min(width - 1, Math.floor(left + Number(crop.width || 0) - 1)));
  const bottom = Math.max(top, Math.min(height - 1, Math.floor(top + Number(crop.height || 0) - 1)));
  const area = {
    left,
    top,
    width: right - left + 1,
    height: bottom - top + 1
  };
  if (area.width < 8 || area.height < 8) return null;
  if (area.left + area.width > width || area.top + area.height > height) return null;
  return area;
}

async function preprocessPortraitView({ sourcePath, imagePath, maskPath, background = "black", sceneMode = "subject", maskMode = "person", maskRefine = "clean", manualKeepMaskPath = null, preprocessSize }) {
  const size = gs3dPreprocessSize(preprocessSize);
  const normalizedSceneMode = normalizeSceneMode(sceneMode);
  const prepared = await sharp(sourcePath)
    .rotate()
    .resize(size, size, { fit: "contain", background: normalizedSceneMode === "full" ? { r: 0, g: 0, b: 0, alpha: 1 } : { r: 0, g: 0, b: 0, alpha: 0 } })
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const { data, info } = prepared;
  const channels = info.channels;
  let alpha = null;
  if (normalizedSceneMode === "full") {
    alpha = Buffer.alloc(info.width * info.height, 255);
  } else if (process.env.GS3D_APPLY_INPUT_MASK !== "0") {
    const segmentSourcePath = maskPath.replace(/\.png$/i, ".segment-source.png");
    await sharp(data, { raw: { width: info.width, height: info.height, channels } })
      .png()
      .toFile(segmentSourcePath);
    try {
      alpha = await rembgAlphaMask(segmentSourcePath, maskPath, info.width, info.height, maskMode, maskRefine);
    } catch (error) {
      console.warn(`rembg segmentation failed, falling back to border mask: ${error.message}`);
      alpha = null;
    }
    if (!alpha) {
      alpha = maskAlphaFromBorder({ data, width: info.width, height: info.height, channels });
      alpha = refineAlphaMask(alpha, info.width, info.height, maskRefine);
    }
  }
  if (manualKeepMaskPath && fs.existsSync(manualKeepMaskPath)) {
    const manual = await sharp(manualKeepMaskPath)
      .resize(info.width, info.height, { fit: "contain", background: { r: 0, g: 0, b: 0, alpha: 0 } })
      .ensureAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });
    const manualChannels = manual.info.channels;
    if (!alpha) alpha = Buffer.alloc(info.width * info.height);
    const manualBinary = Buffer.alloc(info.width * info.height);
    for (let pixel = 0; pixel < info.width * info.height; pixel++) {
      const manualAlpha = manual.data[pixel * manualChannels + 3];
      manualBinary[pixel] = manualAlpha > 8 ? 255 : 0;
    }
    const expandedManual = morphMask(manualBinary, info.width, info.height, "dilate", 2);
    for (let pixel = 0; pixel < info.width * info.height; pixel++) {
      alpha[pixel] = Math.max(alpha[pixel], expandedManual[pixel]);
    }
  }

  const keyBackground = normalizedSceneMode === "subject" && background === "key";
  const keyColor = { r: 255, g: 0, b: 255 };
  const output = Buffer.alloc(info.width * info.height * 4);
  const mask = Buffer.alloc(info.width * info.height);
  const keyAlphaThreshold = Number(process.env.GS3D_KEY_ALPHA_THRESHOLD || 128);
  for (let pixel = 0; pixel < info.width * info.height; pixel++) {
    const inputOffset = pixel * channels;
    const outputOffset = pixel * 4;
    const inputAlpha = channels === 4 ? data[inputOffset + 3] : 255;
    const maskAlpha = alpha ? Math.min(inputAlpha, alpha[pixel]) : inputAlpha;
    const keep = keyBackground ? (maskAlpha >= keyAlphaThreshold ? 1 : 0) : maskAlpha / 255;
    if (keyBackground) {
      output[outputOffset] = Math.round(data[inputOffset] * keep + keyColor.r * (1 - keep));
      output[outputOffset + 1] = Math.round(data[inputOffset + 1] * keep + keyColor.g * (1 - keep));
      output[outputOffset + 2] = Math.round(data[inputOffset + 2] * keep + keyColor.b * (1 - keep));
      output[outputOffset + 3] = 255;
    } else if (background === "white") {
      output[outputOffset] = Math.round(data[inputOffset] * keep + 255 * (1 - keep));
      output[outputOffset + 1] = Math.round(data[inputOffset + 1] * keep + 255 * (1 - keep));
      output[outputOffset + 2] = Math.round(data[inputOffset + 2] * keep + 255 * (1 - keep));
      output[outputOffset + 3] = 255;
    } else {
      output[outputOffset] = Math.round(data[inputOffset] * keep);
      output[outputOffset + 1] = Math.round(data[inputOffset + 1] * keep);
      output[outputOffset + 2] = Math.round(data[inputOffset + 2] * keep);
      output[outputOffset + 3] = maskAlpha;
    }
    mask[pixel] = keyBackground ? (keep ? 255 : 0) : maskAlpha;
  }

  const crop = safeExtractArea(
    normalizedSceneMode === "subject" && alpha && process.env.GS3D_CROP_TO_MASK !== "0" ? maskBounds(alpha, info.width, info.height) : null,
    info.width,
    info.height
  );
  let imagePipeline = sharp(output, { raw: { width: info.width, height: info.height, channels: 4 } });
  let maskPipeline = sharp(mask, { raw: { width: info.width, height: info.height, channels: 1 } });
  if (crop) {
    imagePipeline = imagePipeline
      .extract(crop)
      .resize(size, size, {
        fit: "contain",
        background: keyBackground ? { ...keyColor, alpha: 1 } : { r: 0, g: 0, b: 0, alpha: 0 }
      });
    maskPipeline = maskPipeline
      .extract(crop)
      .resize(size, size, { fit: "contain", background: { r: 0, g: 0, b: 0, alpha: 1 } });
  }
  await imagePipeline.png().toFile(imagePath);
  await maskPipeline.png().toFile(maskPath);
}

async function synthesizeIntermediateView({ fromPath, toPath, imagePath, maskPath, blend = 0.5, preprocessSize }) {
  const size = gs3dPreprocessSize(preprocessSize);
  const [from, to] = await Promise.all([
    sharp(fromPath).resize(size, size, { fit: "contain", background: { r: 0, g: 0, b: 0, alpha: 0 } }).ensureAlpha().raw().toBuffer({ resolveWithObject: true }),
    sharp(toPath).resize(size, size, { fit: "contain", background: { r: 0, g: 0, b: 0, alpha: 0 } }).ensureAlpha().raw().toBuffer({ resolveWithObject: true })
  ]);
  const width = from.info.width;
  const height = from.info.height;
  const output = Buffer.alloc(width * height * 4);
  const mask = Buffer.alloc(width * height);
  const fromWeight = 1 - blend;
  const toWeight = blend;
  for (let pixel = 0; pixel < width * height; pixel++) {
    const offset = pixel * 4;
    const fromAlpha = from.data[offset + 3] / 255;
    const toAlpha = to.data[offset + 3] / 255;
    const alpha = Math.max(fromAlpha, toAlpha);
    const weightedAlpha = Math.max(0.001, fromAlpha * fromWeight + toAlpha * toWeight);
    output[offset] = Math.round((from.data[offset] * fromAlpha * fromWeight + to.data[offset] * toAlpha * toWeight) / weightedAlpha);
    output[offset + 1] = Math.round((from.data[offset + 1] * fromAlpha * fromWeight + to.data[offset + 1] * toAlpha * toWeight) / weightedAlpha);
    output[offset + 2] = Math.round((from.data[offset + 2] * fromAlpha * fromWeight + to.data[offset + 2] * toAlpha * toWeight) / weightedAlpha);
    output[offset + 3] = Math.round(alpha * 255);
    mask[pixel] = output[offset + 3];
  }
  const refined = refineAlphaMask(mask, width, height) || mask;
  for (let pixel = 0; pixel < width * height; pixel++) {
    const keep = refined[pixel] / 255;
    const offset = pixel * 4;
    output[offset] = Math.round(output[offset] * keep);
    output[offset + 1] = Math.round(output[offset + 1] * keep);
    output[offset + 2] = Math.round(output[offset + 2] * keep);
    output[offset + 3] = refined[pixel];
    mask[pixel] = refined[pixel];
  }
  await sharp(output, { raw: { width, height, channels: 4 } }).png().toFile(imagePath);
  await sharp(mask, { raw: { width, height, channels: 1 } }).png().toFile(maskPath);
}

function pointFromPlyRow(row, options = {}) {
  const scaleValues = [row.scale, row.scale_0, row.scale_1, row.scale_2].filter((value) => Number.isFinite(value));
  const expScaleValues = [row.scale_0, row.scale_1, row.scale_2].filter((value) => Number.isFinite(value)).map((value) => Math.exp(value));
  const scaleSource = expScaleValues.length ? expScaleValues : scaleValues;
  const scale = scaleSource.length ? clamp(scaleSource.reduce((sum, value) => sum + value, 0) / scaleSource.length, 0.003, 0.08) : 0.018;
  const hasRgb = Number.isFinite(row.red) && Number.isFinite(row.green) && Number.isFinite(row.blue);
  const hasSh = Number.isFinite(row.f_dc_0) && Number.isFinite(row.f_dc_1) && Number.isFinite(row.f_dc_2);
  const sourceX = Number(row.x || 0);
  const sourceY = Number(row.y || 0);
  const sourceZ = Number(row.z || 0);
  const zUp = options.coordinateSystem === "z_up";
  const flipY = !zUp && process.env.GS3D_FLIP_EXTERNAL_Y !== "0";
  return {
    x: Number(sourceX.toFixed(4)),
    y: Number((zUp ? sourceZ : sourceY * (flipY ? -1 : 1)).toFixed(4)),
    z: Number((zUp ? sourceY : sourceZ).toFixed(4)),
    r: hasRgb ? clampByte(row.red) : hasSh ? shToByte(row.f_dc_0) : 220,
    g: hasRgb ? clampByte(row.green) : hasSh ? shToByte(row.f_dc_1) : 220,
    b: hasRgb ? clampByte(row.blue) : hasSh ? shToByte(row.f_dc_2) : 220,
    a: Number.isFinite(row.alpha) ? clampByte(row.alpha) : Number.isFinite(row.opacity) ? clampByte(sigmoid(row.opacity) * 255) : 245,
    scale: Number(scale.toFixed(4)),
    view: "external"
  };
}

function parsePlyBuffer(buffer, { maxPoints = Number(process.env.GS3D_PREVIEW_POINTS || 450000) } = {}) {
  const headerEnd = buffer.indexOf("end_header");
  if (headerEnd < 0) {
    throw new Error("PLY file is missing end_header.");
  }
  const dataStart = buffer.indexOf("\n", headerEnd) + 1;
  const header = buffer.subarray(0, dataStart).toString("utf8");
  const lines = header.split(/\r?\n/);
  const format = lines.find((line) => line.startsWith("format "))?.split(/\s+/)[1];
  const vertexCount = Number(lines.find((line) => line.startsWith("element vertex "))?.split(/\s+/)[2] || 0);
  const coordinateSystem = lines.some((line) => line.trim() === "comment portrait_splat_forge_coords z_up") ? "z_up" : "external";
  if (!vertexCount) {
    throw new Error("PLY file does not declare vertices.");
  }

  const properties = [];
  let inVertex = false;
  for (const line of lines) {
    if (line.startsWith("element vertex ")) {
      inVertex = true;
      continue;
    }
    if (inVertex && line.startsWith("element ")) break;
    if (!inVertex || !line.startsWith("property ")) continue;
    const [, type, name] = line.trim().split(/\s+/);
    if (!plyTypeSize(type)) {
      throw new Error(`Unsupported PLY vertex property type: ${type}`);
    }
    properties.push({ type, name, size: plyTypeSize(type) });
  }

  const stride = Math.max(1, Math.ceil(vertexCount / maxPoints));
  const points = [];

  if (format === "ascii") {
    const body = buffer.subarray(dataStart).toString("utf8").trim().split(/\r?\n/);
    for (let vertexIndex = 0; vertexIndex < Math.min(vertexCount, body.length); vertexIndex += stride) {
      const values = body[vertexIndex].trim().split(/\s+/).map(Number);
      const row = Object.fromEntries(properties.map((property, index) => [property.name, values[index]]));
      points.push(pointFromPlyRow(row, { coordinateSystem }));
    }
  } else if (format === "binary_little_endian") {
    const rowSize = properties.reduce((sum, property) => sum + property.size, 0);
    const dataView = new DataView(buffer.buffer, buffer.byteOffset + dataStart, buffer.byteLength - dataStart);
    for (let vertexIndex = 0; vertexIndex < vertexCount; vertexIndex += stride) {
      let offset = vertexIndex * rowSize;
      const row = {};
      for (const property of properties) {
        row[property.name] = readPlyValue(dataView, offset, property.type);
        offset += property.size;
      }
      points.push(pointFromPlyRow(row, { coordinateSystem }));
    }
  } else {
    throw new Error(`Unsupported PLY format: ${format || "unknown"}`);
  }

  return {
    format: "portrait-splat-forge.external-ply-preview.v1",
    createdAt: new Date().toISOString(),
    pointCount: points.length,
    originalVertexCount: vertexCount,
    previewStride: stride,
    sourceViews: [],
    points
  };
}

function parseBinarySplatBuffer(buffer) {
  const rowSize = 32;
  if (buffer.byteLength % rowSize !== 0) {
    throw new Error("Invalid .splat file size.");
  }
  const dataView = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  const points = [];
  for (let offset = 0; offset < buffer.byteLength; offset += rowSize) {
    const sx = dataView.getFloat32(offset + 12, true);
    const sy = dataView.getFloat32(offset + 16, true);
    const sz = dataView.getFloat32(offset + 20, true);
    points.push({
      x: Number(dataView.getFloat32(offset, true).toFixed(4)),
      y: Number(dataView.getFloat32(offset + 4, true).toFixed(4)),
      z: Number(dataView.getFloat32(offset + 8, true).toFixed(4)),
      r: dataView.getUint8(offset + 24),
      g: dataView.getUint8(offset + 25),
      b: dataView.getUint8(offset + 26),
      a: dataView.getUint8(offset + 27),
      scale: Number(((sx + sy + sz) / 3).toFixed(4)),
      view: "external"
    });
  }
  return {
    format: "portrait-splat-forge.external-binary-splat-preview.v1",
    createdAt: new Date().toISOString(),
    pointCount: points.length,
    sourceViews: [],
    points
  };
}

function saveSplatArtifacts({ splat, ply }) {
  const id = crypto.randomUUID();
  const jsonPath = path.join(splatDir, `${id}.splat.json`);
  const plyPath = path.join(splatDir, `${id}.ply`);
  const binaryPath = path.join(splatDir, `${id}.splat`);
  const manifestPath = path.join(splatDir, `${id}.manifest.json`);
  const manifest = {
    id,
    createdAt: new Date().toISOString(),
    pointCount: splat.pointCount || splat.points.length,
    files: {
      json: publicPath(jsonPath),
      ply: publicPath(plyPath),
      splat: publicPath(binaryPath)
    }
  };

  fs.writeFileSync(jsonPath, JSON.stringify(splat, null, 2), "utf8");
  fs.writeFileSync(plyPath, ply, "utf8");
  fs.writeFileSync(binaryPath, splatToBinary(splat));
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2), "utf8");

  return {
    ...manifest,
    files: {
      ...manifest.files,
      manifest: publicPath(manifestPath)
    }
  };
}

function saveBuiltSplatArtifacts({ splat, ply, sourcePlyPath, sourceSplatPath }) {
  const id = crypto.randomUUID();
  const jsonPath = path.join(splatDir, `${id}.splat.json`);
  const plyPath = path.join(splatDir, `${id}.ply`);
  const binaryPath = path.join(splatDir, `${id}.splat`);
  const manifestPath = path.join(splatDir, `${id}.manifest.json`);
  const manifest = {
    id,
    createdAt: new Date().toISOString(),
    pointCount: splat.originalVertexCount || splat.pointCount || splat.points.length,
    previewPointCount: splat.points.length,
    files: {
      json: publicPath(jsonPath),
      ply: publicPath(plyPath),
      splat: publicPath(binaryPath)
    }
  };

  fs.writeFileSync(jsonPath, JSON.stringify(splat, null, 2), "utf8");
  if (sourcePlyPath) {
    fs.copyFileSync(sourcePlyPath, plyPath);
  } else {
    fs.writeFileSync(plyPath, ply || splatToPly(splat), "utf8");
  }
  if (sourceSplatPath) {
    fs.copyFileSync(sourceSplatPath, binaryPath);
  } else {
    fs.writeFileSync(binaryPath, splatToBinary(splat));
  }
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2), "utf8");

  return {
    ...manifest,
    files: {
      ...manifest.files,
      manifest: publicPath(manifestPath)
    }
  };
}

function listRecentSplatOutputs(limit = 12) {
  const splats = fs.existsSync(splatDir) ? fs.readdirSync(splatDir)
    .filter((name) => name.endsWith(".manifest.json"))
    .map((name) => {
      const manifestPath = path.join(splatDir, name);
      try {
        const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
        const stat = fs.statSync(manifestPath);
        return {
          ...manifest,
          createdAt: manifest.createdAt || stat.mtime.toISOString(),
          files: {
            ...(manifest.files || {}),
            manifest: publicPath(manifestPath)
          }
        };
      } catch {
        return null;
      }
    }) : [];

  const sequences = fs.existsSync(sharpSequenceDir) ? fs.readdirSync(sharpSequenceDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => {
      const manifestPath = path.join(sharpSequenceDir, entry.name, "manifest.json");
      try {
        if (!fs.existsSync(manifestPath)) return null;
        const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
        const stat = fs.statSync(manifestPath);
        return {
          ...manifest,
          type: "sharp-sequence",
          pointCount: manifest.frames?.[0]?.pointCount || 0,
          previewPointCount: manifest.frames?.[0]?.pointCount || 0,
          createdAt: manifest.createdAt || stat.mtime.toISOString(),
          files: {
            ...(manifest.files || {}),
            manifest: publicPath(manifestPath)
          }
        };
      } catch {
        return null;
      }
    }) : [];

  return [...splats, ...sequences]
    .filter(Boolean)
    .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
    .slice(0, limit);
}

function svgPlaceholder({ label, yaw, sourceUrl }) {
  const hue = yaw === 0 ? 180 : yaw < 0 ? 206 : yaw > 100 ? 31 : 146;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024">
  <defs>
    <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0" stop-color="hsl(${hue}, 42%, 18%)"/>
      <stop offset="1" stop-color="hsl(${(hue + 70) % 360}, 32%, 8%)"/>
    </linearGradient>
    <radialGradient id="skin" cx="50%" cy="30%" r="60%">
      <stop offset="0" stop-color="#e6c2a8"/>
      <stop offset="1" stop-color="#a36f58"/>
    </radialGradient>
  </defs>
  <rect width="1024" height="1024" fill="url(#bg)"/>
  <g transform="translate(512 534)">
    <ellipse cx="0" cy="220" rx="${yaw === 180 ? 210 : 238}" ry="254" fill="#172032"/>
    <path d="M-260 420 C-230 230 -150 118 0 118 C150 118 230 230 260 420 Z" fill="#24344f"/>
    <ellipse cx="0" cy="-96" rx="${yaw === 180 ? 126 : 142}" ry="174" fill="url(#skin)"/>
    <path d="M-125 -170 C-86 -258 82 -260 128 -168 C96 -226 -88 -226 -125 -170 Z" fill="#16120f"/>
    <path d="M-170 432 L-64 150 M170 432 L64 150" stroke="#d2a85f" stroke-width="9" opacity=".5"/>
    ${yaw !== 180 ? `<circle cx="-48" cy="-110" r="13" fill="#251813"/><circle cx="48" cy="-110" r="13" fill="#251813"/><path d="M-58 -28 C-20 4 25 4 62 -28" fill="none" stroke="#4d2f28" stroke-width="10" stroke-linecap="round"/>` : `<path d="M-112 -170 C-74 -235 75 -236 114 -172 C82 -200 -80 -201 -112 -170 Z" fill="#1c1715"/>`}
  </g>
  <text x="48" y="76" fill="white" font-family="Inter, Arial, sans-serif" font-size="34" font-weight="700">${label}</text>
  <text x="48" y="124" fill="rgba(255,255,255,.72)" font-family="Inter, Arial, sans-serif" font-size="22">mock view - set OPENAI_API_KEY for generated output</text>
  <text x="48" y="956" fill="rgba(255,255,255,.58)" font-family="Inter, Arial, sans-serif" font-size="18">${sourceUrl}</text>
</svg>`;
}

async function generateWithOpenAI({ sourcePath, spec }) {
  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const model = process.env.OPENAI_IMAGE_MODEL || "gpt-image-2";
  const prompt = [
    "Create a studio-quality image for a single-image Gaussian Splatting portrait reconstruction pipeline.",
    `View: ${spec.prompt}.`,
    "Keep the same person, facial identity, hair, age, body proportions, denim/clothing details, and bust crop.",
    "Use transparent or plain dark background, neutral lighting, no text, no extra body parts, no accessories unless present in the reference."
  ].join(" ");

  if (model.startsWith("gpt-image")) {
    const referencePng = await sharp(sourcePath)
      .resize(1024, 1024, { fit: "contain", background: { r: 0, g: 0, b: 0, alpha: 0 } })
      .png()
      .toBuffer();
    const response = await client.responses.create({
      model: process.env.OPENAI_RESPONSES_MODEL || "gpt-5.5",
      input: [
        {
          role: "user",
          content: [
            { type: "input_text", text: prompt },
            { type: "input_image", image_url: `data:image/png;base64,${referencePng.toString("base64")}` }
          ]
        }
      ],
      tools: [
        {
          type: "image_generation",
          model,
          size: "1024x1024",
          quality: "medium",
          action: "edit"
        }
      ],
      tool_choice: { type: "image_generation" }
    });

    const imageCall = response.output?.find((output) => output.type === "image_generation_call");
    if (!imageCall?.result) {
      throw new Error("Responses image_generation did not return an image.");
    }
    return Buffer.from(imageCall.result, "base64");
  }

  const result = await client.images.edit({
    model,
    image: fs.createReadStream(sourcePath),
    prompt,
    size: "1024x1024",
    n: 1
  });

  const image = result.data?.[0];
  if (image?.b64_json) {
    return Buffer.from(image.b64_json, "base64");
  }
  if (image?.url) {
    const response = await fetch(image.url);
    if (!response.ok) {
      throw new Error(`Could not download generated image: ${response.status}`);
    }
    return Buffer.from(await response.arrayBuffer());
  }
  throw new Error("OpenAI image response did not include b64_json or url.");
}

function runBackendCommand(command, replacements, cwd) {
  const rendered = Object.entries(replacements).reduce(
    (current, [key, value]) => current.replaceAll(`{${key}}`, String(value)),
    command
  );
  const timeoutMs = Number(process.env.GS3D_BACKEND_TIMEOUT_MS || 1000 * 60 * 60 * 3);

  return new Promise((resolve, reject) => {
    const child = spawn(rendered, {
      cwd,
      shell: true,
      windowsHide: true,
      env: {
        ...process.env,
        GS3D_INPUT_DIR: replacements.inputDir,
        GS3D_OUTPUT_DIR: replacements.outputDir,
        GS3D_MANIFEST: replacements.manifest,
        GS3D_QUALITY: replacements.quality,
        GS3D_JOB_ID: replacements.jobId
      }
    });
    let stdout = "";
    let stderr = "";
    const stdoutPath = replacements.jobPath ? path.join(replacements.jobPath, "backend.stdout.log") : null;
    const stderrPath = replacements.jobPath ? path.join(replacements.jobPath, "backend.stderr.log") : null;
    if (stdoutPath) fs.writeFileSync(stdoutPath, "", "utf8");
    if (stderrPath) fs.writeFileSync(stderrPath, "", "utf8");
    const timer = setTimeout(() => {
      child.kill();
      reject(new Error(`Backend timed out after ${Math.round(timeoutMs / 1000)}s.`));
    }, timeoutMs);

    child.stdout.on("data", (chunk) => {
      const text = chunk.toString();
      stdout += text;
      if (stdoutPath) fs.appendFileSync(stdoutPath, text, "utf8");
    });
    child.stderr.on("data", (chunk) => {
      const text = chunk.toString();
      stderr += text;
      if (stderrPath) fs.appendFileSync(stderrPath, text, "utf8");
    });
    child.on("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code === 0) {
        resolve({ command: rendered, stdout, stderr });
        return;
      }
      reject(new Error(`Backend exited with code ${code}.\n${stderr || stdout}`.trim()));
    });
  });
}

function runTool(command, args, cwd = root) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd,
      windowsHide: true,
      shell: false
    });
    let stdout = "";
    let stderr = "";
    child.stdout?.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr?.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", (error) => reject(error));
    child.on("close", (code) => {
      if (code === 0) {
        resolve({ stdout, stderr });
        return;
      }
      reject(new Error(stderr.trim() || `${command} exited with code ${code}`));
    });
  });
}

async function probeVideo(videoPath) {
  const ffprobe = process.env.FFPROBE_COMMAND || "ffprobe";
  try {
    const { stdout } = await runTool(ffprobe, [
      "-v",
      "error",
      "-print_format",
      "json",
      "-show_format",
      "-show_streams",
      videoPath
    ]);
    const payload = JSON.parse(stdout);
    const videoStream = payload.streams?.find((stream) => stream.codec_type === "video") || {};
    return {
      available: true,
      duration: Number(payload.format?.duration || videoStream.duration || 0),
      width: Number(videoStream.width || 0),
      height: Number(videoStream.height || 0),
      codec: videoStream.codec_name || "",
      frames: Number(videoStream.nb_frames || 0),
      frameRate: videoStream.avg_frame_rate || videoStream.r_frame_rate || ""
    };
  } catch (error) {
    return {
      available: false,
      warning: `ffprobe unavailable or failed: ${error.message}`
    };
  }
}

async function createVideoSummaryImage(videoPath) {
  const ffmpeg = process.env.FFMPEG_COMMAND || "ffmpeg";
  const id = crypto.randomUUID();
  const summaryPath = path.join(viewDir, `video-summary-${id}.jpg`);
  await runTool(ffmpeg, [
    "-y",
    "-ss",
    "0.1",
    "-i",
    videoPath,
    "-frames:v",
    "1",
    "-vf",
    "scale='min(640,iw)':-2",
    "-q:v",
    "3",
    summaryPath
  ]);
  return publicPath(summaryPath);
}

async function extractVideoFrames({ videoPath, frameCount, maxWidth, trimStart, trimEnd }) {
  const safeFrameCount = clamp(Number(frameCount || 60), 1, 300);
  const safeMaxWidth = maxWidth === "source" ? "source" : clamp(Number(maxWidth || 1920), 720, 3840);
  const ffmpeg = process.env.FFMPEG_COMMAND || "ffmpeg";
  const analysis = await probeVideo(videoPath);
  const sourceDuration = analysis.duration > 0 ? analysis.duration : safeFrameCount;
  const start = clamp(Number(trimStart || 0), 0, sourceDuration);
  const requestedEnd = trimEnd === "" || trimEnd === undefined || trimEnd === null ? sourceDuration : Number(trimEnd);
  const end = clamp(Number.isFinite(requestedEnd) ? requestedEnd : sourceDuration, start, sourceDuration);
  const clipDuration = Math.max(0, end - start);
  if (clipDuration <= 0.05) {
    throw new Error("Video trim range is too short. Set an end time greater than the start time.");
  }
  const id = crypto.randomUUID();
  const framesDir = path.join(viewDir, `video-${id}`);
  fs.mkdirSync(framesDir, { recursive: true });
  const outputPattern = path.join(framesDir, "frame-%04d.png");
  const fps = Math.max(0.1, safeFrameCount / clipDuration);

  const scaleFilter = safeMaxWidth === "source"
    ? "scale=iw:-2"
    : `scale='min(${safeMaxWidth},iw)':-2`;

  await runTool(ffmpeg, [
    "-y",
    "-ss",
    String(start),
    "-t",
    String(clipDuration),
    "-i",
    videoPath,
    "-vf",
    `fps=${fps.toFixed(4)},${scaleFilter}`,
    "-frames:v",
    String(safeFrameCount),
    outputPattern
  ]);

  const files = fs.readdirSync(framesDir)
    .filter((name) => name.toLowerCase().endsWith(".png"))
    .sort();
  if (!files.length) {
    throw new Error("FFmpeg did not write any frames.");
  }

  return {
    id,
    requestedFrameCount: safeFrameCount,
    maxWidth: safeMaxWidth,
    trim: {
      start,
      end,
      duration: clipDuration
    },
    frames: files.map((name, index) => ({
      index,
      name,
      timestamp: files.length > 1 ? Number((start + (clipDuration * index) / (files.length - 1)).toFixed(3)) : start,
      url: publicPath(path.join(framesDir, name))
    })),
    analysis
  };
}

async function stageExternalInputs({ jobPath, views, quality, backendDef, sceneMode = "subject", maskMode = "person", maskRefine = "clean", manualKeepMask = null, preprocessSize }) {
  const resolvedPreprocessSize = gs3dPreprocessSize(preprocessSize);
  const normalizedSceneMode = normalizeSceneMode(sceneMode);
  const normalizedMaskMode = normalizeMaskMode(maskMode);
  const normalizedMaskRefine = normalizeMaskRefine(maskRefine);
  const inputDir = path.join(jobPath, "input");
  const imageDir = path.join(inputDir, "images");
  const maskDir = path.join(inputDir, "masks");
  const augmentedImageDir = path.join(inputDir, "augmented", "images");
  const augmentedMaskDir = path.join(inputDir, "augmented", "masks");
  const manualMaskPath = normalizedSceneMode === "subject" && manualKeepMask ? writeDataUrlImage(manualKeepMask, path.join(inputDir, "manual-keep-front.png")) : null;
  fs.mkdirSync(imageDir, { recursive: true });
  fs.mkdirSync(maskDir, { recursive: true });
  fs.mkdirSync(augmentedImageDir, { recursive: true });
  fs.mkdirSync(augmentedMaskDir, { recursive: true });
  const requiredOrder = backendDef?.requiredViews || ["front", "left45", "right45", "back"];
  const stagedViews = [];
  const stagedById = new Map();

  for (const id of requiredOrder) {
    const view = views.find((item) => item.id === id && item.url);
    if (!view) {
      throw new Error(`Required views are missing: ${requiredOrder.join(", ")}.`);
    }
    const sourcePath = outputPathFromPublicUrl(view.url);
    const stagedPath = path.join(imageDir, `${id}.png`);
    const maskPath = path.join(maskDir, `${id}.png`);
    const sharpBackground = normalizedSceneMode === "full" ? "black" : backendDef?.env === "SHARP_COMMAND" ? (process.env.GS3D_SHARP_BACKGROUND || "key") : "black";
    writeJobProgress(jobPath, {
      phase: "segmenting",
      message: normalizedSceneMode === "full"
        ? `Preparing ${view.label || id} full image scene`
        : `Segmenting ${view.label || id} foreground (${normalizedMaskMode === "objectProps" ? "Object+Props" : "Person"}, ${normalizedMaskRefine === "keepProps" ? "Keep props stronger" : "Clean"})`,
      percent: 8
    });
    await preprocessPortraitView({
      sourcePath,
      imagePath: stagedPath,
      maskPath,
      background: sharpBackground,
      sceneMode: normalizedSceneMode,
      maskMode: normalizedMaskMode,
      maskRefine: normalizedMaskRefine,
      manualKeepMaskPath: id === "front" ? manualMaskPath : null,
      preprocessSize: resolvedPreprocessSize
    });
    writeJobProgress(jobPath, {
      phase: "staging",
      message: `Staged ${view.label || id} masked input`,
      percent: 12
    });
    stagedViews.push({
      id,
      label: view.label,
      yaw: view.yaw,
      status: view.status,
      originalUrl: view.url,
      path: stagedPath,
      file: path.relative(inputDir, stagedPath).replaceAll(path.sep, "/"),
      mask: path.relative(inputDir, maskPath).replaceAll(path.sep, "/")
    });
    stagedById.set(id, { imagePath: stagedPath, maskPath });
  }

  const augmentedViews = [];
  if (requiredOrder.length >= 4 && process.env.GS3D_AUGMENT_VIEWS !== "0") {
    const limit = clamp(Number(process.env.GS3D_AUGMENT_VIEW_COUNT || augmentedViewSpecs.length), 0, augmentedViewSpecs.length);
    for (const spec of augmentedViewSpecs.slice(0, limit)) {
      const from = stagedById.get(spec.from);
      const to = stagedById.get(spec.to);
      if (!from || !to) continue;
      const stagedPath = path.join(augmentedImageDir, `${spec.id}.png`);
      const maskPath = path.join(augmentedMaskDir, `${spec.id}.png`);
      await synthesizeIntermediateView({
        fromPath: from.imagePath,
        toPath: to.imagePath,
        imagePath: stagedPath,
        maskPath,
        blend: spec.blend,
        preprocessSize: resolvedPreprocessSize
      });
      augmentedViews.push({
        id: spec.id,
        label: spec.label,
        yaw: spec.yaw,
        status: "augmented",
        from: spec.from,
        to: spec.to,
        file: path.relative(inputDir, stagedPath).replaceAll(path.sep, "/"),
        mask: path.relative(inputDir, maskPath).replaceAll(path.sep, "/")
      });
    }
  }

  const manifest = {
    format: "portrait-splat-forge.gs3d-job.v1",
    createdAt: new Date().toISOString(),
    quality,
    backendRequiredViews: requiredOrder,
    preprocessing: {
      mask: process.env.GS3D_APPLY_INPUT_MASK !== "0",
      sceneMode: normalizedSceneMode,
      maskMode: process.env.GS3D_MASK_MODE || "refined",
      subjectMaskMode: normalizedMaskMode,
      maskRefine: normalizedMaskRefine,
      manualKeepMask: Boolean(manualMaskPath),
      rembgModel: rembgModelForMaskMode(normalizedMaskMode),
      augmentedViews: process.env.GS3D_AUGMENT_VIEWS !== "0",
      size: resolvedPreprocessSize
    },
    views: stagedViews,
    augmentedViews,
    expectedOutput: {
      preferred: "output.ply",
      accepted: ["output.splat.json", "output.ply", "output.splat", "last.ply", "point_cloud.ply"]
    }
  };
  const manifestPath = path.join(inputDir, "manifest.json");
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2), "utf8");
  return { inputDir, manifestPath, stagedViews, augmentedViews };
}

function listFilesRecursive(dir) {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const resolved = path.join(dir, entry.name);
    return entry.isDirectory() ? listFilesRecursive(resolved) : [resolved];
  });
}

function findExternalOutput(outputDirForJob) {
  const files = listFilesRecursive(outputDirForJob);
  const byName = (name) => files.find((file) => path.basename(file).toLowerCase() === name);
  return (
    byName("output.splat.json") ||
    byName("output.ply") ||
    byName("last.ply") ||
    byName("point_cloud.ply") ||
    byName("output.splat") ||
    files.find((file) => file.toLowerCase().endsWith(".splat.json")) ||
    files.find((file) => file.toLowerCase().endsWith(".ply")) ||
    files.find((file) => file.toLowerCase().endsWith(".splat"))
  );
}

function loadExternalSplat(outputPath, stagedViews, options = {}) {
  if (!outputPath || !fs.existsSync(outputPath)) {
    throw new Error("Backend completed but did not write a supported artifact.");
  }
  const lower = outputPath.toLowerCase();
  let splat;
  if (lower.endsWith(".splat.json") || lower.endsWith(".json")) {
    splat = JSON.parse(fs.readFileSync(outputPath, "utf8"));
    if (!splat?.points?.length) {
      throw new Error("External JSON does not contain preview points.");
    }
    splat = { ...splat, pointCount: splat.pointCount || splat.points.length };
  } else if (lower.endsWith(".ply")) {
    splat = parsePlyBuffer(fs.readFileSync(outputPath), { maxPoints: options.previewMaxPoints });
  } else if (lower.endsWith(".splat")) {
    splat = parseBinarySplatBuffer(fs.readFileSync(outputPath));
  } else {
    throw new Error(`Unsupported backend output: ${outputPath}`);
  }

  return {
    ...splat,
    backendArtifact: path.basename(outputPath),
    sourceViews: stagedViews.map(({ id, label, yaw, status, originalUrl }) => ({ id, label, yaw, status, url: originalUrl }))
  };
}

function previewMaxPointsForMode(mode) {
  if (mode === "full") return Number(process.env.GS3D_FULL_PREVIEW_POINTS || 1300000);
  if (mode === "dense") return Number(process.env.GS3D_DENSE_PREVIEW_POINTS || 900000);
  return Number(process.env.GS3D_PREVIEW_POINTS || 450000);
}

async function buildExternalSplat({ backend, views, quality, jobId, previewMode = "auto", sceneMode = "subject", maskMode = "person", maskRefine = "clean", manualKeepMask = null, preprocessSize }) {
  const def = externalBackendDefs[backend];
  if (!def) {
    throw new Error(`Unknown splat backend: ${backend}`);
  }
  const command = process.env[def.env];
  if (!command) {
    throw new Error(`${def.label} is not configured. Set ${def.env} before starting the server.`);
  }

  const id = jobId || crypto.randomUUID();
  const jobPath = path.join(jobDir, id);
  const outputDirForJob = path.join(jobPath, "output");
  fs.mkdirSync(outputDirForJob, { recursive: true });
  writeJobProgress(jobPath, {
    status: "running",
    backend,
    backendLabel: def.label,
    quality,
    phase: "staging",
    message: "Preparing masked inputs and augmented views",
    percent: 4
  });
  const { inputDir, manifestPath, stagedViews } = await stageExternalInputs({ jobPath, views, quality, backendDef: def, sceneMode, maskMode, maskRefine, manualKeepMask, preprocessSize });
  writeJobProgress(jobPath, {
    phase: "backend",
    message: `Running ${def.label}`,
    percent: 8
  });
  const runLog = await runBackendCommand(
    command,
    {
      inputDir,
      outputDir: outputDirForJob,
      manifest: manifestPath,
      quality,
      jobId: id,
      jobPath
    },
    root
  );

  fs.writeFileSync(path.join(jobPath, "backend.command.txt"), runLog.command, "utf8");

  writeJobProgress(jobPath, {
    phase: "importing",
    message: "Importing backend PLY into browser preview",
    percent: 92
  });
  const artifactPath = findExternalOutput(outputDirForJob);
  const splat = loadExternalSplat(artifactPath, stagedViews, { previewMaxPoints: previewMaxPointsForMode(previewMode) });
  const lower = artifactPath.toLowerCase();
  const saved = saveBuiltSplatArtifacts({
    splat,
    sourcePlyPath: lower.endsWith(".ply") ? artifactPath : null,
    sourceSplatPath: lower.endsWith(".splat") ? artifactPath : null
  });

  const result = {
    splat: {
      ...splat,
      backend,
      backendLabel: def.label,
      jobId: id,
      previewMode
    },
    saved
  };
  writeJobProgress(jobPath, {
    status: "completed",
    phase: "completed",
    message: "GS3D build completed",
    percent: 100
  });
  return result;
}

app.post("/api/upload", upload.single("image"), (req, res) => {
  if (!req.file) {
    res.status(400).json({ error: "Missing image file." });
    return;
  }
  const ext = path.extname(req.file.originalname || "") || ".png";
  const finalPath = path.join(uploadDir, `${req.file.filename}${ext}`);
  fs.renameSync(req.file.path, finalPath);
  res.json({
    id: path.basename(finalPath),
    name: req.file.originalname,
    url: publicPath(finalPath)
  });
});

app.post("/api/upload-video", videoUpload.single("video"), async (req, res) => {
  if (!req.file) {
    res.status(400).json({ error: "Missing video file." });
    return;
  }
  const ext = path.extname(req.file.originalname || "") || ".mp4";
  const finalPath = path.join(videoDir, `${req.file.filename}${ext}`);
  fs.renameSync(req.file.path, finalPath);
  let summaryUrl = null;
  let summaryWarning = null;
  try {
    summaryUrl = await createVideoSummaryImage(finalPath);
  } catch (error) {
    summaryWarning = error.message;
  }
  res.json({
    id: path.basename(finalPath),
    name: req.file.originalname,
    size: req.file.size,
    url: publicPath(finalPath),
    summaryUrl,
    summaryWarning
  });
});

app.post("/api/analyze-video", async (req, res) => {
  try {
    const videoPath = videoPathFromId(req.body?.videoId);
    if (!fs.existsSync(videoPath)) {
      res.status(404).json({ error: "Video upload not found." });
      return;
    }
    res.json({ analysis: await probeVideo(videoPath) });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/extract-video-frames", async (req, res) => {
  try {
    const videoPath = videoPathFromId(req.body?.videoId);
    if (!fs.existsSync(videoPath)) {
      res.status(404).json({ error: "Video upload not found." });
      return;
    }
    res.json(await extractVideoFrames({
      videoPath,
      frameCount: req.body?.frameCount,
      maxWidth: req.body?.maxWidth,
      trimStart: req.body?.trimStart,
      trimEnd: req.body?.trimEnd
    }));
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/generate-views", async (req, res) => {
  const { imageId, mode = "mock" } = req.body || {};
  const sourcePath = imageId ? path.join(uploadDir, path.basename(imageId)) : null;

  if (!sourcePath || !fs.existsSync(sourcePath)) {
    res.status(400).json({ error: "Upload an image before generating views." });
    return;
  }

  const jobId = crypto.randomUUID();
  const generated = [];
  const usingOpenAI = mode === "openai" && Boolean(process.env.OPENAI_API_KEY);

  for (const spec of viewSpecs) {
    const sourceExt = extensionFromPath(sourcePath);
    const outExt = spec.id === "front" ? sourceExt : usingOpenAI ? ".png" : ".svg";
    const outPath = path.join(viewDir, `${jobId}-${spec.id}${outExt}`);
    try {
      if (usingOpenAI && spec.id !== "front") {
        const buffer = await generateWithOpenAI({ sourcePath, spec });
        fs.writeFileSync(outPath, buffer);
      } else if (spec.id === "front") {
        fs.copyFileSync(sourcePath, outPath);
      } else {
        fs.writeFileSync(outPath, svgPlaceholder({ ...spec, sourceUrl: imageId }), "utf8");
      }
      generated.push({ ...spec, url: publicPath(outPath), status: spec.id === "front" ? "source" : usingOpenAI ? "generated" : "mock" });
    } catch (error) {
      console.error(`[generate-views] ${spec.id} failed:`, error);
      if (spec.id === "front") {
        generated.push({ ...spec, url: null, status: "failed", error: error.message });
      } else {
        const fallbackPath = path.join(viewDir, `${jobId}-${spec.id}-fallback.svg`);
        fs.writeFileSync(fallbackPath, svgPlaceholder({ ...spec, sourceUrl: imageId }), "utf8");
        generated.push({ ...spec, url: publicPath(fallbackPath), status: "fallback", error: error.message });
      }
    }
  }

  res.json({ jobId, usingOpenAI, views: generated });
});

app.post("/api/save-splat", (req, res) => {
  const { splat, ply } = req.body || {};
  if (!splat?.points?.length || !ply) {
    res.status(400).json({ error: "Missing splat data or PLY content." });
    return;
  }

  res.json(saveSplatArtifacts({ splat, ply }));
});

app.post("/api/build-splat", async (req, res) => {
  try {
    const { views, backend = "lightweight", quality = "balanced", previewMode = "auto", sceneMode = "subject", maskMode = "person", maskRefine = "clean", manualKeepMask = null, preprocessSize } = req.body || {};
    if (!Array.isArray(views)) {
      res.status(400).json({ error: "Missing generated views." });
      return;
    }
    if (backend !== "lightweight") {
      res.json(await buildExternalSplat({ backend, views, quality, previewMode, sceneMode, maskMode, maskRefine, manualKeepMask, preprocessSize }));
      return;
    }

    const splat = await buildLightweightSplat(views, quality);
    const ply = splatToPly(splat);
    const saved = saveBuiltSplatArtifacts({ splat, ply });
    res.json({ splat, saved });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

function combinedJobProgress(jobId) {
  const jobPath = path.join(jobDir, path.basename(jobId));
  const serverProgress = readJobProgress(jobPath) || {};
  const backendProgress = readJobProgress(path.join(jobPath, "output")) || {};
  return {
    ...serverProgress,
    ...backendProgress,
    jobId
  };
}

function saveSharpSequenceManifest(sequence) {
  const sequencePath = path.join(sharpSequenceDir, sequence.id);
  fs.mkdirSync(sequencePath, { recursive: true });
  const manifestPath = path.join(sequencePath, "manifest.json");
  fs.writeFileSync(manifestPath, JSON.stringify(sequence, null, 2), "utf8");
  return {
    ...sequence,
    files: {
      manifest: publicPath(manifestPath)
    }
  };
}

function downsampleSplatForSequence(splat, maxPoints = 100000) {
  const points = Array.isArray(splat?.points) ? splat.points : [];
  const limit = clamp(Number(maxPoints || 100000), 10000, 300000);
  if (points.length <= limit) {
    return {
      ...splat,
      pointCount: splat.pointCount || points.length,
      sequencePreviewPointCount: points.length
    };
  }
  const step = points.length / limit;
  const sampled = [];
  for (let i = 0; sampled.length < limit && Math.floor(i * step) < points.length; i += 1) {
    sampled.push(points[Math.floor(i * step)]);
  }
  return {
    ...splat,
    originalVertexCount: splat.originalVertexCount || splat.pointCount || points.length,
    pointCount: splat.pointCount || points.length,
    sequencePreviewPointCount: sampled.length,
    points: sampled
  };
}

app.post("/api/build-splat-job", (req, res) => {
  try {
    const { views, backend = "lightweight", quality = "balanced", previewMode = "auto", sceneMode = "subject", maskMode = "person", maskRefine = "clean", manualKeepMask = null, preprocessSize } = req.body || {};
    if (!Array.isArray(views)) {
      res.status(400).json({ error: "Missing generated views." });
      return;
    }

    const id = crypto.randomUUID();
    const jobPath = path.join(jobDir, id);
    fs.mkdirSync(jobPath, { recursive: true });
    const def = backend === "lightweight" ? { label: "Lightweight PLY" } : externalBackendDefs[backend];
    writeJobProgress(jobPath, {
      status: "queued",
      backend,
      backendLabel: def?.label || backend,
      quality,
      phase: "queued",
      message: "Queued GS3D build",
      percent: 0
    });

    const jobRecord = { status: "running", startedAt: new Date().toISOString(), promise: null };
    buildJobs.set(id, jobRecord);
    const promise = (async () => {
      try {
        writeJobProgress(jobPath, { status: "running", phase: "starting", message: "Starting GS3D build", percent: 2 });
        let result;
        if (backend !== "lightweight") {
          result = await buildExternalSplat({ backend, views, quality, jobId: id, previewMode, sceneMode, maskMode, maskRefine, manualKeepMask, preprocessSize });
        } else {
          writeJobProgress(jobPath, { phase: "sampling", message: "Sampling four views into a fast preview", percent: 20 });
          const splat = await buildLightweightSplat(views, quality);
          writeJobProgress(jobPath, { phase: "saving", message: "Saving lightweight splat artifacts", percent: 88 });
          const ply = splatToPly(splat);
          const saved = saveBuiltSplatArtifacts({ splat, ply });
          result = { splat: { ...splat, backend, backendLabel: "Lightweight PLY", jobId: id }, saved };
          writeJobProgress(jobPath, { status: "completed", phase: "completed", message: "GS3D build completed", percent: 100 });
        }
        jobRecord.status = "completed";
        jobRecord.result = result;
      } catch (error) {
        writeJobProgress(jobPath, {
          status: "failed",
          phase: "failed",
          message: error.message,
          error: error.message,
          percent: combinedJobProgress(id).percent || 0
        });
        jobRecord.status = "failed";
        jobRecord.error = error.message;
      }
    })();

    jobRecord.promise = promise;
    res.json({ jobId: id, progress: combinedJobProgress(id) });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/build-splat-job/:id", (req, res) => {
  const id = path.basename(req.params.id || "");
  const job = buildJobs.get(id);
  const progress = combinedJobProgress(id);
  const status = job?.status || progress.status || "unknown";
  if (status === "unknown") {
    res.status(404).json({ error: "Build job not found." });
    return;
  }
  res.json({
    jobId: id,
    status,
    progress,
    result: job?.result || null,
    error: job?.error || progress.error || null
  });
});

app.post("/api/build-sharp-sequence-job", (req, res) => {
  try {
    const {
      frames,
      frameLimit = 5,
      quality = "draft",
      previewMode = "auto",
      sceneMode = "subject",
      maskMode = "person",
      maskRefine = "clean",
      preprocessSize,
      sequencePreviewPoints = 100000
    } = req.body || {};

    if (!process.env.SHARP_COMMAND) {
      res.status(400).json({ error: "Apple SHARP is not configured. Set SHARP_COMMAND before starting the server." });
      return;
    }
    if (!Array.isArray(frames) || !frames.length) {
      res.status(400).json({ error: "Extract frames before starting a SHARP sequence." });
      return;
    }

    const selectedFrames = frames
      .filter((frame) => frame?.url)
      .slice(0, clamp(Number(frameLimit || 5), 1, 30));
    if (!selectedFrames.length) {
      res.status(400).json({ error: "No usable frames were provided." });
      return;
    }

    const id = crypto.randomUUID();
    const jobPath = path.join(jobDir, id);
    fs.mkdirSync(jobPath, { recursive: true });
    writeJobProgress(jobPath, {
      status: "queued",
      backend: "sharp-sequence",
      backendLabel: "Apple SHARP sequence",
      quality,
      phase: "queued",
      message: `Queued ${selectedFrames.length} SHARP frames`,
      percent: 0,
      totalFrames: selectedFrames.length,
      frame: 0
    });

    const jobRecord = { status: "running", startedAt: new Date().toISOString(), promise: null };
    buildJobs.set(id, jobRecord);
    const promise = (async () => {
      const outputs = [];
      try {
        for (let index = 0; index < selectedFrames.length; index += 1) {
          const frame = selectedFrames[index];
          const percent = Math.round((index / selectedFrames.length) * 94);
          writeJobProgress(jobPath, {
            status: "running",
            phase: "sharp",
            message: `Running Apple SHARP frame ${index + 1}/${selectedFrames.length}`,
            percent,
            totalFrames: selectedFrames.length,
            frame: index + 1
          });
          const result = await buildExternalSplat({
            backend: "sharp",
            views: [{
              id: "front",
              label: `Frame ${index + 1}`,
              yaw: 0,
              url: frame.url,
              status: "video-frame",
              timestamp: frame.timestamp
            }],
            quality,
            jobId: `${id}-frame-${String(index + 1).padStart(4, "0")}`,
            previewMode,
            sceneMode,
            maskMode,
            maskRefine,
            preprocessSize
          });
          const previewSplat = downsampleSplatForSequence(result.splat, sequencePreviewPoints);
          const sequencePath = path.join(sharpSequenceDir, id);
          fs.mkdirSync(sequencePath, { recursive: true });
          const previewPath = path.join(sequencePath, `frame-${String(index + 1).padStart(4, "0")}.preview.json`);
          fs.writeFileSync(previewPath, JSON.stringify(previewSplat), "utf8");
          outputs.push({
            index,
            timestamp: frame.timestamp,
            source: frame.url,
            saved: result.saved,
            preview: publicPath(previewPath),
            previewPointCount: previewSplat.points.length,
            pointCount: result.splat?.originalVertexCount || result.splat?.pointCount || result.splat?.points?.length || 0
          });
        }

        const sequence = saveSharpSequenceManifest({
          id,
          format: "portrait-splat-forge.sharp-sequence.v1",
          createdAt: new Date().toISOString(),
          backend: "sharp",
          backendLabel: "Apple SHARP sequence",
          frameCount: outputs.length,
          quality,
          previewMode,
          sequencePreviewPoints: clamp(Number(sequencePreviewPoints || 100000), 10000, 300000),
          frames: outputs
        });
        const firstJson = outputs[0]?.preview ? outputPathFromPublicUrl(outputs[0].preview) : outputs[0]?.saved?.files?.json ? outputPathFromPublicUrl(outputs[0].saved.files.json) : null;
        const firstSplat = firstJson && fs.existsSync(firstJson) ? JSON.parse(fs.readFileSync(firstJson, "utf8")) : null;
        const result = {
          sequence,
          splat: firstSplat ? { ...firstSplat, backend: "sharp-sequence", backendLabel: "Apple SHARP sequence", jobId: id, previewMode } : null,
          saved: outputs[0]?.saved || null
        };
        writeJobProgress(jobPath, {
          status: "completed",
          phase: "completed",
          message: `Completed ${outputs.length} SHARP frames`,
          percent: 100,
          totalFrames: selectedFrames.length,
          frame: outputs.length
        });
        jobRecord.status = "completed";
        jobRecord.result = result;
      } catch (error) {
        writeJobProgress(jobPath, {
          status: "failed",
          phase: "failed",
          message: error.message,
          error: error.message,
          percent: combinedJobProgress(id).percent || 0,
          totalFrames: selectedFrames.length,
          frame: outputs.length
        });
        jobRecord.status = "failed";
        jobRecord.error = error.message;
      }
    })();

    jobRecord.promise = promise;
    res.json({ jobId: id, progress: combinedJobProgress(id) });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/build-sharp-sequence-job/:id", (req, res) => {
  const id = path.basename(req.params.id || "");
  const job = buildJobs.get(id);
  const progress = combinedJobProgress(id);
  const status = job?.status || progress.status || "unknown";
  if (status === "unknown") {
    res.status(404).json({ error: "SHARP sequence job not found." });
    return;
  }
  res.json({
    jobId: id,
    status,
    progress,
    result: job?.result || null,
    error: job?.error || progress.error || null
  });
});

app.get("/api/recent-splats", (req, res) => {
  const limit = clamp(Number(req.query.limit || 12), 1, 50);
  res.json({ outputs: listRecentSplatOutputs(limit) });
});

app.get("/api/health", (_req, res) => {
  res.json({
    ok: true,
    imageModel: process.env.OPENAI_IMAGE_MODEL || "gpt-image-2",
    responsesModel: process.env.OPENAI_RESPONSES_MODEL || "gpt-5.5",
    video: {
      ffmpegCommand: process.env.FFMPEG_COMMAND || "ffmpeg",
      ffprobeCommand: process.env.FFPROBE_COMMAND || "ffprobe",
      maxUploadMb: 1024
    },
    openai: Boolean(process.env.OPENAI_API_KEY),
    splatBackends: availableSplatBackends()
  });
});

app.listen(port, "127.0.0.1", () => {
  console.log(`Portrait Splat Forge API listening at http://127.0.0.1:${port}`);
});
