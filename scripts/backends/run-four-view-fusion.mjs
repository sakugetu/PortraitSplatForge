import fs from "node:fs";
import path from "node:path";
import sharp from "sharp";

const inputDir = process.env.GS3D_INPUT_DIR || process.argv[2];
const outputDir = process.env.GS3D_OUTPUT_DIR || process.argv[3];
const quality = process.env.GS3D_QUALITY || "balanced";
const surfaceMode = process.env.FOUR_VIEW_FUSION_SURFACE_MODE || "direct";

if (!inputDir || !outputDir) {
  throw new Error("run-four-view-fusion requires GS3D_INPUT_DIR and GS3D_OUTPUT_DIR.");
}

fs.mkdirSync(outputDir, { recursive: true });

const settingsByQuality = {
  draft: { size: 384, step: 4, angularStep: 9, maxPoints: 220000 },
  balanced: { size: 512, step: 3, angularStep: 6, maxPoints: 420000 },
  high: { size: 768, step: 1, angularStep: 3, maxPoints: 1200000 }
};
const settings = settingsByQuality[quality] || settingsByQuality.balanced;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

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
    backend: "fourviewfusion",
    status: "running",
    quality,
    ...patch,
    updatedAt: new Date().toISOString()
  }, null, 2), "utf8");
}

async function loadView(id) {
  const imagePath = path.join(inputDir, "images", `${id}.png`);
  if (!fs.existsSync(imagePath)) {
    throw new Error(`Missing input image: ${imagePath}`);
  }
  const { data, info } = await sharp(imagePath)
    .resize(settings.size, settings.size, { fit: "contain", background: { r: 0, g: 0, b: 0, alpha: 0 } })
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return { id, data, width: info.width, height: info.height };
}

function alphaAt(view, x, y) {
  if (x < 0 || x >= view.width || y < 0 || y >= view.height) return 0;
  return view.data[(y * view.width + x) * 4 + 3];
}

function colorAt(view, x, y) {
  const xx = clamp(Math.round(x), 0, view.width - 1);
  const yy = clamp(Math.round(y), 0, view.height - 1);
  const offset = (yy * view.width + xx) * 4;
  return [view.data[offset], view.data[offset + 1], view.data[offset + 2], view.data[offset + 3]];
}

function rowBounds(view, threshold = 38) {
  const bounds = Array.from({ length: view.height }, () => ({ valid: false, min: view.width, max: -1 }));
  for (let y = 0; y < view.height; y++) {
    for (let x = 0; x < view.width; x++) {
      if (alphaAt(view, x, y) <= threshold) continue;
      bounds[y].valid = true;
      bounds[y].min = Math.min(bounds[y].min, x);
      bounds[y].max = Math.max(bounds[y].max, x);
    }
  }
  return smoothBounds(bounds);
}

function smoothBounds(bounds) {
  const result = bounds.map((bound) => ({ ...bound }));
  for (let pass = 0; pass < 2; pass++) {
    for (let y = 1; y < bounds.length - 1; y++) {
      if (!result[y].valid) continue;
      const prev = result[y - 1].valid ? result[y - 1] : result[y];
      const next = result[y + 1].valid ? result[y + 1] : result[y];
      result[y] = {
        valid: true,
        min: Math.round((prev.min + result[y].min * 2 + next.min) / 4),
        max: Math.round((prev.max + result[y].max * 2 + next.max) / 4)
      };
    }
  }
  return result;
}

function normalizeImageX(x, size) {
  return (x / (size - 1) - 0.5) * 2.0;
}

function normalizeImageY(y, size) {
  return -(y / (size - 1) - 0.5) * 2.18;
}

function imageXFromCoord(value, size) {
  return clamp(Math.round((value / 2.0 + 0.5) * (size - 1)), 0, size - 1);
}

function imageYFromCoord(value, size) {
  return clamp(Math.round((-value / 2.18 + 0.5) * (size - 1)), 0, size - 1);
}

function insideVisualHull({ x, y, z, views }) {
  const size = settings.size;
  const frontX = imageXFromCoord(x, size);
  const frontY = imageYFromCoord(y, size);
  const sideX = imageXFromCoord(z, size);
  return (
    alphaAt(views.front, frontX, frontY) > 22 &&
    alphaAt(views.back, frontX, frontY) > 22 &&
    alphaAt(views.left45, sideX, frontY) > 22 &&
    alphaAt(views.right45, sideX, frontY) > 22
  );
}

function mixColors(colors) {
  let r = 0;
  let g = 0;
  let b = 0;
  let total = 0;
  for (const [color, weight] of colors) {
    const alpha = color[3] / 255;
    const w = weight * alpha;
    r += color[0] * w;
    g += color[1] * w;
    b += color[2] * w;
    total += w;
  }
  if (total <= 0.001) return [190, 180, 160];
  return [Math.round(r / total), Math.round(g / total), Math.round(b / total)];
}

function surfaceColor({ x, y, z, normal, views }) {
  const size = settings.size;
  const py = imageYFromCoord(y, size);
  const px = imageXFromCoord(x, size);
  const pz = imageXFromCoord(z, size);
  const weights = [
    [colorAt(views.front, px, py), Math.max(0.08, normal.z)],
    [colorAt(views.back, px, py), Math.max(0.08, -normal.z)],
    [colorAt(views.left45, pz, py), Math.max(0.08, -normal.x)],
    [colorAt(views.right45, pz, py), Math.max(0.08, normal.x)]
  ];
  return mixColors(weights);
}

function addPoint(points, point) {
  if (!Number.isFinite(point.x) || !Number.isFinite(point.y) || !Number.isFinite(point.z)) return;
  points.push(point);
}

function addViewProjectedShell({ points, view, boundsA, boundsB, axis, side, scale }) {
  const size = settings.size;
  for (let py = 0; py < size; py += settings.step) {
    const a = boundsA[py];
    const b = boundsB[py];
    if (!a.valid || !b.valid) continue;
    const y = normalizeImageY(py, size);
    const minA = normalizeImageX(a.min, size);
    const maxA = normalizeImageX(a.max, size);
    const minB = normalizeImageX(b.min, size) * 0.72;
    const maxB = normalizeImageX(b.max, size) * 0.72;
    for (let px = 0; px < size; px += settings.step) {
      const color = colorAt(view, px, py);
      if (color[3] <= 28) continue;
      if (axis === "z") {
        const x = normalizeImageX(px, size);
        if (x < minA || x > maxA) continue;
        const cx = (minA + maxA) / 2;
        const cz = (minB + maxB) / 2;
        const rx = Math.max(0.002, (maxA - minA) / 2);
        const rz = Math.max(0.002, (maxB - minB) / 2);
        const normalized = clamp((x - cx) / rx, -1, 1);
        const z = cz + side * Math.sqrt(Math.max(0, 1 - normalized * normalized)) * rz;
        addPoint(points, { x, y, z, r: color[0], g: color[1], b: color[2], scale });
      } else {
        const z = normalizeImageX(px, size) * 0.72;
        if (z < minB || z > maxB) continue;
        const cx = (minA + maxA) / 2;
        const cz = (minB + maxB) / 2;
        const rx = Math.max(0.002, (maxA - minA) / 2);
        const rz = Math.max(0.002, (maxB - minB) / 2);
        const normalized = clamp((z - cz) / rz, -1, 1);
        const x = cx + side * Math.sqrt(Math.max(0, 1 - normalized * normalized)) * rx;
        addPoint(points, { x, y, z, r: color[0], g: color[1], b: color[2], scale });
      }
    }
  }
}

function addEllipticalSurface({ points, views, silhouetteBounds, sideBounds, scale }) {
  const size = settings.size;
  for (let py = 0; py < size; py += settings.step) {
    const fb = silhouetteBounds[py];
    const sb = sideBounds[py];
    if (!fb.valid || !sb.valid) continue;
    const y = normalizeImageY(py, size);
    const minX = normalizeImageX(fb.min, size);
    const maxX = normalizeImageX(fb.max, size);
    const minZ = normalizeImageX(sb.min, size) * 0.72;
    const maxZ = normalizeImageX(sb.max, size) * 0.72;
    const cx = (minX + maxX) / 2;
    const cz = (minZ + maxZ) / 2;
    const rx = Math.max(0.018, (maxX - minX) / 2);
    const rz = Math.max(0.018, (maxZ - minZ) / 2);
    const pyImage = imageYFromCoord(y, size);
    for (let degrees = 0; degrees < 360; degrees += settings.angularStep) {
      const theta = (degrees * Math.PI) / 180;
      const nx = Math.sin(theta);
      const nz = Math.cos(theta);
      const x = cx + nx * rx;
      const z = cz + nz * rz;
      const px = imageXFromCoord(x, size);
      const pz = imageXFromCoord(z / 0.72, size);
      const colors = [
        [colorAt(views.front, px, pyImage), Math.max(0, nz) ** 1.6],
        [colorAt(views.back, px, pyImage), Math.max(0, -nz) ** 1.6],
        [colorAt(views.left45, pz, pyImage), Math.max(0, -nx) ** 1.6],
        [colorAt(views.right45, pz, pyImage), Math.max(0, nx) ** 1.6]
      ];
      const [r, g, b] = mixColors(colors);
      const alpha = Math.max(
        alphaAt(views.front, px, pyImage),
        alphaAt(views.back, px, pyImage),
        alphaAt(views.left45, pz, pyImage),
        alphaAt(views.right45, pz, pyImage)
      );
      if (alpha <= 20) continue;
      addPoint(points, { x, y, z, r, g, b, scale: scale * 0.92 });
    }
  }
}

function buildSurface(views) {
  const size = settings.size;
  const frontBounds = rowBounds(views.front);
  const backBounds = rowBounds(views.back);
  const silhouetteBounds = frontBounds.map((front, y) => {
    const back = backBounds[y];
    if (!front.valid && !back.valid) return { valid: false, min: size, max: -1 };
    if (!front.valid) return back;
    if (!back.valid) return front;
    return { valid: true, min: Math.min(front.min, back.min), max: Math.max(front.max, back.max) };
  });
  const sideBounds = rowBounds(views.left45).map((left, y) => {
    const right = rowBounds(views.right45)[y];
    if (!left.valid && !right.valid) return { valid: false, min: size, max: -1 };
    if (!left.valid) return right;
    if (!right.valid) return left;
    return { valid: true, min: Math.min(left.min, right.min), max: Math.max(left.max, right.max) };
  });
  const points = [];
  const scale = quality === "high" ? 0.012 : quality === "balanced" ? 0.015 : 0.02;

  addViewProjectedShell({ points, view: views.front, boundsA: silhouetteBounds, boundsB: sideBounds, axis: "z", side: -1, scale });
  addViewProjectedShell({ points, view: views.back, boundsA: silhouetteBounds, boundsB: sideBounds, axis: "z", side: 1, scale });
  addViewProjectedShell({ points, view: views.left45, boundsA: silhouetteBounds, boundsB: sideBounds, axis: "x", side: -1, scale });
  addViewProjectedShell({ points, view: views.right45, boundsA: silhouetteBounds, boundsB: sideBounds, axis: "x", side: 1, scale });
  if (surfaceMode === "ellipse") {
    addEllipticalSurface({ points, views, silhouetteBounds, sideBounds, scale });
  }

  for (let py = 0; py < size; py += settings.step) {
    const fb = silhouetteBounds[py];
    const sb = sideBounds[py];
    if (!fb.valid || !sb.valid) continue;
    const y = normalizeImageY(py, size);
    for (let px = fb.min; px <= fb.max; px += settings.step) {
      if (alphaAt(views.front, px, py) <= 28 && alphaAt(views.back, px, py) <= 28) continue;
      const x = normalizeImageX(px, size);
      const zMin = normalizeImageX(sb.min, size) * 0.72;
      const zMax = normalizeImageX(sb.max, size) * 0.72;
      const zFront = zMin;
      const zBack = zMax;
      if (insideVisualHull({ x, y, z: zFront, views })) {
        const [r, g, b] = surfaceColor({ x, y, z: zFront, normal: { x: 0, z: 1 }, views });
        addPoint(points, { x, y, z: zFront, r, g, b, scale });
      }
      if (insideVisualHull({ x, y, z: zBack, views })) {
        const [r, g, b] = surfaceColor({ x, y, z: zBack, normal: { x: 0, z: -1 }, views });
        addPoint(points, { x, y, z: zBack, r, g, b, scale });
      }
      for (let inner = 1; inner < settings.innerSteps; inner++) {
        const t = inner / settings.innerSteps;
        const z = zFront * (1 - t) + zBack * t;
        if (!insideVisualHull({ x, y, z, views })) continue;
        const [r, g, b] = surfaceColor({ x, y, z, normal: { x: 0, z: t < 0.5 ? 1 : -1 }, views });
        addPoint(points, { x, y, z, r, g, b, scale: scale * 0.72 });
      }
    }

    for (let pz = sb.min; pz <= sb.max; pz += settings.step) {
      if (alphaAt(views.left45, pz, py) <= 28 && alphaAt(views.right45, pz, py) <= 28) continue;
      const z = normalizeImageX(pz, size) * 0.72;
      const xMin = normalizeImageX(fb.min, size);
      const xMax = normalizeImageX(fb.max, size);
      for (const [x, normal] of [[xMin, { x: -1, z: 0 }], [xMax, { x: 1, z: 0 }]]) {
        if (!insideVisualHull({ x, y, z, views })) continue;
        const [r, g, b] = surfaceColor({ x, y, z, normal, views });
        addPoint(points, { x, y, z, r, g, b, scale });
      }
    }
  }

  if (points.length <= settings.maxPoints) return points;
  const stride = points.length / settings.maxPoints;
  const sampled = [];
  for (let index = 0; index < settings.maxPoints; index++) {
    sampled.push(points[Math.floor(index * stride)]);
  }
  return sampled;
}

function writePly(points, filePath) {
  const header = [
    "ply",
    "format ascii 1.0",
    "comment portrait_splat_forge_coords z_up",
    `element vertex ${points.length}`,
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
  const body = points
    .map((p) => `${p.x.toFixed(5)} ${p.z.toFixed(5)} ${p.y.toFixed(5)} ${clamp(p.r, 0, 255)} ${clamp(p.g, 0, 255)} ${clamp(p.b, 0, 255)} 245 ${p.scale.toFixed(5)}`)
    .join("\n");
  fs.writeFileSync(filePath, `${header}\n${body}\n`, "utf8");
}

writeProgress({ phase: "loading", message: "Loading four masked orthographic views", percent: 8, step: 1, totalSteps: 3 });
const views = {
  front: await loadView("front"),
  left45: await loadView("left45"),
  right45: await loadView("right45"),
  back: await loadView("back")
};

writeProgress({ phase: "fusing", message: "Building a four-view visual hull and color splats", percent: 35, step: 2, totalSteps: 3 });
const points = buildSurface(views);
if (points.length < 1000) {
  throw new Error(`Four-view fusion produced too few points: ${points.length}`);
}

writeProgress({ phase: "saving", message: `Writing ${points.length} fused Gaussian points`, percent: 86, step: 3, totalSteps: 3 });
writePly(points, path.join(outputDir, "output.ply"));
writeProgress({ status: "completed", phase: "completed", message: "Four-view fused PLY is ready", percent: 100, step: 3, totalSteps: 3, pointCount: points.length });
