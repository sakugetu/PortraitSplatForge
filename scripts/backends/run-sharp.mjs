import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";

const inputDir = process.env.GS3D_INPUT_DIR || process.argv[2];
const outputDir = process.env.GS3D_OUTPUT_DIR || process.argv[3];
const frontImage = path.join(inputDir || "", "images", "front.png");
const rotateY180 = process.env.SHARP_ROTATE_Y_180 !== "0";
const mirrorX = process.env.SHARP_MIRROR_X !== "0";
const densifyFactor = Math.max(1, Math.min(4, Math.round(Number(process.env.SHARP_DENSIFY_FACTOR || 1))));
const densifyJitter = Number(process.env.SHARP_DENSIFY_JITTER || 0.0018);
const filterBackground = process.env.SHARP_FILTER_BACKGROUND !== "0";

if (!inputDir || !outputDir) {
  throw new Error("run-sharp requires GS3D_INPUT_DIR and GS3D_OUTPUT_DIR.");
}
if (!fs.existsSync(frontImage)) {
  throw new Error(`Missing SHARP input image: ${frontImage}`);
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
    backend: "sharp",
    status: "running",
    ...previous,
    ...patch,
    updatedAt: new Date().toISOString()
  }, null, 2), "utf8");
}

writeProgress({
  phase: "preparing",
  message: "Preparing front view for Apple SHARP",
  percent: 15,
  step: 1,
  totalSteps: 3
});

const sharpInputDir = path.join(outputDir, "_sharp_input");
fs.mkdirSync(sharpInputDir, { recursive: true });
fs.copyFileSync(frontImage, path.join(sharpInputDir, "front.png"));

function run(command, args, cwd = process.cwd()) {
  return new Promise((resolve, reject) => {
    const stdoutPath = path.join(outputDir, "sharp.stdout.log");
    const stderrPath = path.join(outputDir, "sharp.stderr.log");
    fs.writeFileSync(stdoutPath, "", "utf8");
    fs.writeFileSync(stderrPath, "", "utf8");
    const child = spawn(command, args, {
      cwd,
      shell: true,
      windowsHide: true,
      env: process.env
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      const text = chunk.toString();
      stdout += text;
      fs.appendFileSync(stdoutPath, text, "utf8");
      if (/Processing/i.test(text)) {
        writeProgress({ phase: "predicting", message: text.trim().slice(-180), percent: 55, step: 2, totalSteps: 3 });
      }
    });
    child.stderr.on("data", (chunk) => {
      const text = chunk.toString();
      stderr += text;
      fs.appendFileSync(stderrPath, text, "utf8");
      if (/download|predict|processing|model|checkpoint/i.test(text)) {
        writeProgress({ phase: "predicting", message: text.trim().slice(-180), percent: 55, step: 2, totalSteps: 3 });
      }
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve();
        return;
      }
      reject(new Error(`SHARP exited with code ${code}.\n${stderr || stdout}`.trim()));
    });
  });
}

const predictArgs = ["predict", "-i", sharpInputDir, "-o", outputDir];
if (process.env.SHARP_CHECKPOINT) {
  predictArgs.push("-c", process.env.SHARP_CHECKPOINT);
}

if (process.env.SHARP_CONDA_ENV) {
  writeProgress({ phase: "predicting", message: "Running Apple SHARP prediction", percent: 35, step: 2, totalSteps: 3 });
  await run("conda", ["run", "--no-capture-output", "-n", process.env.SHARP_CONDA_ENV, "sharp", ...predictArgs]);
} else {
  writeProgress({ phase: "predicting", message: "Running Apple SHARP prediction", percent: 35, step: 2, totalSteps: 3 });
  await run(process.env.SHARP_CLI || "sharp", predictArgs);
}

function listFilesRecursive(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const resolved = path.join(dir, entry.name);
    return entry.isDirectory() ? listFilesRecursive(resolved) : [resolved];
  });
}

const ply = listFilesRecursive(outputDir).find((file) => file.toLowerCase().endsWith(".ply"));
if (!ply) {
  throw new Error("SHARP completed but no .ply file was found.");
}

const finalPath = path.join(outputDir, "output.ply");
if (path.resolve(ply) !== path.resolve(finalPath)) {
  fs.copyFileSync(ply, finalPath);
}

if (rotateY180) {
  rotatePlyY180(finalPath);
}

if (mirrorX) {
  mirrorPlyX(finalPath);
}

if (filterBackground) {
  writeProgress({
    phase: "filtering",
    message: "Filtering SHARP background and low-confidence splats",
    percent: 82,
    step: 3,
    totalSteps: densifyFactor > 1 ? 5 : 4
  });
  filterBinaryPly(finalPath);
}

if (densifyFactor > 1) {
  writeProgress({
    phase: "densifying",
    message: `Densifying SHARP splats x${densifyFactor}`,
    percent: 88,
    step: filterBackground ? 4 : 3,
    totalSteps: filterBackground ? 5 : 4
  });
  densifyBinaryPly(finalPath, densifyFactor, densifyJitter);
}

function shToByte(value) {
  return Math.max(0, Math.min(255, Math.round((Number(value || 0) * 0.28209479177387814 + 0.5) * 255)));
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function filterBinaryPly(filePath) {
  const buffer = fs.readFileSync(filePath);
  const headerEnd = buffer.indexOf("end_header");
  if (headerEnd < 0) return;
  const dataStart = buffer.indexOf("\n", headerEnd) + 1;
  const header = buffer.subarray(0, dataStart).toString("utf8");
  if (!/format binary_little_endian 1\.0/.test(header)) return;
  const layout = plyVertexLayout(header);
  if (!layout.vertexCount || !layout.rowSize) return;
  const nameToOffset = Object.fromEntries(layout.typedProperties.map((property, index) => [property.name, layout.offsets[index]]));
  const vertexBytes = layout.vertexCount * layout.rowSize;
  const vertexData = buffer.subarray(dataStart, dataStart + vertexBytes);
  const tail = buffer.subarray(dataStart + vertexBytes);
  const keptRows = [];
  const radiusValues = [];
  for (let vertex = 0; vertex < layout.vertexCount; vertex++) {
    const offset = vertex * layout.rowSize;
    const x = vertexData.readFloatLE(offset + nameToOffset.x);
    const y = vertexData.readFloatLE(offset + nameToOffset.y);
    const z = vertexData.readFloatLE(offset + nameToOffset.z);
    if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
      radiusValues.push(Math.hypot(x, y, z));
    }
  }
  radiusValues.sort((a, b) => a - b);
  const radiusCut = radiusValues[Math.floor(radiusValues.length * Number(process.env.SHARP_FILTER_RADIUS_QUANTILE || 0.985))] || Infinity;
  for (let vertex = 0; vertex < layout.vertexCount; vertex++) {
    const offset = vertex * layout.rowSize;
    const x = vertexData.readFloatLE(offset + nameToOffset.x);
    const y = vertexData.readFloatLE(offset + nameToOffset.y);
    const z = vertexData.readFloatLE(offset + nameToOffset.z);
    const opacity = nameToOffset.opacity !== undefined ? sigmoid(vertexData.readFloatLE(offset + nameToOffset.opacity)) : 1;
    const r = nameToOffset.red !== undefined ? vertexData.readUInt8(offset + nameToOffset.red) : shToByte(vertexData.readFloatLE(offset + nameToOffset.f_dc_0));
    const g = nameToOffset.green !== undefined ? vertexData.readUInt8(offset + nameToOffset.green) : shToByte(vertexData.readFloatLE(offset + nameToOffset.f_dc_1));
    const b = nameToOffset.blue !== undefined ? vertexData.readUInt8(offset + nameToOffset.blue) : shToByte(vertexData.readFloatLE(offset + nameToOffset.f_dc_2));
    const luma = (r + g + b) / 3;
    const chroma = Math.max(r, g, b) - Math.min(r, g, b);
    const tooDark = luma < Number(process.env.SHARP_FILTER_MIN_LUMA || 18);
    const tooWhite =
      luma > Number(process.env.SHARP_FILTER_MAX_WHITE_LUMA || 248) &&
      chroma < Number(process.env.SHARP_FILTER_MAX_WHITE_CHROMA || 7);
    const tooKeyed =
      r > Number(process.env.SHARP_FILTER_KEY_MIN_R || 210) &&
      g < Number(process.env.SHARP_FILTER_KEY_MAX_G || 70) &&
      b > Number(process.env.SHARP_FILTER_KEY_MIN_B || 210);
    const keySpill =
      r > Number(process.env.SHARP_FILTER_KEY_SPILL_MIN_R || 135) &&
      b > Number(process.env.SHARP_FILTER_KEY_SPILL_MIN_B || 135) &&
      g < Math.min(r, b) * Number(process.env.SHARP_FILTER_KEY_SPILL_GREEN_RATIO || 0.72);
    const lowOpacity = opacity < Number(process.env.SHARP_FILTER_MIN_OPACITY || 0.035);
    const outlier = Math.hypot(x, y, z) > radiusCut;
    if (tooDark || tooWhite || tooKeyed || keySpill || lowOpacity || outlier) continue;
    keptRows.push(vertexData.subarray(offset, offset + layout.rowSize));
  }
  const newHeader = header.replace(/element vertex \d+/, `element vertex ${keptRows.length}`);
  const output = Buffer.alloc(Buffer.byteLength(newHeader) + keptRows.length * layout.rowSize + tail.length);
  output.write(newHeader, 0, "utf8");
  let writeOffset = Buffer.byteLength(newHeader);
  for (const row of keptRows) {
    row.copy(output, writeOffset);
    writeOffset += layout.rowSize;
  }
  tail.copy(output, writeOffset);
  fs.writeFileSync(filePath, output);
}

function plyVertexLayout(header) {
  const typeSizes = {
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
  };
  const lines = header.split(/\r?\n/);
  const vertexCount = Number(lines.find((line) => line.startsWith("element vertex "))?.split(/\s+/)[2] || 0);
  const typedProperties = [];
  let inVertex = false;
  for (const line of lines) {
    if (line.startsWith("element vertex ")) {
      inVertex = true;
      continue;
    }
    if (inVertex && line.startsWith("element ")) break;
    if (inVertex && line.startsWith("property ")) {
      const [, type, name] = line.trim().split(/\s+/);
      typedProperties.push({ type, name, size: typeSizes[type] || 4 });
    }
  }
  const offsets = [];
  const rowSize = typedProperties.reduce((offset, property) => {
    offsets.push(offset);
    return offset + property.size;
  }, 0);
  return {
    lines,
    vertexCount,
    typedProperties,
    offsets,
    rowSize,
    xOffset: offsets[typedProperties.findIndex((property) => property.name === "x")],
    yOffset: offsets[typedProperties.findIndex((property) => property.name === "y")],
    zOffset: offsets[typedProperties.findIndex((property) => property.name === "z")]
  };
}

function densifyBinaryPly(filePath, factor, jitter) {
  const buffer = fs.readFileSync(filePath);
  const headerEnd = buffer.indexOf("end_header");
  if (headerEnd < 0) return;
  const dataStart = buffer.indexOf("\n", headerEnd) + 1;
  const header = buffer.subarray(0, dataStart).toString("utf8");
  if (!/format binary_little_endian 1\.0/.test(header)) return;
  const layout = plyVertexLayout(header);
  if (!layout.vertexCount || !layout.rowSize || layout.xOffset === undefined || layout.yOffset === undefined || layout.zOffset === undefined) return;

  const vertexBytes = layout.vertexCount * layout.rowSize;
  const vertexData = buffer.subarray(dataStart, dataStart + vertexBytes);
  const tail = buffer.subarray(dataStart + vertexBytes);
  const newVertexCount = layout.vertexCount * factor;
  const newHeader = header.replace(/element vertex \d+/, `element vertex ${newVertexCount}`);
  const output = Buffer.alloc(Buffer.byteLength(newHeader) + newVertexCount * layout.rowSize + tail.length);
  output.write(newHeader, 0, "utf8");
  let writeOffset = Buffer.byteLength(newHeader);

  const offsets = [
    [0, 0, 0],
    [jitter, 0, 0],
    [-jitter, 0, 0],
    [0, jitter, 0],
    [0, -jitter, 0],
    [0, 0, jitter],
    [0, 0, -jitter]
  ];

  for (let vertex = 0; vertex < layout.vertexCount; vertex++) {
    const sourceOffset = vertex * layout.rowSize;
    for (let copy = 0; copy < factor; copy++) {
      vertexData.copy(output, writeOffset, sourceOffset, sourceOffset + layout.rowSize);
      if (copy > 0) {
        const [jx, jy, jz] = offsets[copy % offsets.length];
        output.writeFloatLE(output.readFloatLE(writeOffset + layout.xOffset) + jx, writeOffset + layout.xOffset);
        output.writeFloatLE(output.readFloatLE(writeOffset + layout.yOffset) + jy, writeOffset + layout.yOffset);
        output.writeFloatLE(output.readFloatLE(writeOffset + layout.zOffset) + jz, writeOffset + layout.zOffset);
      }
      writeOffset += layout.rowSize;
    }
  }
  tail.copy(output, writeOffset);
  fs.writeFileSync(filePath, output);
}

function rotatePlyY180(filePath) {
  const buffer = fs.readFileSync(filePath);
  const headerEnd = buffer.indexOf("end_header");
  if (headerEnd < 0) return;
  const dataStart = buffer.indexOf("\n", headerEnd) + 1;
  const header = buffer.subarray(0, dataStart).toString("utf8");
  const { lines, vertexCount } = plyVertexLayout(header);
  if (!vertexCount) return;
  const properties = [];
  let inVertex = false;
  for (const line of lines) {
    if (line.startsWith("element vertex ")) {
      inVertex = true;
      continue;
    }
    if (inVertex && line.startsWith("element ")) break;
    if (inVertex && line.startsWith("property ")) {
      properties.push(line.trim().split(/\s+/)[2]);
    }
  }
  const xIndex = properties.indexOf("x");
  const zIndex = properties.indexOf("z");
  if (xIndex < 0 || zIndex < 0) return;

  if (/format binary_little_endian 1\.0/.test(header)) {
    const { rowSize, xOffset, zOffset } = plyVertexLayout(header);
    const rotated = Buffer.from(buffer);
    for (let vertex = 0; vertex < vertexCount; vertex++) {
      const rowOffset = dataStart + vertex * rowSize;
      rotated.writeFloatLE(-rotated.readFloatLE(rowOffset + xOffset), rowOffset + xOffset);
      rotated.writeFloatLE(-rotated.readFloatLE(rowOffset + zOffset), rowOffset + zOffset);
    }
    fs.writeFileSync(filePath, rotated);
    return;
  }

  if (!/format ascii 1\.0/.test(header)) return;
  const body = buffer.subarray(dataStart).toString("utf8").trimEnd().split(/\r?\n/).map((line) => {
    const parts = line.trim().split(/\s+/);
    if (parts.length <= Math.max(xIndex, zIndex)) return line;
    parts[xIndex] = String(-Number(parts[xIndex]));
    parts[zIndex] = String(-Number(parts[zIndex]));
    return parts.join(" ");
  });
  fs.writeFileSync(filePath, `${header}${body.join("\n")}\n`, "utf8");
}

function mirrorPlyX(filePath) {
  const buffer = fs.readFileSync(filePath);
  const headerEnd = buffer.indexOf("end_header");
  if (headerEnd < 0) return;
  const dataStart = buffer.indexOf("\n", headerEnd) + 1;
  const header = buffer.subarray(0, dataStart).toString("utf8");
  const vertexCount = Number(header.split(/\r?\n/).find((line) => line.startsWith("element vertex "))?.split(/\s+/)[2] || 0);
  if (!vertexCount) return;

  if (/format binary_little_endian 1\.0/.test(header)) {
    const { rowSize, xOffset } = plyVertexLayout(header);
    if (!rowSize || xOffset === undefined) return;
    const mirrored = Buffer.from(buffer);
    for (let vertex = 0; vertex < vertexCount; vertex++) {
      const rowOffset = dataStart + vertex * rowSize;
      mirrored.writeFloatLE(-mirrored.readFloatLE(rowOffset + xOffset), rowOffset + xOffset);
    }
    fs.writeFileSync(filePath, mirrored);
    return;
  }

  if (/format ascii 1\.0/.test(header)) {
    const lines = header.split(/\r?\n/);
    const properties = [];
    let inVertex = false;
    for (const line of lines) {
      if (line.startsWith("element vertex ")) {
        inVertex = true;
        continue;
      }
      if (inVertex && line.startsWith("element ")) break;
      if (inVertex && line.startsWith("property ")) {
        properties.push(line.trim().split(/\s+/)[2]);
      }
    }
    const xIndex = properties.indexOf("x");
    if (xIndex < 0) return;
    const body = buffer.subarray(dataStart).toString("utf8").split(/\r?\n/);
    const mirroredBody = body.map((line, index) => {
      if (index >= vertexCount || !line.trim()) return line;
      const values = line.trim().split(/\s+/);
      values[xIndex] = String(-Number(values[xIndex]));
      return values.join(" ");
    });
    fs.writeFileSync(filePath, header + mirroredBody.join("\n"));
  }
}

writeProgress({
  status: "completed",
  phase: "completed",
  message: "Apple SHARP PLY is ready",
  percent: 100,
  step: filterBackground ? (densifyFactor > 1 ? 5 : 4) : (densifyFactor > 1 ? 4 : 3),
  totalSteps: filterBackground ? (densifyFactor > 1 ? 5 : 4) : (densifyFactor > 1 ? 4 : 3)
});
