import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { VRButton } from "three/examples/jsm/webxr/VRButton.js";
import {
  BadgeCheck,
  Box,
  Brush,
  Camera,
  CameraIcon,
  ChevronRight,
  Download,
  Eraser,
  Eye,
  Film,
  Gauge,
  ImagePlus,
  Loader2,
  Pause,
  Play,
  RefreshCw,
  RotateCcw,
  Sparkles,
  Trash2
} from "lucide-react";
import "./styles.css";

const steps = [
  "Upload",
  "Segment",
  "Generate views",
  "Build splat",
  "Preview"
];

function useHealth() {
  const [health, setHealth] = useState(null);
  useEffect(() => {
    fetch("/api/health").then((r) => r.json()).then(setHealth).catch(() => setHealth({ ok: false }));
  }, []);
  return health;
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function formatRecentDate(value) {
  if (!value) return "";
  try {
    return new Intl.DateTimeFormat(undefined, {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit"
    }).format(new Date(value));
  } catch {
    return "";
  }
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "unknown";
  const minutes = Math.floor(seconds / 60);
  const remaining = Math.round(seconds % 60).toString().padStart(2, "0");
  return `${minutes}:${remaining}`;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function isVideoFile(file) {
  if (!file) return false;
  if (file.type?.startsWith("video/")) return true;
  return /\.(mp4|mov|m4v|webm|mkv|avi)$/i.test(file.name || "");
}

const videoViewSlots = [
  { id: "front", label: "Front", yaw: 0 },
  { id: "left45", label: "Left 45", yaw: -45 },
  { id: "right45", label: "Right 45", yaw: 45 },
  { id: "back", label: "Back", yaw: 180 }
];

const videoPresets = {
  draft: {
    label: "Video Draft",
    frames: "60",
    maxWidth: "1920",
    previewMode: "auto",
    estimate: "10-30 min"
  },
  standard: {
    label: "Video Standard",
    frames: "120",
    maxWidth: "1920",
    previewMode: "dense",
    estimate: "30-60+ min"
  },
  high: {
    label: "Video High",
    frames: "180",
    maxWidth: "2560",
    previewMode: "dense",
    estimate: "1-2+ hr"
  },
  ultra: {
    label: "Video Ultra",
    frames: "240",
    maxWidth: "source",
    previewMode: "full",
    estimate: "hours"
  }
};

function autoVideoViewSelection(frames) {
  if (!frames.length) return {};
  const last = frames.length - 1;
  return {
    front: frames[0]?.url,
    left45: frames[Math.floor(last * 0.33)]?.url,
    right45: frames[Math.floor(last * 0.66)]?.url,
    back: frames[last]?.url
  };
}

function downsampleSplatForPlayback(splat, maxPoints = 100000) {
  const points = Array.isArray(splat?.points) ? splat.points : [];
  const limit = Math.max(10000, Math.min(300000, Number(maxPoints || 100000)));
  if (points.length <= limit) return splat;
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

function measurePointBounds(points = []) {
  if (!points.length) return null;
  const bounds = {
    minX: Infinity,
    maxX: -Infinity,
    minY: Infinity,
    maxY: -Infinity,
    minZ: Infinity,
    maxZ: -Infinity
  };
  for (const point of points) {
    bounds.minX = Math.min(bounds.minX, point.x);
    bounds.maxX = Math.max(bounds.maxX, point.x);
    bounds.minY = Math.min(bounds.minY, point.y);
    bounds.maxY = Math.max(bounds.maxY, point.y);
    bounds.minZ = Math.min(bounds.minZ, point.z);
    bounds.maxZ = Math.max(bounds.maxZ, point.z);
  }
  const sizeX = Math.max(0.0001, bounds.maxX - bounds.minX);
  const sizeY = Math.max(0.0001, bounds.maxY - bounds.minY);
  const sizeZ = Math.max(0.0001, bounds.maxZ - bounds.minZ);
  return {
    ...bounds,
    centerX: (bounds.minX + bounds.maxX) / 2,
    centerY: (bounds.minY + bounds.maxY) / 2,
    centerZ: (bounds.minZ + bounds.maxZ) / 2,
    sizeX,
    sizeY,
    sizeZ,
    radius: Math.max(sizeX, sizeY, sizeZ)
  };
}

function quantile(sorted, ratio) {
  if (!sorted.length) return 0;
  const index = Math.max(0, Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * ratio)));
  return sorted[index];
}

function measureRobustPointBounds(points = []) {
  if (!points.length) return null;
  const xs = [];
  const ys = [];
  const zs = [];
  for (const point of points) {
    if (!Number.isFinite(point.x) || !Number.isFinite(point.y) || !Number.isFinite(point.z)) continue;
    xs.push(point.x);
    ys.push(point.y);
    zs.push(point.z);
  }
  if (!xs.length) return null;
  xs.sort((a, b) => a - b);
  ys.sort((a, b) => a - b);
  zs.sort((a, b) => a - b);
  const minX = quantile(xs, 0.08);
  const maxX = quantile(xs, 0.92);
  const minY = quantile(ys, 0.04);
  const maxY = quantile(ys, 0.96);
  const minZ = quantile(zs, 0.08);
  const maxZ = quantile(zs, 0.92);
  const sizeX = Math.max(0.0001, maxX - minX);
  const sizeY = Math.max(0.0001, maxY - minY);
  const sizeZ = Math.max(0.0001, maxZ - minZ);
  return {
    minX,
    maxX,
    minY,
    maxY,
    minZ,
    maxZ,
    centerX: quantile(xs, 0.5),
    centerY: quantile(ys, 0.5),
    centerZ: quantile(zs, 0.5),
    floorY: quantile(ys, 0.04),
    sizeX,
    sizeY,
    sizeZ,
    radius: Math.max(sizeX, sizeY, sizeZ)
  };
}

function normalizeSplatToReference(splat, referenceBounds) {
  const bounds = measureRobustPointBounds(splat?.points);
  if (!bounds || !referenceBounds) return splat;
  const scale = referenceBounds.radius / Math.max(0.0001, bounds.radius);
  const normalizedPoints = splat.points.map((point) => ({
    ...point,
    x: (point.x - bounds.centerX) * scale + referenceBounds.centerX,
    y: (point.y - bounds.floorY) * scale + referenceBounds.floorY,
    z: (point.z - bounds.centerZ) * scale + referenceBounds.centerZ,
    scale: point.scale ? point.scale * scale : point.scale
  }));
  return {
    ...splat,
    sequenceNormalized: true,
    points: normalizedPoints
  };
}

function ManualKeepMaskEditor({ imageUrl, maskDataUrl, onChange }) {
  const imageRef = useRef(null);
  const canvasRef = useRef(null);
  const drawingRef = useRef(false);
  const [tool, setTool] = useState("keep");
  const brushSizes = [4, 8, 16, 32];
  const [brushSize, setBrushSize] = useState(16);
  const [imageReadyKey, setImageReadyKey] = useState(0);

  useEffect(() => {
    const image = imageRef.current;
    const canvas = canvasRef.current;
    if (!image || !canvas || !image.complete || !image.naturalWidth) return;
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (!maskDataUrl) return;
    const mask = new Image();
    mask.onload = () => {
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.drawImage(mask, 0, 0, canvas.width, canvas.height);
    };
    mask.src = maskDataUrl;
  }, [imageUrl, maskDataUrl, imageReadyKey]);

  function syncCanvasSize() {
    const image = imageRef.current;
    const canvas = canvasRef.current;
    if (!image || !canvas || !image.naturalWidth) return;
    if (canvas.width !== image.naturalWidth || canvas.height !== image.naturalHeight) {
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
    }
  }

  function paint(event) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * canvas.width;
    const y = ((event.clientY - rect.top) / rect.height) * canvas.height;
    const context = canvas.getContext("2d");
    context.save();
    context.globalCompositeOperation = tool === "erase" ? "destination-out" : "source-over";
    context.fillStyle = "rgba(47, 194, 160, 0.3)";
    context.beginPath();
    context.arc(x, y, brushSize, 0, Math.PI * 2);
    context.fill();
    context.restore();
  }

  function commitMask() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    onChange(canvas.toDataURL("image/png"));
  }

  function handlePointerDown(event) {
    event.preventDefault();
    syncCanvasSize();
    drawingRef.current = true;
    event.currentTarget.setPointerCapture(event.pointerId);
    paint(event);
  }

  function handlePointerMove(event) {
    if (!drawingRef.current) return;
    paint(event);
  }

  function handlePointerUp(event) {
    if (!drawingRef.current) return;
    drawingRef.current = false;
    event.currentTarget.releasePointerCapture(event.pointerId);
    commitMask();
  }

  function clearMask() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height);
    onChange(null);
  }

  return (
    <div className="manualMask">
      <div className="manualMaskHeader">
        <strong>Manual keep mask</strong>
        <button className="iconTextButton" type="button" onClick={clearMask} disabled={!maskDataUrl}>
          <Trash2 size={15} />
          Clear
        </button>
      </div>
      <div className="manualMaskTools">
        <button className={tool === "keep" ? "toolButton selected" : "toolButton"} type="button" onClick={() => setTool("keep")}>
          <Brush size={16} />
          Keep
        </button>
        <button className={tool === "erase" ? "toolButton selected" : "toolButton"} type="button" onClick={() => setTool("erase")}>
          <Eraser size={16} />
          Erase
        </button>
        <div className="brushSteps" role="group" aria-label="Brush size">
          {brushSizes.map((size) => (
            <button className={brushSize === size ? "brushStep selected" : "brushStep"} key={size} type="button" onClick={() => setBrushSize(size)}>
              {size}
            </button>
          ))}
        </div>
      </div>
      <div className="manualMaskCanvas">
        <img
          ref={imageRef}
          src={imageUrl}
          alt="Manual mask source"
          onLoad={() => {
            syncCanvasSize();
            setImageReadyKey((key) => key + 1);
          }}
          draggable="false"
        />
        <canvas
          ref={canvasRef}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
        />
      </div>
      <p>Paint areas that must survive segmentation.</p>
    </div>
  );
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

function downloadText(filename, text, type) {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function splatToBinaryBuffer(splat) {
  const rowSize = 32;
  const buffer = new ArrayBuffer(splat.points.length * rowSize);
  const view = new DataView(buffer);
  splat.points.forEach((point, index) => {
    const offset = index * rowSize;
    const scale = Number(point.scale || 0.02);
    view.setFloat32(offset, Number(point.x || 0), true);
    view.setFloat32(offset + 4, Number(point.y || 0), true);
    view.setFloat32(offset + 8, Number(point.z || 0), true);
    view.setFloat32(offset + 12, scale, true);
    view.setFloat32(offset + 16, scale, true);
    view.setFloat32(offset + 20, scale, true);
    view.setUint8(offset + 24, Math.max(0, Math.min(255, Number(point.r || 0))));
    view.setUint8(offset + 25, Math.max(0, Math.min(255, Number(point.g || 0))));
    view.setUint8(offset + 26, Math.max(0, Math.min(255, Number(point.b || 0))));
    view.setUint8(offset + 27, Math.max(0, Math.min(255, Number(point.a ?? 255))));
    view.setUint8(offset + 28, 255);
    view.setUint8(offset + 29, 0);
    view.setUint8(offset + 30, 0);
    view.setUint8(offset + 31, 0);
  });
  return buffer;
}

function parseBinarySplat(buffer) {
  const rowSize = 32;
  if (buffer.byteLength % rowSize !== 0) {
    throw new Error("Invalid .splat file size.");
  }
  const view = new DataView(buffer);
  const points = [];
  for (let offset = 0; offset < buffer.byteLength; offset += rowSize) {
    const sx = view.getFloat32(offset + 12, true);
    const sy = view.getFloat32(offset + 16, true);
    const sz = view.getFloat32(offset + 20, true);
    points.push({
      x: Number(view.getFloat32(offset, true).toFixed(4)),
      y: Number(view.getFloat32(offset + 4, true).toFixed(4)),
      z: Number(view.getFloat32(offset + 8, true).toFixed(4)),
      r: view.getUint8(offset + 24),
      g: view.getUint8(offset + 25),
      b: view.getUint8(offset + 26),
      a: view.getUint8(offset + 27),
      scale: Number(((sx + sy + sz) / 3).toFixed(4)),
      view: "binary"
    });
  }
  return {
    format: "portrait-splat-forge.binary-splat.v1",
    createdAt: new Date().toISOString(),
    pointCount: points.length,
    sourceViews: [],
    points
  };
}

function downloadBinary(filename, buffer) {
  const blob = new Blob([buffer], { type: "application/octet-stream" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function createGaussianMaterial() {
  return new THREE.ShaderMaterial({
    transparent: false,
    depthTest: true,
    depthWrite: true,
    blending: THREE.NoBlending,
    vertexColors: true,
    uniforms: {
      pixelRatio: { value: Math.min(window.devicePixelRatio, 2) }
    },
    vertexShader: `
      attribute float splatScale;
      attribute float splatAlpha;
      varying vec3 vColor;
      varying float vAlpha;

      void main() {
        vColor = color;
        vAlpha = splatAlpha;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mvPosition;
        gl_PointSize = clamp((splatScale * 4300.0) / max(0.45, -mvPosition.z), 1.8, 46.0);
      }
    `,
    fragmentShader: `
      varying vec3 vColor;
      varying float vAlpha;

      void main() {
        vec2 centered = gl_PointCoord * 2.0 - 1.0;
        float radius2 = dot(centered, centered);
        if (radius2 > 1.0) discard;
        float alpha = exp(-radius2 * 3.2) * max(vAlpha, 0.24);
        if (alpha < 0.22) discard;
        gl_FragColor = vec4(vColor, 1.0);
      }
    `
  });
}

function SplatPreview({ imageUrl, views, splat }) {
  const hostRef = useRef(null);
  const vrButtonSlotRef = useRef(null);
  const controlsRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraPoseRef = useRef(null);
  const [autoRotating, setAutoRotating] = useState(false);
  const [stereoPreview, setStereoPreview] = useState(false);
  const angleLockOptions = [5, 10, 15, 25, "free"];
  const [angleLockIndex, setAngleLockIndex] = useState(2);
  const angleLock = angleLockOptions[angleLockIndex];

  function applyAngleLock(controls, lockValue = angleLock) {
    if (!controls) return;
    if (lockValue === "free") {
      controls.minAzimuthAngle = -Infinity;
      controls.maxAzimuthAngle = Infinity;
      controls.minPolarAngle = 0;
      controls.maxPolarAngle = Math.PI;
    } else {
      const radians = THREE.MathUtils.degToRad(lockValue);
      controls.minAzimuthAngle = -radians;
      controls.maxAzimuthAngle = radians;
      controls.minPolarAngle = Math.PI / 2 - radians;
      controls.maxPolarAngle = Math.PI / 2 + radians;
    }
    controls.enablePan = true;
    controls.update();
  }

  function cycleAngleLock() {
    setAngleLockIndex((index) => {
      const nextIndex = (index + 1) % angleLockOptions.length;
      applyAngleLock(controlsRef.current, angleLockOptions[nextIndex]);
      return nextIndex;
    });
  }

  function toggleStereoPreview() {
    setStereoPreview((enabled) => {
      const next = !enabled;
      const host = hostRef.current;
      if (host && controlsRef.current) {
        const camera = controlsRef.current.object;
        camera.aspect = next ? 8 / 9 : host.clientWidth / Math.max(1, host.clientHeight);
        camera.updateProjectionMatrix();
      }
      return next;
    });
  }

  useEffect(() => {
    if (!hostRef.current || (!imageUrl && !splat?.points?.length)) return undefined;

    const host = hostRef.current;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, host.clientWidth / host.clientHeight, 0.1, 100);
    camera.position.set(0, 0.1, 4.1);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(host.clientWidth, host.clientHeight);
    renderer.xr.enabled = true;
    renderer.xr.setReferenceSpaceType("local-floor");
    renderer.domElement.className = "splatCanvas";
    host.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    let vrButton = null;
    if (vrButtonSlotRef.current) {
      vrButtonSlotRef.current.replaceChildren();
      vrButton = VRButton.createButton(renderer);
      vrButton.classList.add("vrButton");
      vrButtonSlotRef.current.appendChild(vrButton);
    }

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.autoRotate = false;
    controls.autoRotateSpeed = 0.8;
    controls.enablePan = true;
    controls.enableZoom = true;
    controls.minDistance = 0.08;
    controls.maxDistance = 18;
    controls.zoomSpeed = 1.35;
    controls.target.set(0, -0.05, 0);
    applyAngleLock(controls, angleLock);
    controls.addEventListener("start", () => {
      controls.autoRotate = false;
      setAutoRotating(false);
    });
    controlsRef.current = controls;
    setAutoRotating(false);

    const content = new THREE.Group();
    scene.add(content);

    const group = new THREE.Group();
    content.add(group);

    const light = new THREE.PointLight(0xffffff, 2.2);
    light.position.set(2, 3, 4);
    scene.add(light);

    const targetMarker = new THREE.Mesh(
      new THREE.SphereGeometry(0.025, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0x2fc2a0, transparent: true, opacity: 0.9 })
    );
    targetMarker.visible = false;
    content.add(targetMarker);

    let disposed = false;
    let points = null;
    let hasCustomTarget = false;
    let altPressed = false;

    function saveCameraPose() {
      cameraPoseRef.current = {
        position: camera.position.toArray(),
        target: controls.target.toArray(),
        near: camera.near,
        far: camera.far,
        zoom: camera.zoom
      };
    }

    function restoreCameraPose() {
      const pose = cameraPoseRef.current;
      if (!pose) return false;
      camera.position.fromArray(pose.position);
      controls.target.fromArray(pose.target);
      camera.near = pose.near;
      camera.far = pose.far;
      camera.zoom = pose.zoom;
      camera.updateProjectionMatrix();
      controls.update();
      return true;
    }

    function framePointCloud(pointObject) {
      if (!pointObject?.geometry) return;
      pointObject.geometry.computeBoundingBox();
      const box = pointObject.geometry.boundingBox;
      if (!box || box.isEmpty()) return;
      const center = new THREE.Vector3();
      const size = new THREE.Vector3();
      box.getCenter(center);
      box.getSize(size);
      const radius = Math.max(size.x, size.y, size.z, 0.8) * 0.5;
      const fov = THREE.MathUtils.degToRad(camera.fov);
      const aspect = Math.max(0.1, camera.aspect || host.clientWidth / Math.max(1, host.clientHeight));
      const fitHeightDistance = radius / Math.tan(fov / 2);
      const fitWidthDistance = radius / (Math.tan(fov / 2) * aspect);
      const distance = Math.max(fitHeightDistance, fitWidthDistance) * 1.42;
      const target = new THREE.Vector3(center.x, center.y - size.y * 0.08, center.z);
      controls.target.copy(target);
      camera.position.set(target.x, target.y + size.y * 0.02, target.z + distance);
      camera.near = Math.max(0.01, distance / 120);
      camera.far = Math.max(100, distance * 80);
      camera.updateProjectionMatrix();
      controls.update();
    }

    const buildGeometryFromPoints = (sourcePoints) => {
      const positions = [];
      const colors = [];
      const scales = [];
      const alphas = [];
      for (const point of sourcePoints) {
        positions.push(point.x, point.y, point.z);
        colors.push(Math.max(point.r / 255, 0.08), Math.max(point.g / 255, 0.08), Math.max(point.b / 255, 0.08));
        scales.push(point.scale || 0.022);
        alphas.push(Math.max(0.16, Math.min(1, (point.a ?? 255) / 255)));
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
      geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
      geometry.setAttribute("splatScale", new THREE.Float32BufferAttribute(scales, 1));
      geometry.setAttribute("splatAlpha", new THREE.Float32BufferAttribute(alphas, 1));
      const material = createGaussianMaterial();
      points = new THREE.Points(geometry, material);
      group.add(points);
      if (!restoreCameraPose()) {
        framePointCloud(points);
      }
    };

    const buildPointCloud = (url) => {
      const image = new Image();
      image.crossOrigin = "anonymous";
      image.onload = () => {
        if (disposed) return;
        const canvas = document.createElement("canvas");
        const size = 160;
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        const ratio = Math.min(size / image.width, size / image.height);
        const drawW = image.width * ratio;
        const drawH = image.height * ratio;
        ctx.clearRect(0, 0, size, size);
        ctx.drawImage(image, (size - drawW) / 2, (size - drawH) / 2, drawW, drawH);
        const data = ctx.getImageData(0, 0, size, size).data;
        const positions = [];
        const colors = [];

        for (let y = 0; y < size; y += 2) {
          for (let x = 0; x < size; x += 2) {
            const i = (y * size + x) * 4;
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            const a = data[i + 3];
            const luma = (r + g + b) / 765;
            const nx = (x / size - 0.5) * 2.1;
            const ny = -(y / size - 0.5) * 2.35;
            const mask = a > 20 && (r + g + b > 35);
            const torsoShape = Math.abs(nx) < 0.9 - Math.max(-ny - 0.2, 0) * 0.18 && ny < 0.95 && ny > -1.12;
            const headShape = nx * nx / 0.24 + (ny - 0.55) * (ny - 0.55) / 0.28 < 1;
            if (!mask && !torsoShape && !headShape) continue;
            const z = (luma - 0.44) * 0.5 + Math.cos(nx * Math.PI) * 0.16;
            positions.push(nx, ny, z);
            colors.push(Math.max(r / 255, 0.06), Math.max(g / 255, 0.06), Math.max(b / 255, 0.06));
          }
        }

        buildGeometryFromPoints(positions.reduce((acc, _value, positionIndex) => {
          if (positionIndex % 3 !== 0) return acc;
          const colorIndex = positionIndex;
          acc.push({
            x: positions[positionIndex],
            y: positions[positionIndex + 1],
            z: positions[positionIndex + 2],
            r: Math.round(colors[colorIndex] * 255),
            g: Math.round(colors[colorIndex + 1] * 255),
            b: Math.round(colors[colorIndex + 2] * 255),
            a: 245,
            scale: 0.026
          });
          return acc;
        }, []));
      };
      image.src = url;
    };

    if (splat?.points?.length) {
      buildGeometryFromPoints(splat.points);
    } else {
      buildPointCloud(imageUrl);
    }

    const grid = new THREE.GridHelper(3.6, 24, 0x3f8f98, 0x20343b);
    grid.position.y = -1.25;
    grid.rotation.x = Math.PI / 2;
    grid.material.opacity = 0.28;
    grid.material.transparent = true;
    content.add(grid);

    const enterVr = () => {
      controls.enabled = false;
      controls.autoRotate = false;
      setAutoRotating(false);
      content.position.set(0, 0.05, -2.2);
      content.scale.setScalar(0.9);
      targetMarker.visible = false;
    };

    const exitVr = () => {
      controls.enabled = true;
      content.position.set(0, 0, 0);
      content.scale.setScalar(1);
      if (!hasCustomTarget) framePointCloud(points);
    };

    renderer.xr.addEventListener("sessionstart", enterVr);
    renderer.xr.addEventListener("sessionend", exitVr);

    const raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 0.075;
    const pointer = new THREE.Vector2();

    function setOrbitTargetFromPointer(event) {
      if (!event.altKey || !points) return;
      event.preventDefault();
      event.stopPropagation();
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObject(points, false)[0];
      if (!hit) return;
      controls.target.copy(hit.point);
      targetMarker.position.copy(hit.point);
      hasCustomTarget = true;
      altPressed = event.altKey;
      targetMarker.visible = altPressed;
      controls.update();
    }

    function handleKeyDown(event) {
      if (event.key !== "Alt") return;
      altPressed = true;
      targetMarker.visible = hasCustomTarget;
    }

    function handleKeyUp(event) {
      if (event.key !== "Alt") return;
      altPressed = false;
      targetMarker.visible = false;
    }

    function handleBlur() {
      altPressed = false;
      targetMarker.visible = false;
    }

    renderer.domElement.addEventListener("pointerdown", setOrbitTargetFromPointer, { capture: true });
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    window.addEventListener("blur", handleBlur);

    function resize() {
      if (!host.clientWidth || !host.clientHeight) return;
      camera.aspect = stereoPreview ? 8 / 9 : host.clientWidth / host.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(host.clientWidth, host.clientHeight);
      if (!hasCustomTarget) framePointCloud(points);
    }

    function animate() {
      controls.update();
      if (stereoPreview && !renderer.xr.isPresenting) {
        const width = host.clientWidth;
        const height = host.clientHeight;
        const sbsWidth = Math.min(width, Math.floor(height * (16 / 9)));
        const sbsHeight = Math.min(height, Math.floor(sbsWidth * (9 / 16)));
        const viewportX = Math.floor((width - sbsWidth) / 2);
        const viewportY = Math.floor((height - sbsHeight) / 2);
        const halfWidth = Math.max(1, Math.floor(sbsWidth / 2));
        const target = controls.target.clone();
        const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion).normalize();
        const eyeDistance = Math.max(0.035, camera.position.distanceTo(target) * 0.018);
        const leftCamera = camera.clone();
        const rightCamera = camera.clone();
        leftCamera.position.copy(camera.position).addScaledVector(right, -eyeDistance * 0.5);
        rightCamera.position.copy(camera.position).addScaledVector(right, eyeDistance * 0.5);
        leftCamera.aspect = 8 / 9;
        rightCamera.aspect = 8 / 9;
        leftCamera.updateProjectionMatrix();
        rightCamera.updateProjectionMatrix();
        leftCamera.lookAt(target);
        rightCamera.lookAt(target);
        leftCamera.updateMatrixWorld();
        rightCamera.updateMatrixWorld();
        renderer.clear();
        renderer.setScissorTest(true);
        renderer.setViewport(viewportX, viewportY, halfWidth, sbsHeight);
        renderer.setScissor(viewportX, viewportY, halfWidth, sbsHeight);
        renderer.render(scene, leftCamera);
        renderer.setViewport(viewportX + halfWidth, viewportY, sbsWidth - halfWidth, sbsHeight);
        renderer.setScissor(viewportX + halfWidth, viewportY, sbsWidth - halfWidth, sbsHeight);
        renderer.render(scene, rightCamera);
        renderer.setScissorTest(false);
      } else {
        renderer.setViewport(0, 0, host.clientWidth, host.clientHeight);
        renderer.setScissorTest(false);
        renderer.render(scene, camera);
      }
    }

    const observer = new ResizeObserver(resize);
    observer.observe(host);
    renderer.setAnimationLoop(animate);

    return () => {
      disposed = true;
      saveCameraPose();
      renderer.setAnimationLoop(null);
      observer.disconnect();
      renderer.xr.removeEventListener("sessionstart", enterVr);
      renderer.xr.removeEventListener("sessionend", exitVr);
      scene.traverse((object) => {
        if (object.geometry) object.geometry.dispose();
        if (object.material) object.material.dispose();
      });
      controls.dispose();
      controlsRef.current = null;
      rendererRef.current = null;
      renderer.domElement.removeEventListener("pointerdown", setOrbitTargetFromPointer, { capture: true });
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      window.removeEventListener("blur", handleBlur);
      if (renderer.domElement.parentNode === host) {
        host.removeChild(renderer.domElement);
      }
      if (vrButton?.parentNode) {
        vrButton.parentNode.removeChild(vrButton);
      }
      renderer.dispose();
    };
  }, [imageUrl, views, splat, stereoPreview]);

  function resumeAutoRotate() {
    if (!controlsRef.current) return;
    controlsRef.current.autoRotate = true;
    setAutoRotating(true);
  }

  function captureScreenshot() {
    const renderer = rendererRef.current;
    if (!renderer) return;
    const filename = `portrait-splat-preview-${new Date().toISOString().replace(/[:.]/g, "-")}.png`;
    const link = document.createElement("a");
    link.href = renderer.domElement.toDataURL("image/png");
    link.download = filename;
    link.click();
  }

  return (
    <div className={stereoPreview ? "previewShell stereoMode" : "previewShell"}>
      <div className="previewHeader">
        <span><Box size={16} /> Gaussian sprite preview</span>
        <div className="previewActions">
          <span>
            {splat?.originalVertexCount
              ? `${splat.originalVertexCount.toLocaleString()} source splats, ${splat.points.length.toLocaleString()} preview`
              : splat?.pointCount
                ? `${splat.pointCount.toLocaleString()} splat points`
                : views?.length
                  ? `${views.length} views staged`
                  : "waiting for image"}
          </span>
          <button className="angleLockButton" disabled={!imageUrl} onClick={cycleAngleLock} type="button" title="Cycle view angle lock">
            <Gauge size={15} />
            {angleLock === "free" ? "Free" : `${angleLock}°`}
          </button>
          <button
            className={stereoPreview ? "angleLockButton selected" : "angleLockButton"}
            disabled={!imageUrl}
            onClick={toggleStereoPreview}
            type="button"
            title="Toggle side-by-side stereo preview"
          >
            <Eye size={15} />
            SBS
          </button>
          <span className="vrButtonSlot" ref={vrButtonSlotRef} />
          <button className="iconButton" disabled={!imageUrl || autoRotating} onClick={resumeAutoRotate} type="button" title="Resume auto rotate">
            <RefreshCw size={15} />
          </button>
          <button className="iconButton" disabled={!imageUrl} onClick={captureScreenshot} type="button" title="Save screenshot">
            <CameraIcon size={15} />
          </button>
        </div>
      </div>
      <div className="viewerHint">Drag to rotate, wheel to zoom, right-drag to pan. Alt-click a splat to set the rotation center.</div>
      <div className="threeHost" ref={hostRef}>
        {imageUrl && !splat?.points?.length && (
          <div className="preSplatOverlay">
            <img src={imageUrl} alt="Front view preview" />
            <p>Views are staged. Build a splat to generate the Gaussian preview.</p>
          </div>
        )}
        {!imageUrl && (
          <div className="emptyPreview">
            <Camera size={36} />
            <p>Drop in a bust photo to build the working splat preview.</p>
          </div>
        )}
      </div>
    </div>
  );
}

function App() {
  const health = useHealth();
  const [workMode, setWorkMode] = useState("still");
  const [upload, setUpload] = useState(null);
  const [videoUpload, setVideoUpload] = useState(null);
  const [videoAnalysis, setVideoAnalysis] = useState(null);
  const [videoPreset, setVideoPreset] = useState("draft");
  const [extractFrameCount, setExtractFrameCount] = useState("60");
  const [videoMaxWidth, setVideoMaxWidth] = useState("1920");
  const [videoTrimStart, setVideoTrimStart] = useState("0");
  const [videoTrimEnd, setVideoTrimEnd] = useState("5");
  const [extractingFrames, setExtractingFrames] = useState(false);
  const [extractedFrames, setExtractedFrames] = useState([]);
  const [selectedVideoFrame, setSelectedVideoFrame] = useState(null);
  const [videoViewSelection, setVideoViewSelection] = useState({});
  const [sharpSequenceFrameCount, setSharpSequenceFrameCount] = useState("5");
  const [sharpSequence, setSharpSequence] = useState(null);
  const [sequenceFrameIndex, setSequenceFrameIndex] = useState(0);
  const [sequencePlaying, setSequencePlaying] = useState(false);
  const [sequenceFps, setSequenceFps] = useState("2");
  const [sequencePreviewPoints, setSequencePreviewPoints] = useState("100000");
  const [sequenceNormalize, setSequenceNormalize] = useState(true);
  const sequencePlayingRef = useRef(false);
  const sequenceLoadTokenRef = useRef(0);
  const sequenceReferenceBoundsRef = useRef(null);
  const [uploading, setUploading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [buildingSplat, setBuildingSplat] = useState(false);
  const [generationMode, setGenerationMode] = useState("mock");
  const [splatBackend, setSplatBackend] = useState("lightweight");
  const [splatQuality, setSplatQuality] = useState("balanced");
  const [previewMode, setPreviewMode] = useState("auto");
  const [sceneMode, setSceneMode] = useState("subject");
  const [maskMode, setMaskMode] = useState("person");
  const [maskRefine, setMaskRefine] = useState("clean");
  const [manualKeepMask, setManualKeepMask] = useState(null);
  const [preprocessSize, setPreprocessSize] = useState("1536");
  const [views, setViews] = useState([]);
  const [splat, setSplat] = useState(null);
  const [savedSplat, setSavedSplat] = useState(null);
  const [recentOutputs, setRecentOutputs] = useState([]);
  const [loadingRecent, setLoadingRecent] = useState(false);
  const [buildProgress, setBuildProgress] = useState(null);
  const [activeStep, setActiveStep] = useState(0);
  const [error, setError] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [isVideoDragging, setIsVideoDragging] = useState(false);
  const [draggingViewId, setDraggingViewId] = useState(null);
  const [isSplatDragging, setIsSplatDragging] = useState(false);
  const sequenceFrames = sharpSequence?.frames || [];

  useEffect(() => {
    if (sceneMode === "full") {
      setManualKeepMask(null);
    }
  }, [sceneMode]);

  useEffect(() => {
    refreshRecentOutputs();
  }, []);

  useEffect(() => {
    sequencePlayingRef.current = sequencePlaying;
    if (!sequencePlaying) {
      sequenceLoadTokenRef.current += 1;
    }
  }, [sequencePlaying]);

  useEffect(() => {
    if (!sequencePlaying || !sequenceFrames.length) return undefined;
    const interval = window.setInterval(() => {
      if (!sequencePlayingRef.current) return;
      setSequenceFrameIndex((index) => (index + 1) % sequenceFrames.length);
    }, Math.max(120, 1000 / Number(sequenceFps || 2)));
    return () => window.clearInterval(interval);
  }, [sequencePlaying, sequenceFrames.length, sequenceFps]);

  useEffect(() => {
    if (!sharpSequence || !sequenceFrames.length) return;
    const safeIndex = Math.min(sequenceFrameIndex, sequenceFrames.length - 1);
    if (safeIndex !== sequenceFrameIndex) {
      setSequenceFrameIndex(safeIndex);
    }
    loadSharpSequenceFrame(safeIndex);
  }, [sharpSequence, sequenceFrameIndex]);

  const sourceImage = views.find((view) => view.id === "front")?.url || upload?.url;
  const completeCount = useMemo(() => views.filter((view) => ["source", "generated", "mock", "fallback", "manual"].includes(view.status)).length, [views]);
  const failedViews = useMemo(() => views.filter((view) => view.status === "failed"), [views]);
  const selectedBackend = useMemo(
    () => (health?.splatBackends || []).find((backend) => backend.id === splatBackend),
    [health, splatBackend]
  );
  const requiredViewIds = selectedBackend?.requiredViews || ["front", "left45", "right45", "back"];
  const availableViews = useMemo(() => {
    const merged = upload?.url
      ? [{ id: "front", label: "Front", yaw: 0, url: upload.url, status: "source" }, ...views]
      : views;
    return requiredViewIds
      .map((id) => merged.find((view) => view.id === id && view.url))
      .filter(Boolean);
  }, [requiredViewIds, upload, views]);
  const requiredCompleteCount = availableViews.length;
  const canBuildSplat = requiredCompleteCount >= requiredViewIds.length;
  const previewNeedsRebuild = Boolean(splat && splat.previewMode && splat.previewMode !== previewMode);
  const canGenerateViews = Boolean(upload?.id && upload.source !== "video-frame");
  const selectedFrame = useMemo(
    () => extractedFrames.find((frame) => frame.url === selectedVideoFrame) || extractedFrames[0] || null,
    [extractedFrames, selectedVideoFrame]
  );
  const videoSelectedViews = useMemo(() => {
    const byUrl = new Map(extractedFrames.map((frame) => [frame.url, frame]));
    return videoViewSlots
      .map((slot) => {
        const frame = byUrl.get(videoViewSelection[slot.id]);
        return frame ? { ...slot, url: frame.url, status: slot.id === "front" ? "source" : "manual", name: frame.name, timestamp: frame.timestamp } : null;
      })
      .filter(Boolean);
  }, [extractedFrames, videoViewSelection]);
  const canUseVideoViews = videoSelectedViews.length === videoViewSlots.length;
  const selectedVideoPreset = videoPresets[videoPreset] || videoPresets.draft;
  const estimatedVideoBuild = selectedVideoPreset.estimate;

  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem("portrait-splat-forge-state") || "null");
      if (!saved) return;
      setUpload(saved.upload || null);
      setWorkMode(saved.workMode || "still");
      setVideoUpload(saved.videoUpload || null);
      setVideoAnalysis(saved.videoAnalysis || null);
      setVideoPreset(saved.videoPreset || "draft");
      setExtractFrameCount(saved.extractFrameCount || "60");
      setVideoMaxWidth(saved.videoMaxWidth || "1920");
      setVideoTrimStart(saved.videoTrimStart || "0");
      setVideoTrimEnd(saved.videoTrimEnd || "5");
      setExtractedFrames(Array.isArray(saved.extractedFrames) ? saved.extractedFrames : []);
      setSelectedVideoFrame(saved.selectedVideoFrame || null);
      setVideoViewSelection(saved.videoViewSelection || {});
      setSharpSequenceFrameCount(saved.sharpSequenceFrameCount || "5");
      setSharpSequence(saved.sharpSequence || null);
      setSequenceFrameIndex(saved.sequenceFrameIndex || 0);
      setSequenceFps(saved.sequenceFps || "2");
      setSequencePreviewPoints(saved.sequencePreviewPoints || "100000");
      setSequenceNormalize(saved.sequenceNormalize !== false);
      setViews(Array.isArray(saved.views) ? saved.views : []);
      setSplat(null);
      setSavedSplat(saved.savedSplat || null);
      setGenerationMode(saved.generationMode || "mock");
      setSplatBackend(saved.splatBackend || "lightweight");
      setSplatQuality(saved.splatQuality || "balanced");
      setPreviewMode(saved.previewMode || "auto");
      setSceneMode(saved.sceneMode || "subject");
      setMaskMode(saved.maskMode || "person");
      setMaskRefine(saved.maskRefine || "clean");
      setManualKeepMask(saved.manualKeepMask || null);
      setPreprocessSize(saved.preprocessSize || "1536");
      setActiveStep(saved.views?.length ? 3 : 0);
    } catch {
      localStorage.removeItem("portrait-splat-forge-state");
    }
  }, []);

  useEffect(() => {
    const state = {
      upload,
      workMode,
      videoUpload,
      videoAnalysis,
      videoPreset,
      extractFrameCount,
      videoMaxWidth,
      videoTrimStart,
      videoTrimEnd,
      extractedFrames,
      selectedVideoFrame,
      videoViewSelection,
      sharpSequenceFrameCount,
      sharpSequence,
      sequenceFrameIndex,
      sequenceFps,
      sequencePreviewPoints,
      sequenceNormalize,
      views,
      savedSplat,
      generationMode,
      splatBackend,
      splatQuality,
      previewMode,
      sceneMode,
      maskMode,
      maskRefine,
      manualKeepMask,
      preprocessSize
    };
    try {
      localStorage.setItem("portrait-splat-forge-state", JSON.stringify(state));
    } catch {
      localStorage.removeItem("portrait-splat-forge-state");
      setError("Could not persist workspace state in this browser.");
    }
  }, [upload, workMode, videoUpload, videoAnalysis, videoPreset, extractFrameCount, videoMaxWidth, videoTrimStart, videoTrimEnd, extractedFrames, selectedVideoFrame, videoViewSelection, sharpSequenceFrameCount, sharpSequence, sequenceFrameIndex, sequenceFps, sequencePreviewPoints, sequenceNormalize, views, savedSplat, generationMode, splatBackend, splatQuality, previewMode, sceneMode, maskMode, maskRefine, manualKeepMask, preprocessSize]);

  function resetWorkspace() {
    localStorage.removeItem("portrait-splat-forge-state");
    setWorkMode("still");
    setUpload(null);
    setVideoUpload(null);
    setVideoAnalysis(null);
    setVideoPreset("draft");
    setExtractFrameCount("60");
    setVideoMaxWidth("1920");
    setVideoTrimStart("0");
    setVideoTrimEnd("5");
    setExtractedFrames([]);
    setSelectedVideoFrame(null);
    setVideoViewSelection({});
    setSharpSequenceFrameCount("5");
    setSharpSequence(null);
    setSequenceFrameIndex(0);
    setSequencePlaying(false);
    setSequenceFps("2");
    setSequencePreviewPoints("100000");
    setSequenceNormalize(true);
    sequenceReferenceBoundsRef.current = null;
    setViews([]);
    setSplat(null);
    setSavedSplat(null);
    setBuildProgress(null);
    setManualKeepMask(null);
    setError("");
    setActiveStep(0);
  }

  async function handleFile(file) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("Please drop an image file.");
      return;
    }
    setError("");
    setUploading(true);
    const body = new FormData();
    body.append("image", file);
    try {
      const response = await fetch("/api/upload", { method: "POST", body });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Upload failed");
      setUpload(payload);
      setWorkMode("still");
      setViews([{ id: "front", label: "Front", yaw: 0, url: payload.url, status: "source" }]);
      setSplat(null);
      setSavedSplat(null);
      setBuildProgress(null);
      setManualKeepMask(null);
      setActiveStep(1);
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  }

  function handleDrop(event) {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files?.[0];
    handleFile(file);
  }

  async function analyzeVideo(uploadedVideo = videoUpload) {
    if (!uploadedVideo?.id) return null;
    const response = await fetch("/api/analyze-video", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ videoId: uploadedVideo.id })
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Video analysis failed.");
    setVideoAnalysis(payload.analysis);
    return payload.analysis;
  }

  async function handleVideoFile(file) {
    if (!file) return;
    if (!isVideoFile(file)) {
      setError("Please drop a video file (.mp4, .mov, .webm, .mkv, or .avi).");
      return;
    }
    setError("");
    setUploading(true);
    const body = new FormData();
    body.append("video", file);
    try {
      const response = await fetch("/api/upload-video", { method: "POST", body });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Video upload failed.");
      setVideoUpload(payload);
      setWorkMode("video");
      setVideoAnalysis(null);
      setExtractedFrames([]);
      setSelectedVideoFrame(null);
      setVideoViewSelection({});
      await analyzeVideo(payload);
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  }

  function handleVideoDrop(event) {
    event.preventDefault();
    setIsVideoDragging(false);
    const file = event.dataTransfer.files?.[0];
    handleVideoFile(file);
  }

  async function extractVideoFrames() {
    if (!videoUpload?.id) return;
    setExtractingFrames(true);
    setError("");
    try {
      const response = await fetch("/api/extract-video-frames", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          videoId: videoUpload.id,
          frameCount: Number(extractFrameCount),
          maxWidth: videoMaxWidth,
          trimStart: Number(videoTrimStart || 0),
          trimEnd: videoTrimEnd === "" ? null : Number(videoTrimEnd)
        })
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Frame extraction failed.");
      setExtractedFrames(payload.frames || []);
      setSelectedVideoFrame(payload.frames?.[0]?.url || null);
      setVideoViewSelection(autoVideoViewSelection(payload.frames || []));
      setSharpSequence(null);
      setVideoAnalysis(payload.analysis || videoAnalysis);
    } catch (err) {
      setError(err.message);
    } finally {
      setExtractingFrames(false);
    }
  }

  function applyVideoPreset(presetId) {
    const preset = videoPresets[presetId] || videoPresets.draft;
    setVideoPreset(presetId);
    setExtractFrameCount(preset.frames);
    setVideoMaxWidth(preset.maxWidth);
    setPreviewMode(preset.previewMode);
  }

  function useExtractedFramesAsViews() {
    if (!canUseVideoViews) return;
    const nextViews = videoSelectedViews;
    setUpload({
      id: null,
      name: `${videoUpload?.name || "video"} frame`,
      source: "video-frame",
      url: nextViews.find((view) => view.id === "front")?.url || nextViews[0].url
    });
    setViews(nextViews);
    setSplat(null);
    setSavedSplat(null);
    setBuildProgress(null);
    setManualKeepMask(null);
    setActiveStep(3);
  }

  function assignVideoFrameToView(viewId, frame = selectedFrame) {
    if (!frame) return;
    setVideoViewSelection((selection) => ({ ...selection, [viewId]: frame.url }));
  }

  async function draftBuildFromVideo() {
    if (!canUseVideoViews) return;
    useExtractedFramesAsViews();
    setBuildingSplat(true);
    setError("");
    try {
      await buildSplatFromViews(videoSelectedViews);
    } catch (err) {
      setError(err.message);
    } finally {
      setBuildingSplat(false);
    }
  }

  async function buildSharpSequenceFromFrames() {
    if (!extractedFrames.length) return;
    setBuildingSplat(true);
    setError("");
    setActiveStep(3);
    setBuildProgress({
      status: "queued",
      phase: "queued",
      backendLabel: "Apple SHARP sequence",
      message: `Queued ${sharpSequenceFrameCount} SHARP frames`,
      percent: 0
    });
    try {
      const response = await fetch("/api/build-sharp-sequence-job", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          frames: extractedFrames,
          frameLimit: Number(sharpSequenceFrameCount),
          quality: splatQuality,
          previewMode,
          sceneMode,
          maskMode,
          maskRefine,
          preprocessSize: Number(preprocessSize),
          sequencePreviewPoints: Number(sequencePreviewPoints)
        })
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not start SHARP sequence.");
      setBuildProgress(payload.progress);

      while (true) {
        await delay(2000);
        const statusResponse = await fetch(`/api/build-sharp-sequence-job/${payload.jobId}`);
        const statusPayload = await statusResponse.json();
        if (!statusResponse.ok) throw new Error(statusPayload.error || "Could not read SHARP sequence progress.");
        setBuildProgress(statusPayload.progress);
        if (statusPayload.status === "completed") {
          setSharpSequence(statusPayload.result.sequence);
          sequenceReferenceBoundsRef.current = null;
          setSequenceFrameIndex(0);
          setSequencePlaying(false);
          if (statusPayload.result.splat) setSplat(statusPayload.result.splat);
          if (statusPayload.result.saved) setSavedSplat(statusPayload.result.saved);
          setActiveStep(4);
          refreshRecentOutputs();
          return statusPayload.result;
        }
        if (statusPayload.status === "failed") {
          throw new Error(statusPayload.error || statusPayload.progress?.message || "SHARP sequence failed.");
        }
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setBuildingSplat(false);
    }
  }

  async function uploadViewFile(viewId, file) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("Please drop an image file.");
      return;
    }

    const spec = [
      { id: "front", label: "Front", yaw: 0 },
      { id: "left45", label: "Left 45", yaw: -45 },
      { id: "right45", label: "Right 45", yaw: 45 },
      { id: "back", label: "Back", yaw: 180 }
    ].find((item) => item.id === viewId);

    setError("");
    const body = new FormData();
    body.append("image", file);
    try {
      const response = await fetch("/api/upload", { method: "POST", body });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Upload failed");
      const nextView = { ...spec, url: payload.url, status: viewId === "front" ? "source" : "manual", name: payload.name };
      setViews((currentViews) => {
        const merged = [...currentViews.filter((view) => view.id !== viewId), nextView];
        return ["front", "left45", "right45", "back"].map((id) => merged.find((view) => view.id === id)).filter(Boolean);
      });
      if (viewId === "front") {
        setUpload(payload);
        setManualKeepMask(null);
      }
      setSplat(null);
      setSavedSplat(null);
      setBuildProgress(null);
      setActiveStep(3);
    } catch (err) {
      setError(err.message);
    }
  }

  async function buildSplatFromViews(nextViews) {
    setBuildingSplat(true);
    setActiveStep(3);
    setBuildProgress({
      status: "queued",
      phase: "queued",
      message: "Queued GS3D build",
      percent: 0
    });
    const response = await fetch("/api/build-splat-job", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ views: nextViews, backend: splatBackend, quality: splatQuality, previewMode, sceneMode, maskMode, maskRefine, manualKeepMask, preprocessSize: Number(preprocessSize) })
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not start splat build.");
    setBuildProgress(payload.progress);

    while (true) {
      await delay(1500);
      const statusResponse = await fetch(`/api/build-splat-job/${payload.jobId}`);
      const statusPayload = await statusResponse.json();
      if (!statusResponse.ok) throw new Error(statusPayload.error || "Could not read build progress.");
      setBuildProgress(statusPayload.progress);
      if (statusPayload.status === "completed") {
        setSplat(statusPayload.result.splat);
        setSavedSplat(statusPayload.result.saved);
        setActiveStep(4);
        refreshRecentOutputs();
        return statusPayload.result;
      }
      if (statusPayload.status === "failed") {
        throw new Error(statusPayload.error || statusPayload.progress?.message || "GS3D build failed.");
      }
    }
  }

  async function generateViews() {
    if (!canGenerateViews) {
      setError("Video frames are already staged as views. Build directly or upload a still image to generate synthetic views.");
      return;
    }
    setGenerating(true);
    setError("");
    setActiveStep(2);
    try {
      const response = await fetch("/api/generate-views", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ imageId: upload.id, mode: generationMode })
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Generation failed");
      setViews(payload.views);
      setSplat(null);
      setSavedSplat(null);
      setBuildProgress(null);
      const failures = payload.views?.filter((view) => view.status === "failed" || view.status === "fallback") || [];
      if (failures.length) {
        setError(failures.map((view) => `${view.label}: ${view.status === "fallback" ? "OpenAI failed, using local fallback" : view.error}`).join(" / "));
      }
      await buildSplatFromViews(payload.views);
    } catch (err) {
      setError(err.message);
    } finally {
      setGenerating(false);
      setBuildingSplat(false);
    }
  }

  async function generateSplat() {
    if (!canBuildSplat) return;
    setBuildingSplat(true);
    setError("");
    setActiveStep(3);
    try {
      await buildSplatFromViews(availableViews);
    } catch (err) {
      setError(err.message);
    } finally {
      setBuildingSplat(false);
    }
  }

  function exportSplatJson() {
    if (!splat) return;
    downloadText(`portrait-splat-${Date.now()}.splat.json`, JSON.stringify(splat, null, 2), "application/json");
  }

  function exportPly() {
    if (!splat) return;
    downloadText(`portrait-splat-${Date.now()}.ply`, splatToPly(splat), "application/octet-stream");
  }

  function exportSplatBinary() {
    if (!splat) return;
    downloadBinary(`portrait-splat-${Date.now()}.splat`, splatToBinaryBuffer(splat));
  }

  async function loadSplatFile(file) {
    if (!file) return;
    try {
      let loaded = null;
      if (file.name.endsWith(".splat")) {
        loaded = parseBinarySplat(await file.arrayBuffer());
      } else if (file.name.endsWith(".json")) {
        loaded = JSON.parse(await file.text());
      } else {
        throw new Error("Please load a .splat or .splat.json file.");
      }
      if (!loaded?.points?.length) {
        throw new Error("The selected file does not contain splat points.");
      }
      setSplat({
        ...loaded,
        pointCount: loaded.pointCount || loaded.points.length
      });
      setSavedSplat(null);
      setError("");
      setActiveStep(4);
    } catch (err) {
      setError(err.message);
    }
  }

  async function loadSharpSequenceFrame(index) {
    const frame = sharpSequence?.frames?.[index];
    const jsonUrl = frame?.preview || frame?.saved?.files?.json;
    if (!jsonUrl) return;
    const loadToken = sequenceLoadTokenRef.current + 1;
    sequenceLoadTokenRef.current = loadToken;
    try {
      const response = await fetch(jsonUrl);
      let loaded = await response.json();
      if (sequenceLoadTokenRef.current !== loadToken) return;
      if (!response.ok) throw new Error(loaded.error || "Could not load SHARP sequence frame.");
      if (!loaded?.points?.length) throw new Error("The SHARP sequence frame does not contain preview points.");
      if (!frame?.preview) {
        loaded = downsampleSplatForPlayback(loaded, sequencePreviewPoints);
      }
      if (index === 0 || !sequenceReferenceBoundsRef.current) {
        sequenceReferenceBoundsRef.current = measureRobustPointBounds(loaded.points);
      }
      if (sequenceNormalize && sequenceReferenceBoundsRef.current) {
        loaded = normalizeSplatToReference(loaded, sequenceReferenceBoundsRef.current);
      }
      if (sequenceLoadTokenRef.current !== loadToken) return;
      setSplat({
        ...loaded,
        pointCount: loaded.pointCount || loaded.points.length,
        sequenceFrameIndex: index
      });
      setSavedSplat(frame.saved);
      setActiveStep(4);
    } catch (err) {
      setSequencePlaying(false);
      setError(err.message);
    }
  }

  async function refreshRecentOutputs() {
    try {
      const response = await fetch("/api/recent-splats?limit=8");
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not load recent outputs.");
      setRecentOutputs(Array.isArray(payload.outputs) ? payload.outputs : []);
    } catch {
      setRecentOutputs([]);
    }
  }

  async function loadRecentOutput(output) {
    if (!output?.files?.json && !output?.files?.manifest) return;
    setLoadingRecent(true);
    setError("");
    try {
      if (output.type === "sharp-sequence" || output.format === "portrait-splat-forge.sharp-sequence.v1") {
        const manifestResponse = await fetch(output.files.manifest);
        const sequence = await manifestResponse.json();
        if (!manifestResponse.ok) throw new Error(sequence.error || "Could not load SHARP sequence.");
        if (!Array.isArray(sequence.frames) || !sequence.frames.length) throw new Error("The SHARP sequence has no frames.");
        setSharpSequence({
          ...sequence,
          files: output.files
        });
        sequenceReferenceBoundsRef.current = null;
        setSequenceFrameIndex(0);
        setSequencePlaying(false);
        setSavedSplat(sequence.frames[0]?.saved || null);
        setActiveStep(4);
        return;
      }
      const response = await fetch(output.files.json);
      const loaded = await response.json();
      if (!response.ok) throw new Error(loaded.error || "Could not load saved splat.");
      if (!loaded?.points?.length) throw new Error("The saved output does not contain preview points.");
      setSharpSequence(null);
      setSequencePlaying(false);
      setSplat({
        ...loaded,
        pointCount: loaded.pointCount || loaded.points.length
      });
      setSavedSplat(output);
      setActiveStep(4);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingRecent(false);
    }
  }

  function loadLatestOutput() {
    loadRecentOutput(recentOutputs[0]);
  }

  return (
    <main className="app">
      <aside className="panel controls">
        <div className="brand">
          <div className="brandMark"><Sparkles size={18} /></div>
          <div>
            <h1>Portrait Splat Forge</h1>
            <p>four-view portrait to Gaussian Splat</p>
          </div>
        </div>

        {buildProgress && (
          <div className="buildProgress buildProgressTop">
            <div className="buildProgressHeader">
              <strong>{buildProgress.backendLabel || selectedBackend?.label || "GS3D build"}</strong>
              <span>{Math.round(buildProgress.percent || 0)}%</span>
            </div>
            <div className="progressTrack">
              <div style={{ width: `${Math.max(0, Math.min(100, buildProgress.percent || 0))}%` }} />
            </div>
            <p>{buildProgress.message || buildProgress.phase || "Working..."}</p>
            {buildProgress.totalIterations && (
              <small>
                {Number(buildProgress.iteration || 0).toLocaleString()} / {Number(buildProgress.totalIterations).toLocaleString()} iterations
              </small>
            )}
          </div>
        )}

        <div className="workflowSwitch" role="group" aria-label="Workflow mode">
          <button className={workMode === "still" ? "selected" : ""} type="button" onClick={() => setWorkMode("still")}>
            <ImagePlus size={16} />
            <span>Still image</span>
          </button>
          <button className={workMode === "video" ? "selected" : ""} type="button" onClick={() => setWorkMode("video")}>
            <Film size={16} />
            <span>Video</span>
          </button>
        </div>

        {workMode === "still" ? (
        <label
          className={isDragging ? "dropZone dragging" : "dropZone"}
          onDragEnter={(event) => {
            event.preventDefault();
            setIsDragging(true);
          }}
          onDragOver={(event) => {
            event.preventDefault();
            event.dataTransfer.dropEffect = "copy";
          }}
          onDragLeave={(event) => {
            event.preventDefault();
            if (event.currentTarget.contains(event.relatedTarget)) return;
            setIsDragging(false);
          }}
          onDrop={handleDrop}
        >
          <input data-testid="main-upload" type="file" accept="image/*" onChange={(event) => handleFile(event.target.files?.[0])} />
          {upload ? (
            <img src={upload.url} alt="Uploaded portrait" />
          ) : (
            <span>
              <ImagePlus size={28} />
              <strong>Upload portrait</strong>
              <small>drop an image here or click to browse</small>
            </span>
          )}
        </label>
        ) : (
          <button className="collapsedWorkflow" type="button" onClick={() => setWorkMode("still")}>
            <span><ImagePlus size={16} /> Still image</span>
            <strong>{upload?.name || upload?.id || "No image loaded"}</strong>
          </button>
        )}

        {workMode === "video" ? (
        <div
          className={isVideoDragging ? "videoPanel videoPanelDragging" : "videoPanel"}
          onDragEnter={(event) => {
            event.preventDefault();
            setIsVideoDragging(true);
          }}
          onDragOver={(event) => {
            event.preventDefault();
            event.dataTransfer.dropEffect = "copy";
          }}
          onDragLeave={(event) => {
            event.preventDefault();
            if (event.currentTarget.contains(event.relatedTarget)) return;
            setIsVideoDragging(false);
          }}
          onDrop={handleVideoDrop}
        >
          <div className="videoPanelHeader">
            <strong><Film size={15} /> Video frames</strong>
            {videoUpload?.size && <span>{formatBytes(videoUpload.size)}</span>}
          </div>
          {videoUpload?.summaryUrl && (
            <div className="videoSummary">
              <img src={videoUpload.summaryUrl} alt={`${videoUpload.name || "Uploaded video"} summary`} />
              <span>{videoUpload.name || "Uploaded video"}</span>
            </div>
          )}
          <label className="videoUploadButton">
            <input type="file" accept="video/*" onChange={(event) => handleVideoFile(event.target.files?.[0])} />
            <Film size={17} />
            Upload video
          </label>
          {videoUpload && (
            <>
              <div className="videoMeta">
                <strong>{videoUpload.name || "Uploaded video"}</strong>
                {videoAnalysis?.available ? (
                  <span>
                    {videoAnalysis.width}x{videoAnalysis.height} / {formatDuration(videoAnalysis.duration)}
                  </span>
                ) : (
                  <span>{videoAnalysis?.warning || "Ready to analyze"}</span>
                )}
              </div>
              <div className="videoActions">
                <select className="selectControl compact" value={videoPreset} onChange={(event) => applyVideoPreset(event.target.value)}>
                  {Object.entries(videoPresets).map(([id, preset]) => (
                    <option key={id} value={id}>{preset.label}</option>
                  ))}
                </select>
                <select className="selectControl compact" value={extractFrameCount} onChange={(event) => setExtractFrameCount(event.target.value)}>
                  <option value="5">Extract frames: 5</option>
                  <option value="10">Extract frames: 10</option>
                  <option value="15">Extract frames: 15</option>
                  <option value="30">Extract frames: 30</option>
                  <option value="60">Extract frames: 60</option>
                  <option value="120">Extract frames: 120</option>
                  <option value="180">Extract frames: 180</option>
                  <option value="240">Extract frames: 240</option>
                </select>
                <select className="selectControl compact" value={videoMaxWidth} onChange={(event) => setVideoMaxWidth(event.target.value)}>
                  <option value="1920">Frame width: 1080p/1920</option>
                  <option value="2560">Frame width: 1440p/2560</option>
                  <option value="source">Frame width: Source</option>
                </select>
                <div className="trimControls" aria-label="Video trim range">
                  <label>
                    <span>Trim start</span>
                    <input type="number" min="0" step="0.1" value={videoTrimStart} onChange={(event) => setVideoTrimStart(event.target.value)} />
                  </label>
                  <span className="trimDash">-</span>
                  <label>
                    <span>Trim end</span>
                    <input type="number" min="0" step="0.1" value={videoTrimEnd} onChange={(event) => setVideoTrimEnd(event.target.value)} />
                  </label>
                </div>
                <p className="videoEstimate">Quest Link / PC GPU: {estimatedVideoBuild} generation estimate after extraction.</p>
                <select className="selectControl compact" value={sharpSequenceFrameCount} onChange={(event) => setSharpSequenceFrameCount(event.target.value)}>
                  <option value="5">SHARP sequence frames: 5</option>
                  <option value="10">SHARP sequence frames: 10</option>
                  <option value="15">SHARP sequence frames: 15</option>
                  <option value="30">SHARP sequence frames: 30</option>
                </select>
                <select className="selectControl compact" value={sequencePreviewPoints} onChange={(event) => setSequencePreviewPoints(event.target.value)}>
                  <option value="50000">Sequence preview: 50k</option>
                  <option value="100000">Sequence preview: 100k</option>
                  <option value="200000">Sequence preview: 200k</option>
                </select>
                <button className="secondary" type="button" onClick={() => analyzeVideo()} disabled={uploading || extractingFrames}>
                  <CameraIcon size={16} />
                  Analyze video
                </button>
                <button className="secondary" type="button" onClick={extractVideoFrames} disabled={uploading || extractingFrames}>
                  {extractingFrames ? <Loader2 className="spin" size={16} /> : <RefreshCw size={16} />}
                  Extract frames
                </button>
              </div>
              {extractedFrames.length > 0 && (
                <>
                  <div className="videoFramePicker">
                    <div className="selectedFramePreview">
                      {selectedFrame && <img src={selectedFrame.url} alt={`Selected video frame ${selectedFrame.index + 1}`} />}
                      <span>{selectedFrame ? `Frame ${selectedFrame.index + 1} / ${formatDuration(selectedFrame.timestamp)}` : "Select a frame"}</span>
                    </div>
                    <div className="videoSlotGrid">
                      {videoViewSlots.map((slot) => {
                        const assigned = extractedFrames.find((frame) => frame.url === videoViewSelection[slot.id]);
                        return (
                          <button type="button" key={slot.id} onClick={() => assignVideoFrameToView(slot.id)} disabled={!selectedFrame}>
                            <strong>{slot.label}</strong>
                            <span>{assigned ? `${assigned.index + 1} / ${formatDuration(assigned.timestamp)}` : "unset"}</span>
                          </button>
                        );
                      })}
                    </div>
                    <div className="frameGrid">
                      {extractedFrames.map((frame) => (
                        <button
                          className={selectedFrame?.url === frame.url ? "selected" : ""}
                          key={frame.url}
                          type="button"
                          onClick={() => setSelectedVideoFrame(frame.url)}
                        >
                          <img src={frame.url} alt={`Extracted frame ${frame.index + 1}`} />
                          <span>{frame.index + 1}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                  <button className="primary videoDraftButton" type="button" onClick={buildSharpSequenceFromFrames} disabled={!extractedFrames.length || buildingSplat}>
                    {buildingSplat ? <Loader2 className="spin" size={16} /> : <Sparkles size={16} />}
                    Build Apple SHARP sequence
                  </button>
                  {sharpSequence && (
                    <p className="videoEstimate">
                      SHARP sequence saved: {sharpSequence.frameCount} frames
                    </p>
                  )}
                  <button className="secondary" type="button" onClick={useExtractedFramesAsViews} disabled={!canUseVideoViews || buildingSplat}>
                    <Box size={16} />
                    Stage selected 4 views
                  </button>
                  <button className="secondary videoDraftButton" type="button" onClick={draftBuildFromVideo} disabled={!canUseVideoViews || buildingSplat}>
                    {buildingSplat ? <Loader2 className="spin" size={16} /> : <Box size={16} />}
                    4-view draft build
                  </button>
                </>
              )}
            </>
          )}
        </div>
        ) : (
          <button className="collapsedWorkflow" type="button" onClick={() => setWorkMode("video")}>
            <span><Film size={16} /> Video</span>
            <strong>{videoUpload?.name || "No video loaded"}</strong>
          </button>
        )}

        {workMode === "video" && sharpSequence && (
          <div className="sequencePlayer">
            <div className="sequencePlayerHeader">
              <strong>SHARP sequence</strong>
              <span>{sequenceFrameIndex + 1}/{sequenceFrames.length}</span>
            </div>
            <input
              type="range"
              min="0"
              max={Math.max(0, sequenceFrames.length - 1)}
              step="1"
              value={sequenceFrameIndex}
              onChange={(event) => {
                sequenceLoadTokenRef.current += 1;
                setSequencePlaying(false);
                setSequenceFrameIndex(Number(event.target.value));
              }}
            />
            <div className="sequenceActions">
              <button
                type="button"
                onClick={() => {
                  if (sequencePlaying) sequenceLoadTokenRef.current += 1;
                  setSequencePlaying((playing) => !playing);
                }}
                disabled={!sequenceFrames.length}
              >
                {sequencePlaying ? <Pause size={15} /> : <Play size={15} />}
                {sequencePlaying ? "Pause" : "Play"}
              </button>
              <select value={sequenceFps} onChange={(event) => setSequenceFps(event.target.value)}>
                <option value="1">1 fps</option>
                <option value="2">2 fps</option>
                <option value="4">4 fps</option>
                <option value="8">8 fps</option>
              </select>
            </div>
            <label className="sequenceToggle">
              <input
                type="checkbox"
                checked={sequenceNormalize}
                onChange={(event) => {
                  sequenceReferenceBoundsRef.current = null;
                  sequenceLoadTokenRef.current += 1;
                  setSequenceNormalize(event.target.checked);
                }}
              />
              Normalize frame alignment
            </label>
            <p>
              {sequenceFrames[sequenceFrameIndex]?.timestamp !== undefined
                ? `Source time ${formatDuration(sequenceFrames[sequenceFrameIndex].timestamp)}`
                : "Saved SHARP frame sequence"}
              {splat?.sequencePreviewPointCount ? ` / preview ${splat.sequencePreviewPointCount.toLocaleString()} points` : ""}
            </p>
            {sharpSequence.files?.manifest && (
              <a href={sharpSequence.files.manifest} target="_blank" rel="noreferrer">Open sequence manifest</a>
            )}
          </div>
        )}

        {workMode === "still" && (
          <>
            {upload?.url && sceneMode === "subject" && (
              <ManualKeepMaskEditor imageUrl={upload.url} maskDataUrl={manualKeepMask} onChange={setManualKeepMask} />
            )}

            <button className="primary" disabled={!canBuildSplat || buildingSplat} onClick={generateSplat}>
              {buildingSplat ? <Loader2 className="spin" size={17} /> : <Box size={17} />}
              {requiredViewIds.length === 1 ? "Build 3D GS from front" : "Build GS3D from views"}
            </button>
          </>
        )}

        <div className="recentOutputs">
          <div className="recentHeader">
            <strong>Recent outputs</strong>
            <button type="button" onClick={refreshRecentOutputs} disabled={loadingRecent}>
              <RefreshCw size={14} />
            </button>
          </div>
          <button className="secondary recentLatest" disabled={!recentOutputs.length || loadingRecent} onClick={loadLatestOutput}>
            {loadingRecent ? <Loader2 className="spin" size={17} /> : <Download size={17} />}
            Load latest
          </button>
          {recentOutputs.length > 0 && (
            <div className="recentList">
              {recentOutputs.slice(0, 5).map((output) => (
                <button type="button" key={output.id} onClick={() => loadRecentOutput(output)} disabled={loadingRecent}>
                  <span>{formatRecentDate(output.createdAt)}</span>
                  <strong>
                    {(output.type === "sharp-sequence" || output.format === "portrait-splat-forge.sharp-sequence.v1")
                      ? `${output.frameCount || output.frames?.length || 0} frames`
                      : Number(output.pointCount || output.previewPointCount || 0).toLocaleString()}
                  </strong>
                </button>
              ))}
            </div>
          )}
        </div>

        {workMode === "still" && (
          <>
        <div className="modeSwitch" role="group" aria-label="Generation mode">
          <button className={generationMode === "mock" ? "selected" : ""} onClick={() => setGenerationMode("mock")} type="button">
            Mock
          </button>
          <button className={generationMode === "openai" ? "selected" : ""} onClick={() => setGenerationMode("openai")} type="button" disabled={!health?.openai}>
            OpenAI
          </button>
        </div>

        <select className="selectControl" value={splatBackend} onChange={(event) => setSplatBackend(event.target.value)}>
          {(health?.splatBackends || [{ id: "lightweight", label: "Lightweight PLY", available: true }]).map((backend) => (
            <option key={backend.id} value={backend.id} disabled={!backend.available}>
              {backend.label}{backend.available ? "" : " (not configured)"}
            </option>
          ))}
        </select>

        <select className="selectControl" value={splatQuality} onChange={(event) => setSplatQuality(event.target.value)}>
          <option value="draft">Draft</option>
          <option value="balanced">Balanced</option>
          <option value="high">High density</option>
        </select>

        <select className="selectControl" value={previewMode} onChange={(event) => setPreviewMode(event.target.value)}>
          <option value="auto">Preview: Auto</option>
          <option value="dense">Preview: Dense</option>
          <option value="full">Preview: Full</option>
        </select>

        <select className="selectControl" value={sceneMode} onChange={(event) => setSceneMode(event.target.value)}>
          <option value="subject">Scene: Subject only</option>
          <option value="full">Scene: Full image</option>
        </select>

        {sceneMode === "subject" && (
          <>
            <select className="selectControl" value={maskMode} onChange={(event) => setMaskMode(event.target.value)}>
              <option value="person">Mask: Person</option>
              <option value="objectProps">Mask: Object+Props</option>
            </select>

            <select className="selectControl" value={maskRefine} onChange={(event) => setMaskRefine(event.target.value)}>
              <option value="clean">Mask refine: Clean</option>
              <option value="keepProps">Mask refine: Keep props stronger</option>
            </select>
          </>
        )}

        <select className="selectControl" value={preprocessSize} onChange={(event) => setPreprocessSize(event.target.value)}>
          <option value="1024">Preprocess: 1024 px</option>
          <option value="1536">Preprocess: 1536 px</option>
        </select>
        {previewNeedsRebuild && (
          <p className="inlineHint">Preview density changes apply on the next build.</p>
        )}

        <button className="secondary" disabled={!canGenerateViews || uploading || generating || buildingSplat} onClick={generateViews}>
          {generating ? <Loader2 className="spin" size={18} /> : <RefreshCw size={18} />}
          Generate missing views
        </button>
          </>
        )}

        <button className="secondary" disabled={!splat} onClick={exportPly}>
          <Download size={17} />
          Export .ply
        </button>

        <label
          className={isSplatDragging ? "loadSplatButton dragging" : "loadSplatButton"}
          onDragEnter={(event) => {
            event.preventDefault();
            setIsSplatDragging(true);
          }}
          onDragOver={(event) => {
            event.preventDefault();
            event.dataTransfer.dropEffect = "copy";
          }}
          onDragLeave={(event) => {
            event.preventDefault();
            if (event.currentTarget.contains(event.relatedTarget)) return;
            setIsSplatDragging(false);
          }}
          onDrop={(event) => {
            event.preventDefault();
            setIsSplatDragging(false);
            loadSplatFile(event.dataTransfer.files?.[0]);
          }}
        >
          <input type="file" accept=".splat,.json,application/json" onChange={(event) => loadSplatFile(event.target.files?.[0])} />
          <Download size={17} />
          Load .splat / .json
        </label>

        <button className="secondary danger" onClick={resetWorkspace}>
          <RotateCcw size={17} />
          Reset images
        </button>

        {error && <p className="error">{error}</p>}
        {savedSplat && <p className="success">Saved {savedSplat.pointCount.toLocaleString()} points to outputs/splats.</p>}
      </aside>

      <section className="stage">
        <div className="topbar">
          {steps.map((step, index) => (
            <div className={index <= activeStep ? "step active" : "step"} key={step}>
              <span>{index + 1}</span>
              {step}
              {index < steps.length - 1 && <ChevronRight size={14} />}
            </div>
          ))}
        </div>

        <SplatPreview imageUrl={sourceImage} views={views} splat={splat} />
      </section>

      <aside className="panel views">
        <div className="panelTitle">
          <h2>Generated Views</h2>
          <span>{completeCount}/4</span>
        </div>

        <div className="settings resultSettings">
          <div>
            <span>Image model</span>
            <strong>{health?.imageModel || "checking..."}</strong>
          </div>
          <div>
            <span>Responses model</span>
            <strong>{health?.responsesModel || "checking..."}</strong>
          </div>
          <div>
            <span>Generation mode</span>
            <strong>{generationMode === "openai" ? "OpenAI fill" : "Mock fill"}</strong>
          </div>
          <div>
            <span>Target views</span>
            <strong>front, left, right, back</strong>
          </div>
          <div>
            <span>Splat backend</span>
            <strong>{selectedBackend?.label || splatBackend}</strong>
          </div>
          <div>
            <span>Required views</span>
            <strong>{requiredCompleteCount}/{requiredViewIds.length}</strong>
          </div>
          <div>
            <span>Quality</span>
            <strong>{splatQuality}</strong>
          </div>
          <div>
            <span>Preview</span>
            <strong>{previewMode}</strong>
          </div>
          <div>
            <span>Scene mode</span>
            <strong>{sceneMode === "full" ? "Full image" : "Subject only"}</strong>
          </div>
          {sceneMode === "subject" && <div>
            <span>Mask mode</span>
            <strong>{maskMode === "objectProps" ? "Object+Props" : "Person"}</strong>
          </div>}
          {sceneMode === "subject" && <div>
            <span>Mask refine</span>
            <strong>{maskRefine === "keepProps" ? "Keep props stronger" : "Clean"}</strong>
          </div>}
          {sceneMode === "subject" && <div>
            <span>Manual keep</span>
            <strong>{manualKeepMask ? "Painted" : "None"}</strong>
          </div>}
          <div>
            <span>Preprocess</span>
            <strong>{preprocessSize}px</strong>
          </div>
        </div>

        <div className="viewGrid">
          {["front", "left45", "right45", "back"].map((id) => {
            const view = views.find((item) => item.id === id);
            return (
              <article
                className={draggingViewId === id ? "viewCard viewCardDragging" : "viewCard"}
                key={id}
                onDragEnter={(event) => {
                  event.preventDefault();
                  setDraggingViewId(id);
                }}
                onDragOver={(event) => {
                  event.preventDefault();
                  event.dataTransfer.dropEffect = "copy";
                }}
                onDragLeave={(event) => {
                  event.preventDefault();
                  if (event.currentTarget.contains(event.relatedTarget)) return;
                  setDraggingViewId(null);
                }}
                onDrop={(event) => {
                  event.preventDefault();
                  setDraggingViewId(null);
                  uploadViewFile(id, event.dataTransfer.files?.[0]);
                }}
              >
                <input className="viewFileInput" data-testid={`view-upload-${id}`} type="file" accept="image/*" onChange={(event) => uploadViewFile(id, event.target.files?.[0])} />
                {view?.url ? <img src={view.url} alt={view.label} /> : <div className="viewEmpty"><Camera size={22} /></div>}
                <div>
                  <strong>{view?.label || id}</strong>
                  <span className={view?.status || "pending"}>
                    {(view?.status === "generated" || view?.status === "fallback" || view?.status === "manual") && <BadgeCheck size={14} />}
                    {view?.status || "pending"}
                  </span>
                  <small className="viewHint">drop or click to replace</small>
                  {view?.error && <small className="viewError">{view.error}</small>}
                </div>
              </article>
            );
          })}
        </div>

        {failedViews.length > 0 && (
          <div className="failureBox">
            <strong>View generation failed</strong>
            <p>{failedViews.map((view) => `${view.label}: ${view.error}`).join(" / ")}</p>
          </div>
        )}

        <div className="notes">
          <h3>GS3D output</h3>
          <p>
            {splat
              ? `${(splat.originalVertexCount || splat.pointCount || splat.points.length).toLocaleString()} splats saved. The viewer uses ${splat.points.length.toLocaleString()} preview points when production PLY output is larger.`
              : "Load or generate all four views, then build with the lightweight preview or a configured four-view 3DGS backend."}
          </p>
        </div>

        <button className="secondary wide" disabled={!splat} onClick={exportSplatJson}>
          <RotateCcw size={17} />
          Export .splat.json
        </button>
        <button className="secondary wide" disabled={!splat} onClick={exportSplatBinary}>
          <Download size={17} />
          Export .splat
        </button>
        {savedSplat && (
          <div className="assetLinks">
            <a href={savedSplat.files.ply} target="_blank" rel="noreferrer">Open saved PLY</a>
            <a href={savedSplat.files.json} target="_blank" rel="noreferrer">Open JSON</a>
            <a href={savedSplat.files.splat} target="_blank" rel="noreferrer">Open .splat</a>
          </div>
        )}
      </aside>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
