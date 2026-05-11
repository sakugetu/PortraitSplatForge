import { chromium } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

const root = process.cwd();
const samplePath = path.join(root, "samples", "mock-portrait.svg");
const outputDir = path.join(root, "outputs", "qa");
fs.mkdirSync(outputDir, { recursive: true });

async function runViewport({ name, width, height }) {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width, height } });
  await page.goto("http://127.0.0.1:5173/", { waitUntil: "networkidle" });
  await page.getByRole("button", { name: "Reset images" }).click().catch(() => {});
  await page.getByTestId("main-upload").setInputFiles(samplePath);
  await page.getByRole("button", { name: "Generate missing views" }).click();
  await page.waitForSelector("text=splat points", { timeout: 20000 });
  await page.reload({ waitUntil: "networkidle" });
  await page.waitForSelector("text=4/4", { timeout: 10000 });
  await page.getByRole("button", { name: "Build GS3D from views" }).click();
  await page.waitForSelector("text=splat points", { timeout: 20000 });
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(outputDir, `${name}.png`), fullPage: true });

  const pixelStats = await page.locator("section canvas.splatCanvas").evaluate((canvas) => {
    const context = canvas.getContext("webgl2") || canvas.getContext("webgl");
    const width = canvas.width;
    const height = canvas.height;
    const pixels = new Uint8Array(width * height * 4);
    context.readPixels(0, 0, width, height, context.RGBA, context.UNSIGNED_BYTE, pixels);
    let nonBlank = 0;
    for (let i = 0; i < pixels.length; i += 4) {
      if (pixels[i] > 8 || pixels[i + 1] > 8 || pixels[i + 2] > 8) nonBlank += 1;
    }
    return { width, height, nonBlank };
  });

  await browser.close();
  if (pixelStats.nonBlank < 500) {
    throw new Error(`${name} canvas appears blank: ${JSON.stringify(pixelStats)}`);
  }
  return pixelStats;
}

async function runLoadJsonCheck() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 820 } });
  await page.goto("http://127.0.0.1:5173/", { waitUntil: "networkidle" });
  await page.getByTestId("main-upload").setInputFiles(samplePath);
  await page.getByRole("button", { name: "Generate missing views" }).click();
  await page.waitForSelector("text=splat points", { timeout: 20000 });
  const jsonUrl = await page.locator(".assetLinks a", { hasText: "Open JSON" }).getAttribute("href");
  const splatUrl = await page.locator(".assetLinks a", { hasText: "Open .splat" }).getAttribute("href");
  const jsonPath = path.join(root, jsonUrl.replace("/outputs/", "outputs/"));
  const splatPath = path.join(root, splatUrl.replace("/outputs/", "outputs/"));
  await page.reload({ waitUntil: "networkidle" });
  await page.locator(".loadSplatButton input").setInputFiles(jsonPath);
  await page.waitForSelector("text=splat points", { timeout: 20000 });
  await page.screenshot({ path: path.join(outputDir, "load-json.png"), fullPage: true });
  await page.reload({ waitUntil: "networkidle" });
  await page.locator(".loadSplatButton input").setInputFiles(splatPath);
  await page.waitForSelector("text=splat points", { timeout: 20000 });
  await page.screenshot({ path: path.join(outputDir, "load-binary-splat.png"), fullPage: true });
  await browser.close();
  return { loadedJson: path.relative(root, jsonPath), loadedSplat: path.relative(root, splatPath) };
}

const results = [];
for (const viewport of [
  { name: "desktop", width: 1440, height: 900 },
  { name: "mobile", width: 390, height: 844 }
]) {
  results.push({ viewport: viewport.name, ...(await runViewport(viewport)) });
}

console.log(JSON.stringify({ ok: true, results, loadJson: await runLoadJsonCheck() }, null, 2));
