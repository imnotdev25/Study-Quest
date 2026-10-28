// Render every scene in headless Chrome, frame by frame, into ../public/videos/<id>.mp4 + <id>.jpg (poster),
// and write ../public/videos/videos.json (the manifest the app reads).
// Usage: node record.mjs [scene ...] [--fps 30]
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { serve } from "./serve.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.join(here, "../public/videos");
const args = process.argv.slice(2);
const fi = args.indexOf("--fps");
const fps = fi >= 0 ? Number(args[fi + 1]) : 30;
const only = args.filter((a, i) => !a.startsWith("--") && args[i - 1] !== "--fps");
fs.mkdirSync(OUT, { recursive: true });

const port = 5181;
const server = await serve(port);
const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"] });
const page = await browser.newPage({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1 });
page.on("pageerror", (e) => console.error("page error:", e.message));

async function open(id) {
  await page.goto(`http://127.0.0.1:${port}/?record&scene=${id}`);
  await page.waitForFunction(() => window.viewerReady === true, null, { timeout: 60000 });
  return page.evaluate(() => window.META);
}

await open("rl_loop");
const scenes = only.length ? only : await page.evaluate(() => window.SCENES);
const manifestPath = path.join(OUT, "videos.json");
const manifest = fs.existsSync(manifestPath) ? JSON.parse(fs.readFileSync(manifestPath, "utf8")) : [];
const stage = page.locator("#stage");

for (const id of scenes) {
  const meta = await open(id);
  const frames = Math.round(meta.duration * fps), file = path.join(OUT, `${id}.mp4`);
  const ff = spawn("ffmpeg", ["-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", String(fps), "-c:v", "mjpeg", "-i", "-",
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "27", "-preset", "slow", "-tune", "animation", "-movflags", "+faststart", "-an", file],
    { stdio: ["pipe", "inherit", "inherit"] });
  const t0 = Date.now();
  for (let i = 0; i < frames; i++) {
    await page.evaluate((t) => window.renderAt(t), i / fps);
    const buf = await stage.screenshot({ type: "jpeg", quality: 90 });
    if (!ff.stdin.write(buf)) await new Promise((ok) => ff.stdin.once("drain", ok));
  }
  ff.stdin.end();
  await new Promise((ok) => ff.on("close", ok));
  await page.evaluate((t) => window.renderAt(t), meta.poster ?? meta.duration / 2);
  await stage.screenshot({ path: path.join(OUT, `${id}.jpg`), type: "jpeg", quality: 82 });
  const size = fs.statSync(file).size;
  const entry = { id, title: meta.title, desc: meta.desc, path: meta.path, keywords: meta.keywords, duration: meta.duration,
                  src: `videos/${id}.mp4`, poster: `videos/${id}.jpg` };
  const k = manifest.findIndex((m) => m.id === id);
  if (k >= 0) manifest[k] = entry; else manifest.push(entry);
  console.log(`${id}: ${frames} frames, ${(size / 1e6).toFixed(1)} MB, ${((Date.now() - t0) / 1000).toFixed(0)}s`);
}
const order = await page.evaluate(() => window.SCENES);
manifest.sort((a, b) => order.indexOf(a.id) - order.indexOf(b.id));
fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 1) + "\n");
await browser.close();
server.close();
