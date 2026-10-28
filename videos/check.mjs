// Load scenes headless, render a few frames each, report page errors. Usage: node check.mjs [scene ...]
import { chromium } from "playwright";
import fs from "node:fs";
import { serve } from "./serve.mjs";
const server = await serve(5183);
const b = await chromium.launch({ channel: "chrome", headless: true, args: ["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"] });
const p = await b.newPage({ viewport: { width: 1280, height: 720 } });
let errs = [];
p.on("pageerror", (e) => errs.push(e.message));
p.on("console", (m) => m.type() === "error" && !m.text().includes("404") && errs.push(m.text()));
await p.goto("http://127.0.0.1:5183/?record");
await p.waitForFunction(() => window.viewerReady === true, null, { timeout: 30000 }).catch(() => {});
const ids = process.argv.slice(2).length ? process.argv.slice(2) : await p.evaluate(() => window.SCENES);
fs.mkdirSync("/tmp/sqcheck", { recursive: true });
for (const id of ids) {
  errs = [];
  await p.goto(`http://127.0.0.1:5183/?record&scene=${id}`);
  try { await p.waitForFunction(() => window.viewerReady === true, null, { timeout: 20000 }); } catch { errs.push("never ready"); }
  const d = await p.evaluate(() => window.META?.duration ?? 0);
  const times = [2, ...Array.from({ length: 8 }, (_, k) => 4 + (k + 0.5) * (d - 4) / 8)];
  for (const t of times) {
    try { await p.evaluate((t) => window.renderAt(t), t); } catch (e) { errs.push(`t=${t.toFixed(1)}: ${e.message}`); }
  }
  await p.locator("#stage").screenshot({ path: `/tmp/sqcheck/${id}.png` });
  console.log(id.padEnd(18), String(d).padEnd(4), errs.length ? errs.slice(0, 3) : "ok");
}
await b.close(); server.close();
