// Copy Excalidraw's fonts into public/ so the drawing board works offline (the app's CSP blocks its CDN).
// Runs before `pnpm dev` and `pnpm build`. public/fonts/ is git-ignored.
import { cpSync, existsSync } from "node:fs";

const src = "node_modules/@excalidraw/excalidraw/dist/prod/fonts";
if (!existsSync(src)) {
  console.warn(`[excalidraw] ${src} not found; run pnpm install`);
  process.exit(0);
}
cpSync(src, "public/fonts", { recursive: true });
