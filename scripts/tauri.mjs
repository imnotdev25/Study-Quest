// `pnpm tauri ...` wrapper. On macOS, `tauri build` runs create-dmg, whose optional AppleScript step (it only
// arranges the icons in the DMG window) needs Automation permission for Finder and otherwise fails with
// "Not authorized to send Apple events to Finder (-1743)". CI=true makes Tauri's bundler skip that step, so local
// builds behave like CI. Set STUDY_QUEST_DMG_LAYOUT=1 to keep the Finder layout step.
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const bin = path.join(here, "..", "node_modules", ".bin", process.platform === "win32" ? "tauri.cmd" : "tauri");
const args = process.argv.slice(2);
const env = { ...process.env };
if (process.platform === "darwin" && args[0] === "build" && !env.STUDY_QUEST_DMG_LAYOUT) env.CI ??= "true";

const child = spawn(bin, args, { stdio: "inherit", env, shell: process.platform === "win32" });
child.on("exit", (code, signal) => process.exit(signal ? 1 : code ?? 1));
