# AGENTS.md

Study Quest: a Tauri 2 desktop app (React + TypeScript frontend, Rust backend) for learning RL, LLMs, System Design and AI Safety.
Runs on macOS, Windows and Linux.

## Commands
- Install: `pnpm install` and `uv sync` (./.venv with pinned Python packages; `--extra torch` for PyTorch)
- Dev app: `pnpm tauri dev` (first time: `pnpm tauri icon src-tauri/icons/source.png`)
- Frontend type-check + build: `pnpm build`. Run it before saying a task is done.
- Rust check: `cd src-tauri && cargo check`
- Regenerate content (needs pandoc, tectonic, `../.venv`, `../.venv-post`): `pnpm content`
- Guides only (AI Safety, RL Environments; Markdown in `guides/`): `uv run python scripts/build_guides.py --check` (runs every Python block; run after build_content.py, it merges into catalog.json)
- Exercises only: `uv run python scripts/build_exercises.py` (fails unless every solution passes and every starter fails)
- Explainer videos: `cd videos && npm install && npm run record [scene]` renders `videos/web/scenes/<id>.js` to `public/videos/<id>.mp4` + poster and updates `public/videos/videos.json` (needs ffmpeg + Chrome). Scenes are pure functions of time; keep the light theme from `videos/web/lib.js`.
- Check a notebook runs: `marimo export html notebooks/interactive/<name>.py -o /tmp/x.html`

## Layout
- `src/views/` pages; `src/components/` shared UI; `src/lib/store.ts` all persisted state (localStorage)
- `src/lib/platform.ts` OS detection and per-OS wording/keys; `src/lib/theme.ts` light/dark + accent
- `src/nudges.json` every notification message (compiled into the Rust binary with `include_str!`)
- `src-tauri/src/platform.rs` every OS-specific call (open, reveal, trash, speech, home dir)
- `src-tauri/src/marimo.rs` the local notebook server; `notebooks.rs` `~/.study-quest/notebooks`; `runner.rs` runs user Python; `pyenv.rs` picks/creates the interpreter (custom path → ~/.study-quest/.venv → ./.venv → system)
- `scripts/` content builders. `public/content/`, `public/figs/`, `notebooks/` are GENERATED but committed (CI can't regenerate them).

## Conventions
- Never call `window.confirm` / `prompt` or browser downloads: Tauri's web views ignore them. Use `confirmDialog` / `promptDialog`
  (`src/components/Dialog.tsx`) and the `save_download` command.
- OS-specific behaviour goes through `platform.rs` / `platform.ts`, not inline `cfg!` or `/usr/bin/...` paths.
  macOS-only features (Calendar via osascript, vibrancy) must degrade cleanly elsewhere.
- Pass user data to external programs as arguments or env vars, never interpolated into a shell or script string.
- Icons: use `src/components/Icon.tsx` (inline SVG), not new icon packages. Pin exact dependency versions (Python pins live in both `requirements.txt` and `pyproject.toml`; keep them equal).
- XP for repeatable actions goes through `awardOnce(key, …)` so editing text can't farm points.
- Topic ids are `book:index` (e.g. `rl:12`); paths sent to the frontend use forward slashes.

## Gotchas
- Changing `src/nudges.json` needs a Rust rebuild (it's `include_str!`'d).
- `tsconfig` has `noUnusedLocals`: remove imports you stop using.
- marimo runs in its own process group / tree-kill; always stop it via `marimo::shutdown`, never by killing the child alone.
- Settings added to `Settings` in `store.ts` need a default in `fresh()`; `load()` merges defaults into saved state.

## Boundaries
- Nudges/roasts tease study habits only: never appearance, identity, health or anything genuinely hurtful.
- Don't commit secrets, `.venv*`, `node_modules`, `src-tauri/target`.
