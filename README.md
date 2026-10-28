# 🦦 Study Quest

[![Build](https://github.com/imnotdev25/Study-Quest/actions/workflows/build.yml/badge.svg)](https://github.com/imnotdev25/Study-Quest/actions/workflows/build.yml)
[![Release](https://img.shields.io/github/v/release/imnotdev25/Study-Quest?include_prereleases)](https://github.com/imnotdev25/Study-Quest/releases)
![Platforms](https://img.shields.io/badge/platforms-macOS%20%7C%20Windows%20%7C%20Linux-7c5cf6)

**Learn Reinforcement Learning, LLMs, System Design and AI Safety by doing, not just reading.**

Study Quest is a desktop app that turns five long-form books and two hands-on guides into lessons you can read, run, sketch and quiz yourself on. It has live [marimo](https://marimo.io) notebooks, a coding playground with tested exercises, [Excalidraw](https://excalidraw.com) boards for system design, and Byte the otter 🦦, who nudges (and roasts) you until you open a lesson.

Built with Tauri 2, React and Rust. Runs on macOS, Windows and Linux. Everything stays on your computer.

---

## ✨ What's inside

### Four learning paths, one at a time
| Path | Books | Topics |
|---|---|---|
| 🎯 **RL** | Reinforcement Learning · RL Environments & Verifiers | 201 |
| 🧠 **LLMs** | LLM Fundamentals · Pre-Training to Post-Training | 145 |
| 🏗️ **System Design** | System Design: Zero to Expert · DSA and ML Interviews | 54 |
| 🛡️ **AI Safety** | Alignment · Interpretability · Scalable oversight · Truthfulness · Adversarial robustness · Safety evaluation | 30 |

Pick a path in the sidebar and only that path is shown, with a progress ring, chapter list and a **Continue** button. Lessons keep their math (KaTeX), diagrams (compiled from TikZ) and code.

**RL Environments & Verifiers** is the practical side of RL for LLMs: building environments, designing verifiers that resist reward hacking, sandboxes for agentic rollouts, and training-data pipelines that survive a real run.

### Learn by doing
- **▶ Runnable lessons:** every Python block has **Run / Edit**, notebook-style: earlier cells run first, output and plots appear inline.
- **📓 Notebooks in the page:** reactive marimo notebooks open inside lessons. 23 interactive ones (bandits, PPO clipping, attention heatmaps, KV-cache calculator, reward hacking, calibration, linear probes, judge bias, prompt injection, verifier and sandbox labs…) plus every book script. Create your own; they live in `~/.study-quest/notebooks`.
- **🧪 Coding playground:** 60 tested exercises across DSA, ML math, transformers, RL, inference, system design, agents, RL environments and AI safety, plus a free scratchpad.
- **🎬 3D explainer videos:** worked-example animations for the major topics: bandits, MDPs and returns, value iteration, TD vs Monte Carlo, policy gradients, PPO, RLHF, GRPO, embeddings, attention, the transformer block, next-token prediction and training on a **Videos** page and inside the lessons they explain.
- **✏️ Design it yourself:** every System Design and DSA lesson has an Excalidraw board, saved per lesson. All your boards are on the **Diagrams** page; export as `.excalidraw`.

### Make it stick
- **🔮 Warm-up:** rate what you know and guess before reading.
- **🧠 Quiz me:** derivations reveal one step at a time; key ideas hide until you've recalled them.
- **✍️ Explain it back:** write it in your own words; the app checks which key terms you used and compares with your guess.
- **🔗 Connect it**, **🌫️ What's unclear?** (shown on your dashboard until resolved) and **📈 confidence** before vs after.
- **🎙 Voice typing** on every box (falls back to 🌐 twice on macOS, Win + H on Windows).

### Keep coming back
- Streaks, XP, levels, badges and an activity card (this week vs last, daily-goal chart, heatmap).
- **Desktop nudges** with a troll level from 😇 Nice to 🔥 Unhinged. At the top end Byte brings up your crush's LinkedIn and your screen-time report.
- **Plan my week:** books study blocks on weekday evenings. macOS adds them to Calendar (✕ removes them again); Windows and Linux open an `.ics` in your calendar app.
- **🎨 Themes:** System / Light / Dark and eight accent colours (or pick your own).

---

## 📦 Install

Download from **[Releases](https://github.com/imnotdev25/Study-Quest/releases)**: a versioned release, or **nightly** for the latest commit:

| OS | File | First launch (builds aren't signed yet) |
|---|---|---|
| macOS (Apple Silicon / Intel) | `.dmg` | See *macOS first launch* below |
| Windows 10 / 11 | `.exe` installer or `.msi` | SmartScreen: **More info → Run anyway** |
| Linux x86_64 / ARM64 | `.AppImage` or `.deb` | `chmod +x Study*.AppImage && ./Study*.AppImage`, or `sudo apt install ./study-quest*.deb` |
| Linux ARMv7 (e.g. Raspberry Pi 32-bit) | `.deb` | `sudo apt install ./study-quest*armhf.deb` |

**Python for exercises and notebooks.** In **Settings → Python**, pick one:

- **Set up with uv (recommended):** one click creates the app's own environment in `~/.study-quest/.venv` with the pinned
  packages from `requirements.txt` (marimo, numpy, matplotlib; optionally PyTorch). Needs [uv](https://docs.astral.sh/uv/):
  `brew install uv`, `winget install astral-sh.uv`, or `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- **Use my own Python:** paste any interpreter path (a conda env, a project `.venv`, pyenv…) and press **Test**. It shows the
  version and which packages it has. Install what's missing with `python -m pip install -r requirements.txt`.

The app uses, in order: your custom path, then the uv environment, then the system Python.

<details>
<summary>Platform differences</summary>

| | macOS | Windows | Linux |
|---|---|---|---|
| Calendar | Synced to a "Study Quest" calendar | `.ics` export | `.ics` export |
| Sidebar | Translucent (vibrancy) | Solid | Solid |
| Tray / menu bar | Menu-bar streak 🔥 | System tray | Needs an AppIndicator desktop; otherwise closing the window quits |
| Dictation fallback | 🌐 twice | Win + H | Desktop dictation tool |

Linux needs WebKitGTK 4.1 (preinstalled on Ubuntu 22.04+, Fedora 37+).
</details>

---

## 🛠 Develop

Requirements: Node 24+, pnpm 10, Rust (stable), [uv](https://docs.astral.sh/uv/) (it fetches Python for you). On Linux also the Tauri system libraries:
`sudo apt install libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev`.

```bash
pnpm install
uv sync                                       # ./.venv with the pinned Python packages (add --extra torch for PyTorch)
pnpm tauri icon src-tauri/icons/source.png   # once: generates the app icons
pnpm tauri dev                                # run with hot reload
pnpm build                                    # type-check + build the frontend
pnpm tauri build                              # installers for your OS
```

The book content (`public/content/`, `public/figs/`) and `notebooks/` are generated from LaTeX sources kept outside this repo, and committed here so anyone can build the app. The AI Safety and RL Environments guides are Markdown in `guides/`, built by `scripts/build_guides.py`.

| Path | What |
|---|---|
| `src/views/`, `src/components/` | React pages and UI |
| `src/lib/store.ts` | All saved progress and settings |
| `src/lib/platform.ts`, `src-tauri/src/platform.rs` | Everything that differs between macOS, Windows and Linux |
| `src-tauri/src/` | Rust: notebooks folder, marimo server, Python runner, Calendar, nudges, tray |
| `scripts/exercises_src.py` | All playground exercises (source of truth) |
| `scripts/build_notebooks.py` | Example marimo notebooks |
| `guides/*.md`, `scripts/build_guides.py` | Markdown guides (AI Safety, RL Environments) |
| `videos/web/scenes/*.js`, `videos/record.mjs` | 3D explainer scenes and the recorder that renders them to `public/videos/` |
| `src/nudges.json` | Every notification and roast |
| `AGENTS.md` | Conventions for AI coding agents (and humans) |

Every push to `master` refreshes the **nightly** pre-release. Pushing a tag like `v0.2.0` publishes a versioned release with every installer.

### macOS first launch
The builds are ad-hoc signed, not notarised by Apple (that needs a paid Apple Developer account), so macOS blocks
the first launch. Drag the app to Applications, then either:

- open **System Settings → Privacy & Security** and click **Open Anyway** next to Study Quest, or
- run `xattr -dr com.apple.quarantine "/Applications/Study Quest.app"` once in Terminal.

Building the `.dmg` yourself: `pnpm tauri build` skips create-dmg's Finder step (it only arranges the icons in
the DMG window and needs Automation permission, otherwise it fails with *Not authorized to send Apple events to
Finder*). Set `STUDY_QUEST_DMG_LAYOUT=1` to keep it.

---

## 💜 Contributing

**The most useful thing you can add is a topic.** If you know something a learner on the RL, LLM, System Design or AI Safety path should know, please share it. PRs of any size are welcome, and so are issues that just say *"this topic is missing"*.

### Suggest or write a topic
Open an issue with the **topic** label, or a PR. For AI Safety and RL Environments, add a `## Topic` straight to `guides/ai_safety.md` or `guides/rl_envs.md` and run `uv run python scripts/build_guides.py --check` (it runs every Python block). Other paths: a Markdown or LaTeX file under `contrib/topics/`. A good topic:

1. covers **one idea**, with a real-life analogy;
2. builds any math up **step by step** (each step becomes a reveal in *Quiz me*);
3. puts **key terms in bold** (*Explain it back* checks for them);
4. ends with a small runnable Python example and 2–3 key ideas.

Topics we'd love: newer RL-for-LLM methods, inference systems, evaluation, safety and alignment, distributed training, system design case studies, and clearer explanations of anything that confused you.

### Add a coding exercise
Add an `E(...)` entry to `scripts/exercises_src.py` (prompt, starter, solution, `test_*` tests, hints, `topic=` keyword), then run:

```bash
uv run python scripts/build_exercises.py
```

It refuses to write unless the solution passes every test and the starter fails at least one.

### Add a notebook
Add an entry to `INTERACTIVE` in `scripts/build_notebooks.py` (sliders and plots welcome), give it a group in `GROUPS` and a `topic` keyword, and check it runs with `marimo export html`.

### Add a 3D explainer video
Copy a file in `videos/web/scenes/`, export `meta` (title, path, duration, `keywords` that match lesson titles) and `setup(stage)` returning `renderAt(t)`, and add its id to `SCENES` in `videos/web/main.js`. Preview with `cd videos && npm install && npm run serve`, then render with `npm run record <id>` (needs ffmpeg and Google Chrome). Keep the light background and keep each video around 30 s.

### Add nudges and roasts
Add lines to `src/nudges.json`. Keep them funny and about **studying**: tease habits (scrolling, "I'll start Monday", 47 open tabs), never appearance, identity, health or anything that would actually hurt. ML puns score double.

### Pull requests
- One topic, exercise or fix per PR.
- `pnpm build` must pass; CI builds macOS, Windows and Linux on every PR.
- Say what a learner gains from the change.

---

## 🔒 Privacy

Everything stays on your computer. Progress is saved locally (export or reset it in Settings). Code runs with your own Python in a temp folder with a time limit. marimo listens only on `127.0.0.1` with a random token and stops when you close the window. On macOS, Calendar events are created through Automation, which asks your permission first.
