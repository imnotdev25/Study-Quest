# 🦦 Study Quest

A macOS app for learning **Reinforcement Learning, LLMs and System Design** interactively. It turns five long-form books into lessons you can read, run and quiz yourself on, with a coding playground, live [marimo](https://marimo.io) notebooks, and Byte the otter, who nudges (and roasts) you until you study.

> Built with Tauri 2, React and Rust. Runs on macOS, Windows and Linux.

## Features

- **Three learning paths, one at a time:** 🎯 RL, 🧠 LLMs (LLM fundamentals + pre/post-training), 🏗️ System Design (system design + DSA/ML interviews). Each lesson keeps its math (KaTeX), diagrams (compiled from TikZ) and code.
- **Runnable lessons:** every Python block has **▶ Run / Edit**, notebook-style, with output and plots inline.
- **Notebooks inside lessons:** reactive marimo notebooks open right in the page. Create your own; they live in `~/.study-quest/notebooks`.
- **Coding playground:** 50 tested exercises (DSA, ML math, transformers, RL, inference, system design, agents) plus a scratchpad.
- **Active learning:** warm-up predictions, *Quiz me* step-by-step reveal, *Explain it back* with key-term check, *Connect it*, muddiest point, confidence before/after. Voice typing on every box.
- **Habit loop:** streaks, XP, badges, activity heatmap, native notifications with an adjustable troll level, menu-bar streak, and macOS Calendar sync for study blocks.

## Install

Download from [Releases](../../releases):

| OS | File | First launch (builds aren't signed yet) |
|---|---|---|
| macOS (Apple Silicon / Intel) | `.dmg` | Right-click the app → **Open** |
| Windows 10/11 | `.exe` installer or `.msi` | SmartScreen: **More info → Run anyway** |
| Linux (x86_64) | `.AppImage` or `.deb` | `chmod +x Study*.AppImage && ./Study*.AppImage`, or `sudo apt install ./study-quest*.deb` |

Linux needs WebKitGTK 4.1 (preinstalled on Ubuntu 22.04+, Fedora 37+). The tray icon needs an AppIndicator-capable desktop; without one, closing the window quits the app.

What differs by platform: Calendar sync is macOS-only (Windows and Linux export an `.ics` to your calendar app instead). The translucent sidebar is macOS-only. Voice typing falls back to the OS dictation (🌐 twice on macOS, Win + H on Windows).

Exercises and notebooks use your local Python. Install marimo into it:

```bash
python3 -m pip install marimo numpy matplotlib
```

Then point **Settings → Python** at that interpreter (it auto-detects common locations).

## Develop

Requirements: macOS 13+, Node 22+, pnpm 10, Rust (stable), Python 3.11.

```bash
cd study_app
pnpm install
pnpm tauri icon src-tauri/icons/source.png   # once, generates the app icons
pnpm tauri dev                                # run with hot reload
pnpm tauri build                              # produce .app and .dmg
```

Regenerating the book content needs more tools (pandoc, tectonic, and the repo's `.venv` / `.venv-post` Python environments):

```bash
pnpm content   # books → public/content, notebooks → notebooks/, exercises → public/content/exercises.json
```

The generated files are committed so CI can build without those tools.

### Layout

| Path | What |
|---|---|
| `src/` | React frontend (views, components, store) |
| `src-tauri/src/` | Rust: notebooks folder, marimo server, Python runner, Calendar, nudges, tray |
| `scripts/build_content.py` | LaTeX books → per-topic HTML + SVG figures |
| `scripts/exercises_src.py` | All playground exercises (source of truth) |
| `scripts/build_notebooks.py` | Example marimo notebooks |
| `src/nudges.json` | Every notification message |

## Contributing 💜

**The most useful thing you can add is a topic.** If you know something well that a learner on the RL, LLM or System Design path should know, please contribute it. Pull requests of any size are welcome, and so are issues that just say "this topic is missing".

### Add or improve a topic

Topics come from the LaTeX books (`../rl_enhanced.tex`, `../llm_book/`, `../posttrain_book/`, and the System Design books). A topic is one `\section` (one `\chapter` in the System Design books).

1. Write it in LaTeX in the matching book. Use the existing boxes so the app can style them and quiz on them:
   `startbox` (where to begin), `derive{title}` with `\step{n}` (derivations, revealed one step at a time in *Quiz me*), `algobox{title}`, `keyideas`, `analogy{title}`.
2. Put **key terms in bold** (`\textbf{...}`): *Explain it back* checks whether learners use them.
3. Prefer one idea per topic, a real-life analogy, a derivation from first principles, and a small runnable Python example.
4. Run `pnpm content` and check the lesson in `pnpm tauri dev`.

Topic ideas we'd love: newer RL-for-LLM methods, inference systems, evaluation, safety and alignment, distributed training, interview-style system design case studies, and clearer explanations of anything that confused you.

### Add a coding exercise

Add an `E(...)` entry to `scripts/exercises_src.py` with a prompt, starter, reference solution, tests (`test_*` functions) and hints. Then run:

```bash
../.venv-post/bin/python scripts/build_exercises.py
```

It refuses to write unless the solution passes every test and the starter fails at least one. Set `topic=` to a keyword so the exercise shows up inside matching lessons.

### Add a notebook

Add an entry to `INTERACTIVE` in `scripts/build_notebooks.py` (sliders and plots welcome), give it a group in `GROUPS` and a `topic` keyword, then run `pnpm content` and `marimo export html` on it to check it runs.

### Add nudges and roasts

Add lines to `src/nudges.json`. Keep them funny and about **studying**: tease habits (scrolling, procrastinating, "I'll start Monday"), never appearance, identity, health or anything that would actually hurt. ML puns score double.

### Pull requests

- One topic, exercise or fix per PR is easiest to review.
- Make sure `pnpm build` passes. CI builds the macOS app on every PR.
- Describe what a learner gains from the change.

## Privacy

Everything stays on your Mac. Progress is stored locally (export it from Settings). Code runs with your own Python in a temp folder. marimo listens only on `127.0.0.1` with a random token. Calendar events are created through macOS Automation, which asks your permission first.
