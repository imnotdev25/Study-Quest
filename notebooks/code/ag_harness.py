# sq-meta: {"title": "ag_harness", "emoji": "📓", "group": "Agents & Evals", "kind": "code", "desc": "Agents ch. 'Harnesses': a minimal coding-agent harness. Tool registry with schemas, a ReAct loop,", "topic": ""}
import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

@app.cell
def _():
    import marimo as mo
    import sys as _sys
    _sys.path.insert(0, '/Users/bhavik/Developer/ClaudeProjects/RL/llm_book/code')
    return (mo,)


@app.cell
def _(mo):
    mo.md("# ag_harness\n\nAgents ch. 'Harnesses': a minimal coding-agent harness. Tool registry with schemas, a ReAct loop,\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Agents ch. 'Harnesses': a minimal coding-agent harness. Tool registry with schemas, a ReAct loop,
    a permission gate, memory file loading, output truncation, token accounting, and a harness-side
    definition of done (it runs the tests itself before accepting 'finished').

    The model is a scripted stand-in so the harness runs offline. It sees only the transcript, as a real
    model would. Replace ScriptedModel.step with an API call (see ag_api.py) to drive it with an LLM."""
    import json
    import pathlib
    import re
    import shutil
    import subprocess
    import sys
    import tempfile

    FILES = {
        "AGENTS.md": "# Project notes\n- Run tests with run_tests before saying you are done.\n- Never delete files.\n",
        "calc.py": "def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n",
        "test_calc.py": "from calc import add, mul\n\nassert add(2, 3) == 5, f'add(2, 3) = {add(2, 3)}'\n"
                        "assert mul(2, 3) == 6\nprint('PASS')\n",
        "NOTES.txt": "scratch\n",
    }

    TOOLS = {
        "list_files": {"desc": "List files in the workspace.", "params": {}},
        "read_file": {"desc": "Read a file; returns numbered lines.", "params": {"path": str}},
        "grep": {"desc": "Search all files for a regex; returns path:line: text.", "params": {"pattern": str}},
        "edit": {"desc": "Replace one exact, unique occurrence of `old` with `new` in `path`.",
                 "params": {"path": str, "old": str, "new": str}},
        "run_tests": {"desc": "Run the test suite; returns PASS or the failure output.", "params": {}},
        "delete_file": {"desc": "Delete a file.", "params": {"path": str}, "dangerous": True},
    }


    class Workspace:
        def __init__(self, files):
            self.root = pathlib.Path(tempfile.mkdtemp(prefix="agent_ws_"))
            for name, text in files.items():
                (self.root / name).write_text(text)

        def path(self, p):
            full = (self.root / p).resolve()
            if self.root.resolve() not in full.parents:          # no escaping the workspace
                raise ValueError("path outside workspace")
            return full

        def files(self):
            return sorted(p for p in self.root.iterdir() if p.is_file())

        def list_files(self):
            return "\n".join(sorted(p.name for p in self.files()))

        def read_file(self, path):
            return "\n".join(f"{i + 1:3d} {l}" for i, l in enumerate(self.path(path).read_text().splitlines()))

        def grep(self, pattern):
            rx = re.compile(pattern)
            return "\n".join(f"{p.name}:{i + 1}: {l}" for p in self.files()
                             for i, l in enumerate(p.read_text().splitlines()) if rx.search(l)) or "no matches"

        def edit(self, path, old, new):
            text = self.path(path).read_text()
            n = text.count(old)
            if n != 1:
                raise ValueError(f"`old` occurs {n} times in {path}; it must be unique. Include more context.")
            self.path(path).write_text(text.replace(old, new))
            return f"edited {path}"

        def run_tests(self):
            r = subprocess.run([sys.executable, "test_calc.py"], cwd=self.root, capture_output=True, text=True, timeout=10)
            return "PASS" if r.returncode == 0 else (r.stdout + r.stderr).strip().splitlines()[-1]

        def delete_file(self, path):
            self.path(path).unlink()
            return f"deleted {path}"


    class Harness:
        def __init__(self, ws, model, allow_dangerous=(), max_steps=15, max_obs_chars=300):
            self.ws, self.model, self.allow = ws, model, set(allow_dangerous)
            self.max_steps, self.max_obs = max_steps, max_obs_chars
            self.input_tokens = 0

        def system_prompt(self):
            tools = "\n".join(f"- {n}({', '.join(t['params'])}): {t['desc']}" for n, t in TOOLS.items())
            memory = (self.ws.root / "AGENTS.md").read_text()      # project memory file, loaded every session
            return (f"You are a coding agent. Use one tool per step.\nFormat: Thought: ... then "
                    f"Action: tool {{json args}}, or Final: <summary>.\nTools:\n{tools}\n\n{memory}")

        def call_tool(self, name, args):
            spec = TOOLS.get(name)
            if spec is None:
                return f"error: unknown tool {name}"
            bad = [k for k, t in spec["params"].items() if not isinstance(args.get(k), t)]
            if bad:
                return f"error: missing or invalid arguments {bad}"
            if spec.get("dangerous") and name not in self.allow:
                return f"permission denied: {name} needs user approval (policy)"
            try:
                out = getattr(self.ws, name)(**args)
            except Exception as e:                                # errors become observations, not crashes
                return f"error: {e}"
            return out if len(out) <= self.max_obs else out[:self.max_obs] + f"... [{len(out) - self.max_obs} chars truncated]"

        def run(self, goal):
            msgs = [{"role": "system", "content": self.system_prompt()}, {"role": "user", "content": goal}]
            for step in range(1, self.max_steps + 1):
                self.input_tokens += sum(len(m["content"]) for m in msgs) // 4   # whole transcript re-sent
                act = self.model.step(msgs)
                if "final" in act:
                    msgs.append({"role": "assistant", "content": "Final: " + act["final"]})
                    check = self.ws.run_tests()                    # definition of done, enforced by the harness
                    print(f"[step {step:2d}] Final: {act['final']}  -> harness verification: {check}")
                    if check == "PASS":
                        return step
                    msgs.append({"role": "user", "content": f"Verification failed: {check}. Keep working."})
                    continue
                msgs.append({"role": "assistant",
                             "content": f"Thought: {act['thought']}\nAction: {act['tool']} {json.dumps(act['args'])}"})
                obs = self.call_tool(act["tool"], act["args"])
                msgs.append({"role": "user", "content": "Observation: " + obs})
                short = obs.replace("\n", " | ")
                print(f"[step {step:2d}] {act['tool']}{json.dumps(act['args'])[:60]}\n          -> {short[:95]}")
            return None


    class ScriptedModel:
        """Offline stand-in for an LLM: a fixed plan that reacts to what it observes."""

        def __init__(self):
            self.s = 0

        def step(self, msgs):
            obs, s = msgs[-1]["content"], self.s
            act = lambda t, tool, **a: {"thought": t, "tool": tool, "args": a}
            plan = {
                0: lambda: act("Reproduce the failure first.", "run_tests"),
                1: lambda: act("add is wrong; find it.", "grep", pattern=r"def add"),
                2: lambda: act("Read the file.", "read_file", path="calc.py"),
                3: lambda: act("Change the return line.", "edit", path="calc.py", old="return a", new="return abs(a"),
                4: lambda: act("Too vague; be specific.", "edit", path="calc.py", old="return a - b", new="return abs(a - b)"),
                5: lambda: act("Tidy up the scratch file.", "delete_file", path="NOTES.txt"),
                6: lambda: {"final": "Fixed add."},
                7: lambda: act("abs() is wrong: add must sum.", "edit", path="calc.py", old="return abs(a - b)", new="return a + b"),
                8: lambda: act("Check.", "run_tests"),
                9: lambda: {"final": "add now returns a + b; tests pass."},
            }
            self.s += 1
            return plan[s]()


    ws = Workspace(FILES)
    h = Harness(ws, ScriptedModel())
    steps = h.run("The add() test fails. Fix it.")
    print(f"[harness] verified done in {steps} steps; ~{h.input_tokens:,} input tokens over the run "
          f"(the full transcript is re-sent every step)")
    print(f"[harness] NOTES.txt still exists: {(ws.root / 'NOTES.txt').exists()}")
    shutil.rmtree(ws.root)
    return


if __name__ == "__main__":
    app.run()
