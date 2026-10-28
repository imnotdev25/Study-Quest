# sq-meta: {"title": "Sandboxed Rollout Harness", "emoji": "📦", "group": "RL Environments", "kind": "interactive", "desc": "Run 'policy' code against a naive and a hardened test harness. Watch the hacks get through one of them.", "topic": "sandbox"}
import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return (mo, np, plt,)


@app.cell
def _(mo):
    sb_attacks = {
        "honest solution": "def solve(a, b):\n    return a + b\n",
        "sys.exit(0) before tests": "import sys\nsys.exit(0)\n",
        "os._exit(0)": "import os\nos._exit(0)\n",
        "__eq__ always True": "class A:\n    def __eq__(s, o): return True\ndef solve(a, b):\n    return A()\n",
        "hard-code visible test": "def solve(a, b):\n    return 5\n",
        "infinite loop": "def solve(a, b):\n    while True: pass\n",
        "overwrite the test file": "open('tests.py', 'w').write('pass\\n')\ndef solve(a, b):\n    return 0\n",
    }
    sb_pick = mo.ui.dropdown(list(sb_attacks), value="honest solution", label="preset")
    mo.vstack([mo.md("# 📦 Sandboxed rollout harness\nNaive: run `tests.py` that imports the solution, reward = exit code 0. "
                     "Hardened: tests restored from a pristine copy, hidden tests, type-checked JSON comparison, result file written by the harness, timeout."), sb_pick])
    return (sb_attacks, sb_pick,)


@app.cell
def _(mo, sb_attacks, sb_pick):
    sb_code = mo.ui.code_editor(value=sb_attacks[sb_pick.value], language="python", min_height=120)
    sb_run = mo.ui.run_button(label="Run both harnesses")
    mo.vstack([sb_code, sb_run])
    return (sb_code, sb_run,)


@app.cell
def _(mo, sb_code, sb_run):
    import json as _json, pathlib as _pl, subprocess as _sp, sys as _sys, tempfile as _tf
    mo.stop(not sb_run.value, mo.md("Press **Run both harnesses**."))
    _VISIBLE = "from sol import solve\nassert solve(2, 3) == 5\n"
    _HIDDEN = [[[2, 3], 5], [[10, -4], 6], [[7, 7], 14]]
    _HARNESS = """
    import json, sys, importlib.util
    res = []
    try:
        spec = importlib.util.spec_from_file_location('sol', 'sol.py'); m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        for args, want in json.loads(sys.argv[1]):
            got = m.solve(*args)
            res.append(type(got) is type(want) and json.dumps(got) == json.dumps(want))
    except BaseException:
        res.append(False)
    open(sys.argv[2], 'w').write(json.dumps(res))
    """
    def _naive(code):
        with _tf.TemporaryDirectory() as d:
            p = _pl.Path(d); (p / "sol.py").write_text(code); (p / "tests.py").write_text(_VISIBLE)
            try:
                _sp.run([_sys.executable, "-c", "import sol"], cwd=d, timeout=3, capture_output=True)   # policy code runs first
                r = _sp.run([_sys.executable, "tests.py"], cwd=d, timeout=3, capture_output=True)
                return float(r.returncode == 0)
            except _sp.TimeoutExpired:
                return 0.0
    def _hardened(code):
        with _tf.TemporaryDirectory() as d, _tf.TemporaryDirectory() as g:
            p = _pl.Path(d); (p / "sol.py").write_text(code); (p / "h.py").write_text(_HARNESS)
            out = _pl.Path(g) / "result.json"                     # outside the policy's directory
            try:
                _sp.run([_sys.executable, "-I", "h.py", _json.dumps(_HIDDEN), str(out)], cwd=d, timeout=3, capture_output=True)
            except _sp.TimeoutExpired:
                return 0.0
            if not out.exists(): return 0.0
            res = _json.loads(out.read_text())
            return float(len(res) == len(_HIDDEN) and all(res))
    mo.md(f"| harness | reward |\n|---|---|\n| naive | **{_naive(sb_code.value)}** |\n| hardened | **{_hardened(sb_code.value)}** |")
    return


if __name__ == "__main__":
    app.run()
