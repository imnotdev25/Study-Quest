#!/usr/bin/env python3
"""Verify every exercise (solution passes all tests, starter fails at least one) and write
public/content/exercises.json plus src-tauri/resources/harness.py.

Usage: ../.venv-post/bin/python scripts/build_exercises.py
"""
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
APP = HERE.parent
sys.path.insert(0, str(HERE))
from exercises_src import EXERCISES  # noqa: E402


def run(code, tests):
    with tempfile.TemporaryDirectory() as d:
        (pathlib.Path(d) / "code.py").write_text(code)
        (pathlib.Path(d) / "tests.py").write_text(tests)
        r = subprocess.run([sys.executable, str(HERE / "harness.py"), "code.py", "tests.py"], cwd=d,
                           capture_output=True, text=True, timeout=60)
    lines = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]
    return lines[-1] if lines else {"passed": 0, "total": 0, "error": r.stderr[-800:]}, lines


ok = True
ids = set()
for ex in EXERCISES:
    assert ex["id"] not in ids, ex["id"]
    ids.add(ex["id"])
    sol, lines = run(ex["solution"], ex["tests"])
    st, _ = run(ex["starter"], ex["tests"])
    good = sol["total"] > 0 and sol["passed"] == sol["total"] and st["passed"] < st["total"]
    ok &= good
    status = "ok  " if good else "FAIL"
    print(f"{status} {ex['track']:14s} {ex['id']:16s} solution {sol['passed']}/{sol['total']}  starter {st['passed']}/{st['total']}")
    if not good:
        for l in lines:
            if not l.get("ok", True) or l["type"] == "load_error":
                print("     ", l.get("name", ""), l.get("error", "")[:300])

if not ok:
    sys.exit("some exercises failed verification")
(APP / "public/content").mkdir(parents=True, exist_ok=True)
(APP / "public/content/exercises.json").write_text(json.dumps(EXERCISES, ensure_ascii=False))
res = APP / "src-tauri/resources"
res.mkdir(parents=True, exist_ok=True)
shutil.copy(HERE / "harness.py", res / "harness.py")
tracks = {}
for e in EXERCISES:
    tracks[e["track"]] = tracks.get(e["track"], 0) + 1
print(f"wrote {len(EXERCISES)} exercises: {tracks}")
