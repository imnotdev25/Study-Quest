# sq-meta: {"title": "Calibration and Abstention", "emoji": "🎚️", "group": "AI Safety", "kind": "interactive", "desc": "Reliability diagrams, ECE, and the threshold that maximises score when wrong answers cost points.", "topic": "calibration"}
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
    cal_sharp = mo.ui.slider(0.2, 2.5, step=0.05, value=0.5, label="confidence exponent (<1 = overconfident)")
    cal_pen = mo.ui.slider(0.0, 4.0, step=0.25, value=1.0, label="penalty for a wrong answer")
    mo.vstack([mo.md("# 🎚️ Calibration and knowing when to abstain"), mo.hstack([cal_sharp, cal_pen])])
    return (cal_pen, cal_sharp,)


@app.cell
def _(cal_pen, cal_sharp, mo, np, plt):
    _rng = np.random.default_rng(0)
    _p = _rng.beta(2, 1.5, 5000); _ok = _rng.random(5000) < _p
    _conf = np.clip(_p ** cal_sharp.value, 0, 1)
    _edges = np.linspace(0, 1, 11); _xs, _ys, _ece = [], [], 0.0
    for _lo, _hi in zip(_edges[:-1], _edges[1:]):
        _m = (_conf > _lo) & (_conf <= _hi)
        if _m.any():
            _xs.append(_conf[_m].mean()); _ys.append(_ok[_m].mean()); _ece += _m.mean() * abs(_ys[-1] - _xs[-1])
    _ts = np.linspace(0, 1, 101)
    _score = [((_conf >= _t) * np.where(_ok, 1, -cal_pen.value)).mean() for _t in _ts]
    _best = _ts[int(np.argmax(_score))]
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    _a1.plot([0, 1], [0, 1], "--", color="gray"); _a1.plot(_xs, _ys, "o-")
    _a1.set_xlabel("stated confidence"); _a1.set_ylabel("accuracy"); _a1.set_title(f"reliability (ECE {_ece:.3f})")
    _a2.plot(_ts, _score); _a2.axvline(_best, color="red", ls=":")
    _a2.set_xlabel("answer only if confidence ≥"); _a2.set_title("expected score per question")
    mo.vstack([_fig, mo.md(f"Best threshold **{_best:.2f}**: answer {np.mean(_conf >= _best):.0%} of questions. "
                           "With penalty 0 (binary grading) the best policy is to always answer. That's the guessing incentive.")])
    return


if __name__ == "__main__":
    app.run()
