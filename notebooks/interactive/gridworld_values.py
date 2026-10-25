# sq-meta: {"title": "Gridworld Value Iteration", "emoji": "🗺️", "group": "Reinforcement Learning", "kind": "interactive", "desc": "Change gamma, slip and step cost; watch values and the policy update.", "topic": "value iteration"}
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
    gw_gamma = mo.ui.slider(0.5, 0.99, step=0.01, value=0.9, label="gamma")
    gw_slip = mo.ui.slider(0.0, 0.4, step=0.05, value=0.1, label="slip probability")
    gw_step = mo.ui.slider(-0.3, 0.0, step=0.01, value=-0.04, label="reward per step")
    mo.vstack([mo.md("# 🗺️ Gridworld value iteration\nGoal +1 (top right), pit -1 below it, grey cells are walls."),
               mo.hstack([gw_gamma, gw_slip, gw_step])])
    return (gw_gamma, gw_slip, gw_step,)


@app.cell
def _(gw_gamma, gw_slip, gw_step, mo, np, plt):
    _N, _goal, _pit, _wall = 5, (0, 4), (1, 4), {(1, 1), (2, 1), (3, 3)}
    _acts, _arrows = [(-1, 0), (1, 0), (0, -1), (0, 1)], "↑↓←→"
    def _move(_s, _a):
        _r, _c = _s[0] + _a[0], _s[1] + _a[1]
        return _s if not (0 <= _r < _N and 0 <= _c < _N) or (_r, _c) in _wall else (_r, _c)
    def _qs(_V, _s):
        _out = []
        for _a in _acts:
            _branches = [(1 - gw_slip.value, _a), (gw_slip.value / 2, (_a[1], _a[0])), (gw_slip.value / 2, (-_a[1], -_a[0]))]
            _out.append(sum(_p * (gw_step.value + gw_gamma.value * _V[_move(_s, _b)]) for _p, _b in _branches))
        return _out
    _V, _iters = np.zeros((_N, _N)), 0
    for _iters in range(1, 500):
        _V2 = _V.copy()
        for _r in range(_N):
            for _c in range(_N):
                if (_r, _c) not in _wall and (_r, _c) not in (_goal, _pit):
                    _V2[_r, _c] = max(_qs(_V, (_r, _c)))
        _V2[_goal], _V2[_pit] = 1.0, -1.0
        _done = np.abs(_V2 - _V).max() < 1e-6
        _V = _V2
        if _done:
            break
    _fig, _ax = plt.subplots(figsize=(5, 5))
    _show = _V.copy()
    for _w in _wall:
        _show[_w] = np.nan
    _ax.imshow(_show, cmap="RdYlGn", vmin=-1, vmax=1)
    for _r in range(_N):
        for _c in range(_N):
            if (_r, _c) in _wall:
                continue
            _lab = "GOAL" if (_r, _c) == _goal else "PIT" if (_r, _c) == _pit else f"{_V[_r, _c]:.2f}\n{_arrows[int(np.argmax(_qs(_V, (_r, _c))))]}"
            _ax.text(_c, _r, _lab, ha="center", va="center", fontsize=9)
    _ax.set_xticks([]); _ax.set_yticks([])
    mo.vstack([_fig, mo.md(f"Converged in **{_iters}** sweeps. Make steps expensive and the agent risks the pit to finish fast; "
                           "raise slip and it takes the safe long way round.")])
    return


if __name__ == "__main__":
    app.run()
