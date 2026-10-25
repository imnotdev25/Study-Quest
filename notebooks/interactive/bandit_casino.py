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
    mo.md("""
    # 🎰 The Bandit Casino
    Three machines, hidden payouts. Pull arms below and try to find the best one. Every pull you spend on a bad
    arm is **regret**. Then scroll down and watch three algorithms play the same game.
    """)
    return


@app.cell
def _(mo):
    get_pulls, set_pulls = mo.state([])
    return (get_pulls, set_pulls,)


@app.cell
def _(mo, np, set_pulls):
    casino_means = [0.30, 0.55, 0.45]
    _rng = np.random.default_rng()
    def _pull(i):
        set_pulls(lambda p: p + [(i, float(_rng.normal(casino_means[i], 0.3)))])
        return 0
    arms = mo.ui.array([mo.ui.button(label=f"Pull {c}", on_click=lambda _, i=i: _pull(i)) for i, c in enumerate("ABC")])
    reset = mo.ui.button(label="Reset", on_click=lambda _: set_pulls([]) or 0)
    mo.hstack([*arms, reset], justify="start")
    return (arms, casino_means, reset,)


@app.cell
def _(casino_means, get_pulls, mo, np):
    _p = get_pulls()
    _rows = []
    for _i, _c in enumerate("ABC"):
        _r = [r for a, r in _p if a == _i]
        _rows.append({"arm": _c, "pulls": len(_r), "mean reward": round(float(np.mean(_r)), 3) if _r else "-"})
    _regret = sum(max(casino_means) - casino_means[a] for a, _ in _p)
    _msg = "Pull some arms!" if not _p else f"**{len(_p)} pulls, regret so far {_regret:.2f}.**"
    if len(_p) >= 20:
        _msg += f" The true means were {casino_means}: arm B was best."
    mo.vstack([mo.ui.table(_rows, selection=None), mo.md(_msg)])
    return


@app.cell
def _(mo):
    eps = mo.ui.slider(0.0, 0.5, step=0.01, value=0.1, label="epsilon")
    steps = mo.ui.slider(200, 5000, step=100, value=2000, label="steps")
    runs = mo.ui.slider(10, 300, step=10, value=100, label="runs to average")
    mo.hstack([eps, steps, runs])
    return (eps, runs, steps,)


@app.cell
def _(eps, np, plt, runs, steps):
    _means = np.array([0.30, 0.55, 0.45])
    def _play(kind, rng):
        Q, N, opt = np.zeros(3), np.zeros(3), np.zeros(steps.value)
        for t in range(steps.value):
            if kind == "ucb":
                a = int(np.argmax(Q + 2 * np.sqrt(np.log(t + 1) / (N + 1e-9)))) if t >= 3 else t
            elif kind == "eps" and rng.random() < eps.value:
                a = int(rng.integers(3))
            else:
                a = int(np.argmax(Q))
            r = rng.normal(_means[a], 0.3)
            N[a] += 1; Q[a] += (r - Q[a]) / N[a]; opt[t] = a == 1
        return opt
    _fig, _ax = plt.subplots(figsize=(7, 3.2))
    for _kind, _lab in (("greedy", "greedy"), ("eps", f"epsilon-greedy ({eps.value})"), ("ucb", "UCB")):
        _rng = np.random.default_rng(0)
        _ax.plot(np.mean([_play(_kind, _rng) for _ in range(runs.value)], 0), label=_lab)
    _ax.set_xlabel("step"); _ax.set_ylabel("% optimal arm"); _ax.legend(); _ax.set_ylim(0, 1)
    _fig
    return


if __name__ == "__main__":
    app.run()
