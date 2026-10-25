# sq-meta: {"title": "Gradient Descent Valley", "emoji": "⛰️", "group": "Math & Optimisation", "kind": "interactive", "desc": "Condition number vs learning rate on a quadratic bowl.", "topic": "optimisation"}
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
    kappa = mo.ui.slider(1, 50, step=1, value=10, label="condition number kappa")
    lr = mo.ui.slider(0.01, 2.2, step=0.01, value=0.9, label="learning rate x (1/L)")
    mom = mo.ui.slider(0.0, 0.95, step=0.05, value=0.0, label="momentum")
    mo.vstack([mo.md("# ⛰️ Gradient descent in a valley"), mo.hstack([kappa, lr, mom])])
    return (kappa, lr, mom,)


@app.cell
def _(kappa, lr, mo, mom, np, plt):
    _A = np.diag([1.0, float(kappa.value)]); _eta = lr.value / kappa.value
    _x, _v, _path = np.array([4.0, 1.5]), np.zeros(2), []
    for _ in range(60):
        _path.append(_x.copy()); _v = mom.value * _v - _eta * (_A @ _x); _x = _x + _v
    _path = np.array(_path)
    _g = np.linspace(-5, 5, 200); _X, _Y = np.meshgrid(_g, _g)
    _fig, _ax = plt.subplots(figsize=(6, 3.6))
    _ax.contour(_X, _Y, 0.5 * (_X ** 2 + kappa.value * _Y ** 2), levels=15, cmap="Blues")
    _ax.plot(_path[:, 0], _path[:, 1], "o-", color="#dc2626", ms=3); _ax.set_xlim(-5, 5); _ax.set_ylim(-2.5, 2.5)
    _d = np.linalg.norm(_path[-1])
    mo.vstack([_fig, mo.md(f"Distance to the minimum after 60 steps: **{_d:.2e}**. "
                           + ("Diverged: lr above 2/L." if _d > 10 else "Raise kappa and watch it zig-zag; add momentum to fix it."))])
    return


if __name__ == "__main__":
    app.run()
