# sq-meta: {"title": "Central Limit Theorem", "emoji": "🔔", "group": "Probability & Statistics", "kind": "interactive", "desc": "Average n draws from a skewed distribution and watch a bell curve appear.", "topic": "central limit"}
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
    clt_dist = mo.ui.dropdown(["exponential", "uniform", "bernoulli(0.1)", "die roll"], value="exponential", label="distribution")
    clt_n = mo.ui.slider(1, 100, step=1, value=5, label="n draws per average")
    mo.vstack([mo.md("# 🔔 The central limit theorem"), mo.hstack([clt_dist, clt_n])])
    return (clt_dist, clt_n,)


@app.cell
def _(clt_dist, clt_n, mo, np, plt):
    _rng = np.random.default_rng(0)
    _draw = {"exponential": lambda s: _rng.exponential(1, s), "uniform": lambda s: _rng.uniform(0, 1, s),
             "bernoulli(0.1)": lambda s: (_rng.random(s) < 0.1).astype(float), "die roll": lambda s: _rng.integers(1, 7, s).astype(float)}[clt_dist.value]
    _m = _draw((20000, clt_n.value)).mean(1)
    _mu, _sd = _m.mean(), _m.std()
    _x = np.linspace(_m.min(), _m.max(), 300)
    _fig, _ax = plt.subplots(figsize=(7, 3.2))
    _ax.hist(_m, bins=60, density=True, color="#c4b5fd")
    _ax.plot(_x, np.exp(-(_x - _mu) ** 2 / (2 * _sd ** 2)) / np.sqrt(2 * np.pi * _sd ** 2), color="#6d28d9", lw=2)
    _skew = np.mean(((_m - _mu) / _sd) ** 3)
    mo.vstack([_fig, mo.md(f"Mean {_mu:.3f}, sd {_sd:.4f}, skewness **{_skew:+.2f}** (0 for a perfect bell). The sd shrinks like 1/sqrt(n).")])
    return


if __name__ == "__main__":
    app.run()
