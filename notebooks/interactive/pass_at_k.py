# sq-meta: {"title": "pass@k vs pass^k", "emoji": "🎯", "group": "Agents & Evals", "kind": "interactive", "desc": "Same agent, two very different reliability numbers.", "topic": "pass@k"}
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
    pk_p = mo.ui.slider(0.05, 0.95, step=0.05, value=0.4, label="per-attempt success p")
    mo.vstack([mo.md("# 🎯 pass@k vs pass^k"), pk_p])
    return (pk_p,)


@app.cell
def _(mo, np, pk_p, plt):
    _k = np.arange(1, 17)
    _fig, _ax = plt.subplots(figsize=(7, 3.2))
    _ax.plot(_k, 1 - (1 - pk_p.value) ** _k, "o-", label="pass@k: at least one of k works")
    _ax.plot(_k, pk_p.value ** _k, "s-", label="pass^k: all k work")
    _ax.set_xlabel("k"); _ax.set_ylim(0, 1); _ax.legend(); _ax.grid(alpha=0.3)
    mo.vstack([_fig, mo.md("Benchmarks love pass@k. Users live with pass^k: an agent that works 90% of the time fails one task in ten.")])
    return


if __name__ == "__main__":
    app.run()
