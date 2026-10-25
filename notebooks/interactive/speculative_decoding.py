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
    acc = mo.ui.slider(0.3, 0.98, step=0.01, value=0.8, label="per-token acceptance")
    cost = mo.ui.slider(0.0, 0.3, step=0.01, value=0.05, label="draft cost / target cost")
    mo.vstack([mo.md("# 🏎️ Speculative decoding speedup"), mo.hstack([acc, cost])])
    return (acc, cost,)


@app.cell
def _(acc, cost, mo, np, plt):
    _k = np.arange(1, 21); _a, _c = acc.value, cost.value
    _tok = (1 - _a ** (_k + 1)) / (1 - _a)
    _sp = _tok / (1 + _c * _k)
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.plot(_k, _tok, "o-", label="tokens per target pass"); _ax.plot(_k, _sp, "s-", label="wall-clock speedup")
    _ax.set_xlabel("draft length k"); _ax.legend(); _ax.grid(alpha=0.3)
    mo.vstack([_fig, mo.md(f"Best k = **{_k[_sp.argmax()]}**, speedup **{_sp.max():.2f}x**. "
                           "Longer drafts stop paying once a^k is small.")])
    return


if __name__ == "__main__":
    app.run()
