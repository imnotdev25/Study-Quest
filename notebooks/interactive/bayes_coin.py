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
    a = mo.ui.slider(0.5, 20, step=0.5, value=2, label="prior a (imaginary heads)")
    b = mo.ui.slider(0.5, 20, step=0.5, value=2, label="prior b (imaginary tails)")
    heads = mo.ui.slider(0, 100, step=1, value=7, label="observed heads")
    tails = mo.ui.slider(0, 100, step=1, value=3, label="observed tails")
    mo.vstack([mo.md("# 🪙 Beta-Bernoulli updating"), mo.hstack([a, b]), mo.hstack([heads, tails])])
    return (a, b, heads, tails,)


@app.cell
def _(a, b, heads, mo, np, plt, tails):
    from math import lgamma
    _t = np.linspace(0.001, 0.999, 400)
    def _beta(x, p, q):
        return np.exp((p - 1) * np.log(x) + (q - 1) * np.log(1 - x) - (lgamma(p) + lgamma(q) - lgamma(p + q)))
    _A, _B = a.value + heads.value, b.value + tails.value
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.plot(_t, _beta(_t, a.value, b.value), "--", color="gray", label=f"prior Beta({a.value:g},{b.value:g})")
    _ax.plot(_t, _beta(_t, _A, _B), lw=2.5, color="#6366f1", label=f"posterior Beta({_A:g},{_B:g})")
    _ax.axvline(heads.value / max(1, heads.value + tails.value), color="#dc2626", ls=":", label="MLE h/n"); _ax.legend()
    mo.vstack([_fig, mo.md(f"Posterior mean **{_A / (_A + _B):.3f}**: the prior pulls the MLE toward {a.value / (a.value + b.value):.2f} "
                           "until the data outweighs it.")])
    return


if __name__ == "__main__":
    app.run()
