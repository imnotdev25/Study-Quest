# sq-meta: {"title": "KL Estimators (k1 vs k3)", "emoji": "📏", "group": "Reinforcement Learning", "kind": "interactive", "desc": "The two Monte Carlo KL estimators used in RLHF, side by side.", "topic": "KL"}
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
    kl_shift = mo.ui.slider(0.05, 2.0, step=0.05, value=0.5, label="distance between policies (mean shift)")
    kl_n = mo.ui.slider(10, 2000, step=10, value=200, label="samples per estimate")
    mo.vstack([mo.md("# 📏 Estimating KL(p‖q) from samples\np = N(0,1) is the policy, q = N(shift,1) the reference. True KL = shift²/2."),
               mo.hstack([kl_shift, kl_n])])
    return (kl_n, kl_shift,)


@app.cell
def _(kl_n, kl_shift, mo, np, plt):
    _rng = np.random.default_rng(1)
    _x = _rng.normal(0, 1, (3000, kl_n.value))
    _logr = -0.5 * (_x - kl_shift.value) ** 2 + 0.5 * _x ** 2          # log q(x)/p(x)
    _k1 = (-_logr).mean(1)
    _k3 = (np.exp(_logr) - 1 - _logr).mean(1)
    _true = kl_shift.value ** 2 / 2
    _fig, _ax = plt.subplots(figsize=(7, 3.2))
    _ax.hist(_k1, bins=60, alpha=0.6, label=f"k1 = -log r   (sd {_k1.std():.3f})", color="#f59e0b")
    _ax.hist(_k3, bins=60, alpha=0.6, label=f"k3 = r-1-log r (sd {_k3.std():.3f})", color="#7c3aed")
    _ax.axvline(_true, color="black", ls="--", label=f"true KL {_true:.3f}"); _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md("Both are unbiased. k3 is never negative and has lower variance when the policies are close, "
                           "which is why GRPO-style trainers use it for the KL penalty.")])
    return


if __name__ == "__main__":
    app.run()
