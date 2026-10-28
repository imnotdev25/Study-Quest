# sq-meta: {"title": "Linear Probe for Truth", "emoji": "🔦", "group": "AI Safety", "kind": "interactive", "desc": "Train a probe on fake activations, then check it against random labels and a causal edit.", "topic": "probe"}
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
    pr_signal = mo.ui.slider(0.0, 2.0, step=0.05, value=1.2, label="strength of the truth direction")
    pr_n = mo.ui.slider(50, 3000, step=50, value=600, label="training examples")
    pr_dim = mo.ui.slider(16, 512, step=16, value=256, label="activation dimension")
    mo.vstack([mo.md("# 🔦 Probing a 'truth' direction"), pr_signal, pr_n, pr_dim])
    return (pr_dim, pr_n, pr_signal,)


@app.cell
def _(mo, np, plt, pr_dim, pr_n, pr_signal):
    _rng = np.random.default_rng(0)
    _d, _n = pr_dim.value, pr_n.value + 1000
    _dir = _rng.normal(size=_d); _dir /= np.linalg.norm(_dir)
    _y = _rng.integers(0, 2, _n)
    _A = _rng.normal(size=(_n, _d)) + np.outer(2 * _y - 1, _dir) * pr_signal.value
    def _fit(X, y):
        w = np.zeros(X.shape[1]); b = 0.0
        for _ in range(300):
            p = 1 / (1 + np.exp(-(X @ w + b))); w -= 0.5 * (X.T @ (p - y) / len(y) + 1e-2 * w); b -= 0.5 * (p - y).mean()
        return w, b
    _tr, _te = slice(0, pr_n.value), slice(pr_n.value, None)
    _w, _b = _fit(_A[_tr], _y[_tr])
    _acc = (((_A[_te] @ _w + _b) > 0) == _y[_te]).mean()
    _yr = _rng.integers(0, 2, _n)
    _wr, _br = _fit(_A[_tr], _yr[_tr])
    _acc_r = (((_A[_te] @ _wr + _br) > 0) == _yr[_te]).mean()
    _tr_r = (((_A[_tr] @ _wr + _br) > 0) == _yr[_tr]).mean()
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _proj = _A[_te] @ _dir
    _ax.hist(_proj[_y[_te] == 1], 40, alpha=0.6, label="true"); _ax.hist(_proj[_y[_te] == 0], 40, alpha=0.6, label="false")
    _ax.set_title("activations projected on the planted direction"); _ax.legend()
    mo.vstack([_fig, mo.md(f"""
    | | train | test |
    |---|---|---|
    | probe on real labels | | **{_acc:.1%}** |
    | control: random labels | {_tr_r:.1%} | {_acc_r:.1%} |

    cos(probe, planted direction) = **{_w @ _dir / (np.linalg.norm(_w) + 1e-9):.2f}**. With few examples in high dimension
    the control probe fits random labels on the training set: that's memorisation, and why you report held-out accuracy.
    """)])
    return


if __name__ == "__main__":
    app.run()
