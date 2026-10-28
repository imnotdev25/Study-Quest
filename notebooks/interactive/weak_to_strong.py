# sq-meta: {"title": "Weak-to-Strong Generalisation", "emoji": "🐣", "group": "AI Safety", "kind": "interactive", "desc": "A strong student learns from a weak teacher's labels. Measure performance gap recovered.", "topic": "weak-to-strong"}
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
    ws_weak = mo.ui.slider(2, 30, step=1, value=6, label="features the weak teacher sees")
    ws_sal = mo.ui.slider(0.0, 3.0, step=0.1, value=2.0, label="how salient the concept is in the student's representation")
    ws_conf = mo.ui.slider(0.0, 0.9, step=0.05, value=0.5, label="confidence loss weight (trust yourself over the teacher)")
    mo.vstack([mo.md("# 🐣 Weak-to-strong: PGR = (student - weak) / (ceiling - weak)\nThe student has its own representation "
                     "(think: pre-trained features). If the true concept is already easy to read from it, weak labels are "
                     "enough to point at it."), ws_weak, ws_sal, ws_conf])
    return (ws_conf, ws_sal, ws_weak,)


@app.cell
def _(mo, np, plt, ws_conf, ws_sal, ws_weak):
    _rng = np.random.default_rng(0)
    _n, _d, _m = 9000, 40, 20
    _X = _rng.normal(size=(_n, _d)); _wt = _rng.normal(size=_d)
    _y = (_X @ _wt > 0).astype(float)
    _truth = _X @ _wt / np.linalg.norm(_wt)
    _Z = np.c_[ws_sal.value * _truth + 0.3 * _rng.normal(size=_n), _rng.normal(size=(_n, _m - 1)) + 0.4 * _X[:, :_m - 1]]
    def _fit(X, y, conf=0.0, steps=400):
        w = np.zeros(X.shape[1])
        for _ in range(steps):
            p = 1 / (1 + np.exp(-X @ w))
            target = (1 - conf) * y + conf * (p > 0.5)          # blend weak labels with own hardened predictions
            w -= 0.5 * X.T @ (p - target) / len(y)
        return w
    _acc = lambda w, X, y: ((X @ w > 0) == y).mean()
    _a, _b, _c = slice(0, 3000), slice(3000, 6000), slice(6000, None)
    _k = ws_weak.value
    _wk = _fit(_X[_a, :_k], _y[_a])
    _labels = (_X[_b, :_k] @ _wk > 0).astype(float)
    _st = _fit(_Z[_b], _labels, ws_conf.value)
    _ce = _fit(_Z[_b], _y[_b])
    _aw, _as, _ac = _acc(_wk, _X[_c, :_k], _y[_c]), _acc(_st, _Z[_c], _y[_c]), _acc(_ce, _Z[_c], _y[_c])
    _pgr = (_as - _aw) / max(1e-9, _ac - _aw)
    _fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.bar(["weak teacher", "strong on weak labels", "strong ceiling"], [_aw, _as, _ac], color=["#f59e0b", "#6366f1", "#16a34a"])
    _ax.set_ylim(0.5, 1); _ax.set_ylabel("test accuracy")
    mo.vstack([_fig, mo.md(f"**PGR = {_pgr:.2f}.** Set salience to 0 and the student can only copy the teacher (PGR ≈ 0). "
                           "With a salient concept, the confidence loss lets it overrule labels that disagree with it.")])
    return


if __name__ == "__main__":
    app.run()
