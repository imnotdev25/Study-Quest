# sq-meta: {"title": "Reward Hacking Lab", "emoji": "🎣", "group": "AI Safety", "kind": "interactive", "desc": "Push harder on a proxy reward and watch the true reward peak, then fall.", "topic": "Reward hacking"}
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
    rh_weight = mo.ui.slider(0.0, 1.5, step=0.05, value=0.6, label="how much the proxy over-rewards the exploit")
    rh_cost = mo.ui.slider(0.0, 1.0, step=0.05, value=0.3, label="how bad the exploit really is")
    rh_tail = mo.ui.slider(2, 30, step=1, value=3, label="exploit tail heaviness (t dof, lower = heavier)")
    mo.vstack([mo.md("# 🎣 Goodhart in one plot\nEach behaviour has true quality plus an *exploit* component the proxy likes. "
                     "Optimisation pressure = how far into the proxy's tail we select (best-of-n)."), rh_weight, rh_cost, rh_tail])
    return (rh_cost, rh_tail, rh_weight,)


@app.cell
def _(mo, np, plt, rh_cost, rh_tail, rh_weight):
    _rng = np.random.default_rng(0)
    _n = 200_000
    _q = _rng.normal(size=_n); _x = _rng.standard_t(df=rh_tail.value, size=_n)
    _proxy = _q + rh_weight.value * _x
    _true = _q - rh_cost.value * np.abs(_x)
    _order = np.argsort(-_proxy)
    _ns = np.unique(np.logspace(0, np.log10(_n / 2), 40).astype(int))
    _kl = np.log(_ns) - (_ns - 1) / _ns              # KL of best-of-n vs base policy
    _pk = [_proxy[_order[: _n // k]].mean() for k in _ns]
    _tk = [_true[_order[: _n // k]].mean() for k in _ns]
    _fig, _ax = plt.subplots(figsize=(7, 3.4))
    _ax.plot(_kl, _pk, label="proxy reward"); _ax.plot(_kl, _tk, label="true reward")
    _peak = int(np.argmax(_tk)); _ax.axvline(_kl[_peak], color="gray", ls=":")
    _ax.set_xlabel("KL from base policy (best-of-n)"); _ax.legend(); _ax.grid(alpha=0.3)
    mo.vstack([_fig, mo.md(f"True reward peaks at best-of-**{_ns[_peak]}** and declines after. "
                           "Set the exploit cost to 0 or the weight to 0 and the curves stop diverging.")])
    return


if __name__ == "__main__":
    app.run()
