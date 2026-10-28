# sq-meta: {"title": "RL Data Pipeline: Difficulty Filtering", "emoji": "🧹", "group": "RL Environments", "kind": "interactive", "desc": "Zero-variance groups, pass-rate filters, and how much of your batch actually trains.", "topic": "difficulty"}
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
    df_a = mo.ui.slider(0.1, 3.0, step=0.1, value=0.6, label="solve-rate distribution: Beta(a, a)")
    df_G = mo.ui.slider(2, 32, step=1, value=8, label="group size G")
    df_band = mo.ui.range_slider(0.0, 1.0, step=0.05, value=[0.1, 0.9], label="keep prompts with estimated pass rate in")
    df_k = mo.ui.slider(2, 64, step=2, value=8, label="samples used to estimate pass rate")
    mo.vstack([mo.md("# 🧹 Difficulty filtering for GRPO-style training"), df_a, df_G, df_band, df_k])
    return (df_G, df_a, df_band, df_k,)


@app.cell
def _(df_G, df_a, df_band, df_k, mo, np, plt):
    _rng = np.random.default_rng(0)
    _p = _rng.beta(df_a.value, df_a.value, 20_000)
    _est = _rng.binomial(df_k.value, _p) / df_k.value
    _lo, _hi = df_band.value
    _keep = (_est >= _lo) & (_est <= _hi)
    _dead = lambda p: p ** df_G.value + (1 - p) ** df_G.value
    _signal = lambda p: p * (1 - p)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.2))
    _a1.hist(_p, 40, alpha=0.5, label="all prompts"); _a1.hist(_p[_keep], 40, alpha=0.7, label="kept")
    _a1.set_xlabel("true solve rate"); _a1.legend()
    _a2.bar(["all", "kept"], [_dead(_p).mean(), _dead(_p[_keep]).mean()], color=["#94a3b8", "#6366f1"])
    _a2.set_title("share of groups with zero gradient"); _a2.set_ylim(0, 1)
    mo.vstack([_fig, mo.md(f"Kept **{_keep.mean():.0%}** of prompts. Mean reward variance p(1-p): all {_signal(_p).mean():.3f} → kept "
                           f"{_signal(_p[_keep]).mean():.3f}. Fewer estimation samples (k) misclassify prompts near the band edges.")])
    return


if __name__ == "__main__":
    app.run()
