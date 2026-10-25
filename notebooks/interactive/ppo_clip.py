# sq-meta: {"title": "PPO Clipping", "emoji": "✂️", "group": "Reinforcement Learning", "kind": "interactive", "desc": "Drag epsilon and see where PPO's gradient switches off.", "topic": "PPO"}
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
    clip = mo.ui.slider(0.05, 0.5, step=0.01, value=0.2, label="clip epsilon")
    adv = mo.ui.dropdown({"A > 0 (good action)": 1.0, "A < 0 (bad action)": -1.0}, value="A > 0 (good action)", label="advantage")
    mo.vstack([mo.md("# ✂️ The PPO clipped objective"), mo.hstack([clip, adv])])
    return (adv, clip,)


@app.cell
def _(adv, clip, mo, np, plt):
    _r = np.linspace(0.3, 2.0, 400); _A = adv.value; _e = clip.value
    _obj = np.minimum(_r * _A, np.clip(_r, 1 - _e, 1 + _e) * _A)
    _grad = np.gradient(_obj, _r)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(9, 3))
    _a1.plot(_r, _r * _A, "--", color="gray", label="r A (unclipped)"); _a1.plot(_r, _obj, lw=2.5, label="PPO objective")
    _a1.axvspan(1 - _e, 1 + _e, color="#e0e7ff"); _a1.set_xlabel("ratio r = pi_new / pi_old"); _a1.legend(fontsize=8)
    _a2.plot(_r, _grad, color="#dc2626"); _a2.set_title("d objective / d r"); _a2.set_xlabel("r")
    mo.vstack([_fig, mo.md("Shaded band: the trust region. Outside it on the side that would *help*, the gradient is 0. "
                           "On the side that would hurt (A < 0, r > 1+eps), it stays on: PPO keeps fixing mistakes.")])
    return


if __name__ == "__main__":
    app.run()
