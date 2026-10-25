# sq-meta: {"title": "GRPO Group Advantages", "emoji": "👥", "group": "Reinforcement Learning", "kind": "interactive", "desc": "Group size, solve rate, and how often a group carries no learning signal.", "topic": "GRPO"}
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
    gr_G = mo.ui.slider(2, 32, step=1, value=8, label="group size G")
    gr_p = mo.ui.slider(0.01, 0.99, step=0.01, value=0.2, label="prompt solve rate p")
    mo.vstack([mo.md("# 👥 GRPO: advantages within a group\nA = (r - mean) / std over G answers to the same prompt."),
               mo.hstack([gr_G, gr_p])])
    return (gr_G, gr_p,)


@app.cell
def _(gr_G, gr_p, mo, np, plt):
    _rng = np.random.default_rng(3)
    _r = (_rng.random(gr_G.value) < gr_p.value).astype(float)
    _adv = (_r - _r.mean()) / (_r.std() + 1e-6)
    _Gs = np.arange(2, 33)
    _dead = gr_p.value ** _Gs + (1 - gr_p.value) ** _Gs
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.2))
    _a1.bar(range(gr_G.value), _adv, color=["#16a34a" if _x > 0 else "#dc2626" for _x in _adv])
    _a1.set_title("one sampled group: advantage per answer"); _a1.set_xlabel("answer")
    _a2.plot(_Gs, _dead, "o-", ms=3); _a2.axvline(gr_G.value, color="red", ls=":")
    _a2.set_title("P(all answers same) = no gradient"); _a2.set_xlabel("G"); _a2.set_ylim(0, 1)
    mo.vstack([_fig, mo.md(f"At G={gr_G.value}, p={gr_p.value:.2f}: **{_dead[gr_G.value - 2]:.0%}** of prompts give zero signal. "
                           "That's why DAPO filters out all-correct and all-wrong groups.")])
    return


if __name__ == "__main__":
    app.run()
