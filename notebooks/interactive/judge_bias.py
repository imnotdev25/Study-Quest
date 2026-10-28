# sq-meta: {"title": "LLM Judge Bias", "emoji": "⚖️", "group": "AI Safety", "kind": "interactive", "desc": "Position and length bias in a simulated judge, and how swapping and length control fix it.", "topic": "judge"}
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
    jb_pos = mo.ui.slider(0.0, 0.4, step=0.01, value=0.15, label="position bias (prob. of picking the first answer regardless)")
    jb_len = mo.ui.slider(0.0, 2.0, step=0.05, value=0.6, label="length bias (logit per +100 tokens)")
    jb_skill = mo.ui.slider(0.5, 4.0, step=0.1, value=2.0, label="judge skill (logit per unit quality gap)")
    mo.vstack([mo.md("# ⚖️ Is the judge measuring quality?"), jb_pos, jb_len, jb_skill])
    return (jb_len, jb_pos, jb_skill,)


@app.cell
def _(jb_len, jb_pos, jb_skill, mo, np, plt):
    _rng = np.random.default_rng(0)
    _n = 3000
    _qa, _qb = _rng.normal(size=_n), _rng.normal(size=_n)
    _la, _lb = _rng.integers(50, 600, _n), _rng.integers(50, 600, _n)
    _human = (_qa > _qb).astype(int)
    def _judge(q1, q2, l1, l2):
        logit = jb_skill.value * (q1 - q2) + jb_len.value * (l1 - l2) / 100
        pick1 = _rng.random(_n) < 1 / (1 + np.exp(-logit))
        return np.where(_rng.random(_n) < jb_pos.value, True, pick1)
    _ab = _judge(_qa, _qb, _la, _lb).astype(int)            # 1 = A wins
    _ba = 1 - _judge(_qb, _qa, _lb, _la).astype(int)        # same pair, swapped, mapped back to "A wins"
    _cons = _ab == _ba
    def _kappa(a, b):
        po = (a == b).mean(); pe = sum((a == k).mean() * (b == k).mean() for k in (0, 1)); return (po - pe) / (1 - pe)
    _close = np.abs(_qa - _qb) < 0.3
    _longer = (_la > _lb).astype(int)
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _bins = np.linspace(-550, 550, 12); _mid = (_bins[1:] + _bins[:-1]) / 2
    _rate = [_ab[(_la - _lb >= lo) & (_la - _lb < hi)].mean() for lo, hi in zip(_bins[:-1], _bins[1:])]
    _ax.plot(_mid, _rate, "o-"); _ax.axhline(_human.mean(), color="gray", ls=":")
    _ax.set_xlabel("length(A) - length(B) tokens"); _ax.set_ylabel("judge picks A"); _ax.set_ylim(0, 1)
    mo.vstack([_fig, mo.md(f"""
    - κ with humans, A shown first: **{_kappa(_human, _ab):.2f}** (on close pairs only: {_kappa(_human[_close], _ab[_close]):.2f})
    - verdicts consistent after swapping positions: **{_cons.mean():.0%}**; κ on those: {_kappa(_human[_cons], _ab[_cons]):.2f}
    - judge agrees with "longer wins" {(_ab == _longer).mean():.0%} of the time (humans: {(_human == _longer).mean():.0%})
    """)])
    return


if __name__ == "__main__":
    app.run()
