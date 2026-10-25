# sq-meta: {"title": "Softmax and Temperature", "emoji": "🌡️", "group": "Transformers & LLMs", "kind": "interactive", "desc": "See how temperature reshapes next-token probabilities and entropy.", "topic": "softmax"}
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
    logits_text = mo.ui.text(value="2.0, 1.0, 0.1, -1.0, 3.0", label="logits")
    temp = mo.ui.slider(0.05, 5.0, step=0.05, value=1.0, label="temperature")
    top_p = mo.ui.slider(0.05, 1.0, step=0.05, value=1.0, label="top-p")
    mo.vstack([mo.md("# 🌡️ Softmax, temperature and top-p"), logits_text, mo.hstack([temp, top_p])])
    return (logits_text, temp, top_p,)


@app.cell
def _(logits_text, mo, np, plt, temp, top_p):
    _z = np.array([float(x) for x in logits_text.value.split(",") if x.strip()])
    _p = np.exp((_z - _z.max()) / temp.value); _p /= _p.sum()
    _order = np.argsort(-_p); _keep = _order[:np.searchsorted(np.cumsum(_p[_order]), top_p.value) + 1]
    _q = np.zeros_like(_p); _q[_keep] = _p[_keep]; _q /= _q.sum()
    _H = -(_q[_q > 0] * np.log(_q[_q > 0])).sum()
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.bar(range(len(_z)), _p, color="#c7d2fe", label="softmax(z / T)")
    _ax.bar(range(len(_z)), _q, color="#6366f1", width=0.5, label="after top-p")
    _ax.set_xticks(range(len(_z)), [f"tok{i}\nz={v:g}" for i, v in enumerate(_z)]); _ax.set_ylim(0, 1); _ax.legend()
    mo.vstack([_fig, mo.md(f"Entropy **{_H:.3f} nats** (max {np.log(len(_z)):.3f}). Low T: argmax. High T: uniform.")])
    return


if __name__ == "__main__":
    app.run()
