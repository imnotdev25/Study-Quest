# sq-meta: {"title": "Attention Heatmap", "emoji": "🔦", "group": "Transformers & LLMs", "kind": "interactive", "desc": "Type a sentence and see who attends to whom, with and without a causal mask.", "topic": "attention"}
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
    att_text = mo.ui.text(value="the cat sat on the mat because the cat was tired", label="sentence", full_width=True)
    att_causal = mo.ui.checkbox(value=True, label="causal mask")
    att_sharp = mo.ui.slider(0.2, 4.0, step=0.1, value=1.5, label="sharpness (1 / temperature)")
    mo.vstack([mo.md("# 🔦 Attention heatmap\nEach word gets a fixed random vector, so repeated words match strongly."),
               att_text, mo.hstack([att_causal, att_sharp])])
    return (att_causal, att_sharp, att_text,)


@app.cell
def _(att_causal, att_sharp, att_text, mo, np, plt):
    _words = att_text.value.split()[:24] or ["empty"]
    def _vec(_w):
        return np.random.default_rng(sum(ord(_ch) * 31 ** _i for _i, _ch in enumerate(_w)) % 2**32).normal(size=16)
    _E = np.stack([_vec(_w) for _w in _words])
    _S = _E @ _E.T / 4.0 * att_sharp.value
    if att_causal.value:
        _S = np.where(np.tril(np.ones_like(_S, dtype=bool)), _S, -np.inf)
    _P = np.exp(_S - _S.max(1, keepdims=True)); _P /= _P.sum(1, keepdims=True)
    _fig, _ax = plt.subplots(figsize=(6.5, 5.5))
    _ax.imshow(_P, cmap="Purples", vmin=0, vmax=1)
    _ax.set_xticks(range(len(_words)), _words, rotation=60, ha="right"); _ax.set_yticks(range(len(_words)), _words)
    _ax.set_xlabel("key (attended to)"); _ax.set_ylabel("query")
    mo.vstack([_fig, mo.md("Each row sums to 1. With the mask, row *i* only sees columns ≤ *i*; the first word can only look at itself.")])
    return


if __name__ == "__main__":
    app.run()
