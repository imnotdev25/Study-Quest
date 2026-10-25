# sq-meta: {"title": "Low-Rank Approximation (SVD)", "emoji": "🧩", "group": "Math & Optimisation", "kind": "interactive", "desc": "Keep k singular values of an image: the idea behind LoRA and compression.", "topic": "SVD"}
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
    svd_k = mo.ui.slider(1, 60, step=1, value=5, label="rank k")
    svd_noise = mo.ui.slider(0.0, 0.5, step=0.05, value=0.1, label="noise")
    mo.vstack([mo.md("# 🧩 Best rank-k approximation"), mo.hstack([svd_k, svd_noise])])
    return (svd_k, svd_noise,)


@app.cell
def _(mo, np, plt, svd_k, svd_noise):
    _g = np.linspace(-1, 1, 80)
    _X, _Y = np.meshgrid(_g, _g)
    _img = (np.exp(-((_X - 0.3) ** 2 + _Y ** 2) * 6) + 0.6 * ((_X ** 2 + (_Y + 0.4) ** 2) < 0.12) + 0.4 * np.sin(6 * _X) * (_Y > 0.5))
    _img = _img + svd_noise.value * np.random.default_rng(0).normal(size=_img.shape)
    _U, _S, _Vt = np.linalg.svd(_img)
    _k = svd_k.value
    _approx = (_U[:, :_k] * _S[:_k]) @ _Vt[:_k]
    _err = np.linalg.norm(_img - _approx) / np.linalg.norm(_img)
    _fig, _axs = plt.subplots(1, 3, figsize=(10, 3.2))
    _axs[0].imshow(_img, cmap="magma"); _axs[0].set_title("original (80×80)")
    _axs[1].imshow(_approx, cmap="magma"); _axs[1].set_title(f"rank {_k}")
    _axs[2].semilogy(_S, "o-", ms=3); _axs[2].axvline(_k - 0.5, color="red"); _axs[2].set_title("singular values")
    for _a in _axs[:2]:
        _a.set_xticks([]); _a.set_yticks([])
    mo.vstack([_fig, mo.md(f"Relative error **{_err:.3f}** using {_k * 161} numbers instead of 6400 "
                           f"({_k * 161 / 6400:.0%}). LoRA bets that a fine-tune's weight *update* has a spectrum like this.")])
    return


if __name__ == "__main__":
    app.run()
