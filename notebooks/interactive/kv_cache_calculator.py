# sq-meta: {"title": "KV Cache Calculator", "emoji": "🧮", "group": "Inference", "kind": "interactive", "desc": "How many users fit on one GPU? Weights, KV cache and precision.", "topic": "KV cache"}
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
    preset = mo.ui.dropdown({"Llama-3.1-8B": (8.0, 32, 8, 128), "Qwen3-32B": (32.8, 64, 8, 128), "Llama-3.1-70B": (70.6, 80, 8, 128)},
                            value="Llama-3.1-8B", label="model")
    ctx = mo.ui.slider(1024, 131072, step=1024, value=8192, label="context per request")
    gpu = mo.ui.slider(24, 192, step=8, value=80, label="GPU memory (GB)")
    wdt = mo.ui.dropdown({"bf16": 2, "fp8": 1, "int4": 0.53}, value="bf16", label="weights")
    kdt = mo.ui.dropdown({"bf16": 2, "fp8": 1}, value="bf16", label="KV cache")
    mo.vstack([mo.md("# 🧮 KV cache calculator"), mo.hstack([preset, gpu]), mo.hstack([ctx, wdt, kdt])])
    return (ctx, gpu, kdt, preset, wdt,)


@app.cell
def _(ctx, gpu, kdt, mo, plt, preset, wdt):
    _P, _L, _kv, _hd = preset.value
    _w = _P * 1e9 * wdt.value / 1e9
    _per_tok = 2 * _L * _kv * _hd * kdt.value
    _free = 0.9 * gpu.value - _w - 2
    _users = max(0, int(_free * 1e9 / (_per_tok * ctx.value)))
    _fig, _ax = plt.subplots(figsize=(7, 1.4))
    _ax.barh([0], [_w], color="#6366f1", label="weights"); _ax.barh([0], [max(_free, 0)], left=[_w], color="#a5b4fc", label="KV cache")
    _ax.set_xlim(0, gpu.value); _ax.set_yticks([]); _ax.set_xlabel("GB"); _ax.legend(loc="lower right", fontsize=8)
    _txt = (f"KV per token **{_per_tok / 1024:.0f} KiB**, per request **{_per_tok * ctx.value / 2**30:.2f} GiB**. "
            + (f"Room for **{_users} concurrent requests** at this context." if _free > 0 else "**Weights alone don't fit.** Try fp8 or int4."))
    mo.vstack([_fig, mo.md(_txt)])
    return


if __name__ == "__main__":
    app.run()
