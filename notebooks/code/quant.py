import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

@app.cell
def _():
    import marimo as mo
    import sys as _sys
    _sys.path.insert(0, '/Users/bhavik/Developer/ClaudeProjects/RL/llm_book/code')
    return (mo,)


@app.cell
def _(mo):
    mo.md("# quant\n\nInference ch. 'Quantization and KV Compression': int8/int4, outliers, AWQ scaling, FP8, KV quantisation\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Inference ch. 'Quantization and KV Compression': int8/int4, outliers, AWQ scaling, FP8, KV quantisation."""
    import torch

    torch.manual_seed(0)


    def q_sym(x, bits, dim=None, group=None):
        """Symmetric absmax quantise-dequantise. dim: reduce over this dim; group: group size on last dim."""
        qmax = 2 ** (bits - 1) - 1
        if group:
            shp = x.shape
            x = x.reshape(*shp[:-1], -1, group)
            s = x.abs().amax(-1, keepdim=True) / qmax
            return ((x / s).round().clamp(-qmax, qmax) * s).reshape(shp)
        s = (x.abs().amax() if dim is None else x.abs().amax(dim, keepdim=True)) / qmax
        return (x / s).round().clamp(-qmax, qmax) * s


    rel = lambda a, b: ((a - b).norm() / b.norm()).item()

    W = torch.randn(1024, 1024) * 0.02
    W[:4] *= 30                                                    # a few output rows with huge weights
    print(f"[int8] weight error per-tensor {rel(q_sym(W, 8), W):.4f}  per-channel {rel(q_sym(W, 8, dim=1), W):.4f}")

    # Activations: a few input channels carry outliers (seen in LLMs above ~6B params).
    X = torch.randn(512, 1024)
    X[:, :8] *= 40
    W = torch.randn(1024, 1024) * 0.02
    Y = X @ W.T
    print(f"[int8] activation per-tensor quant, output error {rel(q_sym(X, 8) @ W.T, Y):.4f} "
          f"(outliers squeeze every other channel into a few levels)")

    # W4 group quantisation, and AWQ: scale salient input channels up before quantising, fold 1/s into X.
    base = rel(X @ q_sym(W, 4, group=128).T, Y)
    best = (0.0, base)
    for a in (0.25, 0.5, 0.75, 1.0):
        s = X.abs().mean(0).pow(a)
        s = s / s.mean()
        err = rel((X / s) @ q_sym(W * s, 4, group=128).T, Y)
        if err < best[1]:
            best = (a, err)
    print(f"[awq] int4 g128 output error: plain {base:.4f}  AWQ (alpha={best[0]}) {best[1]:.4f}")

    # FP8 e4m3 keeps relative precision across magnitudes; int8 per-tensor is uniform.
    if hasattr(torch, "float8_e4m3fn"):
        Xs = X / X.abs().amax() * 448                              # scale to e4m3 max
        x8 = Xs.to(torch.float8_e4m3fn).to(torch.float32) * X.abs().amax() / 448
        print(f"[fp8] activation quant, output error: e4m3 {rel(x8 @ W.T, Y):.4f}  vs int8 {rel(q_sym(X, 8) @ W.T, Y):.4f}")

    # KV cache in int8 (per token, per head) and its effect on attention output.
    T, hd = 2048, 128
    q, k, v = torch.randn(1, hd), torch.randn(T, hd), torch.randn(T, hd)
    att = lambda k, v: ((q @ k.T) / hd**0.5).softmax(-1) @ v
    print(f"[kv] attention output error with int8 KV {rel(att(q_sym(k, 8, dim=1), q_sym(v, 8, dim=1)), att(k, v)):.4f}"
          f"  int4 KV {rel(att(q_sym(k, 4, dim=1), q_sym(v, 4, dim=1)), att(k, v)):.4f}")

    # Capacity on one 80 GB GPU for Llama-3.1-8B (128 KiB/token of KV at bf16).
    print("[cap] weights  KV      weights GiB  KV tokens (90% of 80 GB, 2 GiB act.)")
    for wname, wb in (("bf16", 2), ("fp8", 1), ("int4", 0.53)):
        for kname, kb in (("bf16", 2), ("fp8", 1)):
            w = 8.0e9 * wb
            tokens = (0.9 * 80e9 - w - 2 * 2**30) / (32 * 8 * 128 * 2 * kb)
            print(f"[cap] {wname:5s}    {kname:5s}  {w / 2**30:9.1f}    {tokens:10,.0f}")
    return


if __name__ == "__main__":
    app.run()
