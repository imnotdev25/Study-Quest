# sq-meta: {"title": "lora_qlora", "emoji": "📓", "group": "Transformers & LLMs", "kind": "code", "desc": "Toolkit ch. 7: LoRA from scratch, and the NF4 quantisation inside QLoRA", "topic": ""}
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
    mo.md("# lora_qlora\n\nToolkit ch. 7: LoRA from scratch, and the NF4 quantisation inside QLoRA\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Toolkit ch. 7: LoRA from scratch, and the NF4 quantisation inside QLoRA."""
    import math
    import torch

    torch.manual_seed(0)


    class LoRALinear(torch.nn.Module):
        """y = x W^T + (alpha/r) x A^T B^T, with W frozen, A random, B zero (so it starts as W)."""

        def __init__(self, base: torch.nn.Linear, r=8, alpha=16):
            super().__init__()
            self.base = base.requires_grad_(False)
            self.A = torch.nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
            self.B = torch.nn.Parameter(torch.zeros(base.out_features, r))
            self.scale = alpha / r

        def forward(self, x):
            return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T

        def merge(self):
            return self.base.weight + self.scale * self.B @ self.A


    d = 256
    base = torch.nn.Linear(d, d, bias=False)
    delta = 0.1 * torch.randn(d, 4) @ torch.randn(4, d) / 2          # the "task" is a rank-4 change
    target = base.weight.detach() + delta
    layer = LoRALinear(base, r=8)
    X = torch.randn(4096, d)
    print(f"[lora] starts equal to base: {torch.allclose(layer(X[:4]), base(X[:4]))}")
    print(f"[lora] trainable {sum(p.numel() for p in layer.parameters() if p.requires_grad):,}"
          f" vs full {d * d:,}")
    opt = torch.optim.Adam([layer.A, layer.B], lr=3e-3)
    for step in range(601):
        idx = torch.randint(0, len(X), (256,))
        loss = ((layer(X[idx]) - X[idx] @ target.T) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0:
            print(f"[lora] step {step:3d}  loss {loss.item():.5f}")
    with torch.no_grad():
        print(f"[lora] merged weight error {(layer.merge() - target).norm() / delta.norm():.3f} (relative to ||delta||)")


    # NF4: 16 levels at quantiles of N(0,1), rescaled to [-1, 1], with an exact zero.
    def nf4_levels(offset=0.9677083):
        icdf = lambda p: math.sqrt(2) * torch.erfinv(2 * p - 1)
        pos = icdf(torch.linspace(offset, 0.5, 9)[:-1])                # 8 positive levels
        neg = -icdf(torch.linspace(offset, 0.5, 8)[:-1])               # 7 negative levels
        v = torch.cat([pos, torch.zeros(1), neg]).sort().values
        return v / v.abs().max()


    def quantize(w, levels, block=64):
        wb = w.reshape(-1, block)
        absmax = wb.abs().max(1, keepdim=True).values                  # one scale per block
        codes = ((wb / absmax)[..., None] - levels).abs().argmin(-1)   # 4-bit index per weight
        return (levels[codes] * absmax).reshape(w.shape)


    NF4 = nf4_levels()
    INT4 = torch.linspace(-1, 1, 16)
    print(f"[nf4] levels {[round(v, 3) for v in NF4.tolist()]}")
    W = torch.randn(4096, 4096) * 0.02
    for name, lv in (("int4 uniform", INT4), ("nf4", NF4)):
        err = (quantize(W, lv) - W).pow(2).mean() / W.pow(2).mean()
        print(f"[nf4] {name:12s} relative MSE {err:.4f}")

    # Memory: 4 bits per weight + fp32 absmax per 64 weights, or 8-bit absmax with one fp32 per 256 blocks.
    plain = 4 + 32 / 64
    double = 4 + 8 / 64 + 32 / (64 * 256)
    print(f"[nf4] bits/param: single quant {plain:.3f}, double quant {double:.3f}")
    for name, params in (("8B", 8e9), ("70B", 70e9)):
        print(f"[mem] {name}: bf16 weights {params * 2 / 2**30:6.1f} GiB  nf4+dq {params * double / 8 / 2**30:5.1f} GiB")
    return


if __name__ == "__main__":
    app.run()
