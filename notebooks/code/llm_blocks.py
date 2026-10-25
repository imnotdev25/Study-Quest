# sq-meta: {"title": "llm_blocks", "emoji": "📓", "group": "Transformers & LLMs", "kind": "code", "desc": "Toolkit ch. 6: the distinctive blocks of Llama 4, DeepSeek-V3 and Qwen3, at toy size", "topic": ""}
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
    mo.md("# llm_blocks\n\nToolkit ch. 6: the distinctive blocks of Llama 4, DeepSeek-V3 and Qwen3, at toy size\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Toolkit ch. 6: the distinctive blocks of Llama 4, DeepSeek-V3 and Qwen3, at toy size."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(0)


    class RMSNorm(nn.Module):
        def __init__(self, d, eps=1e-6):
            super().__init__()
            self.w, self.eps = nn.Parameter(torch.ones(d)), eps

        def forward(self, x):
            return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


    def rope(x, pos, base=10000.0):
        """x: (..., T, d) with d even. Rotate pairs (x_2i, x_2i+1) by angle pos * base^(-2i/d)."""
        d = x.shape[-1]
        ang = pos[:, None].float() * base ** (-torch.arange(0, d, 2).float() / d)
        cos, sin = ang.cos(), ang.sin()
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


    class GQA(nn.Module):
        """Grouped-query attention with optional QK-norm (Qwen3) and optional NoPE (Llama 4)."""

        def __init__(self, d, n_heads, n_kv, qk_norm=False, use_rope=True):
            super().__init__()
            self.h, self.kv, self.hd, self.use_rope = n_heads, n_kv, d // n_heads, use_rope
            self.q = nn.Linear(d, n_heads * self.hd, bias=False)
            self.k = nn.Linear(d, n_kv * self.hd, bias=False)
            self.v = nn.Linear(d, n_kv * self.hd, bias=False)
            self.o = nn.Linear(n_heads * self.hd, d, bias=False)
            self.qn = RMSNorm(self.hd) if qk_norm else nn.Identity()
            self.kn = RMSNorm(self.hd) if qk_norm else nn.Identity()

        def forward(self, x, cache=None, start=0):
            B, T, _ = x.shape
            q = self.qn(self.q(x).view(B, T, self.h, self.hd)).transpose(1, 2)
            k = self.kn(self.k(x).view(B, T, self.kv, self.hd)).transpose(1, 2)
            v = self.v(x).view(B, T, self.kv, self.hd).transpose(1, 2)
            if self.use_rope:
                pos = torch.arange(start, start + T)
                q, k = rope(q, pos), rope(k, pos)
            if cache is not None:
                if "k" in cache:
                    k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
                cache["k"], cache["v"] = k, v
            k = k.repeat_interleave(self.h // self.kv, 1)
            v = v.repeat_interleave(self.h // self.kv, 1)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=cache is None or T > 1)
            return self.o(y.transpose(1, 2).reshape(B, T, -1))


    class SwiGLU(nn.Module):
        def __init__(self, d, f):
            super().__init__()
            self.g, self.u, self.dn = (nn.Linear(d, f, bias=False), nn.Linear(d, f, bias=False),
                                       nn.Linear(f, d, bias=False))

        def forward(self, x):
            return self.dn(F.silu(self.g(x)) * self.u(x))


    class MoE(nn.Module):
        """Shared expert + routed experts. DeepSeek-V3 style: sigmoid scores, top-k chosen on
        score + bias, the bias is nudged toward balance instead of using an auxiliary loss."""

        def __init__(self, d, f, n_exp, top_k):
            super().__init__()
            self.shared = SwiGLU(d, f)
            self.experts = nn.ModuleList(SwiGLU(d, f) for _ in range(n_exp))
            self.router = nn.Linear(d, n_exp, bias=False)
            self.register_buffer("bias", torch.zeros(n_exp))
            self.k = top_k

        def forward(self, x):
            flat = x.reshape(-1, x.shape[-1])
            s = torch.sigmoid(self.router(flat))
            idx = (s + self.bias).topk(self.k, -1).indices            # bias only affects *which* experts
            w = s.gather(1, idx)
            w = w / w.sum(-1, keepdim=True)                           # gate weights come from s alone
            out = self.shared(flat)
            for e, expert in enumerate(self.experts):
                rows, slot = (idx == e).nonzero(as_tuple=True)
                if len(rows):
                    out = out.index_add(0, rows, w[rows, slot, None] * expert(flat[rows]))
            self.load = torch.bincount(idx.flatten(), minlength=len(self.experts)).float()
            return out.view_as(x)

        @torch.no_grad()
        def update_bias(self, gamma=0.01):
            self.bias += gamma * torch.sign(self.load.mean() - self.load)


    class MLA(nn.Module):
        """DeepSeek multi-head latent attention: cache a small latent c instead of full K and V."""

        def __init__(self, d, n_heads, d_latent):
            super().__init__()
            self.h, self.hd = n_heads, d // n_heads
            self.q = nn.Linear(d, d, bias=False)
            self.down = nn.Linear(d, d_latent, bias=False)            # c = W_down x   (this is cached)
            self.up_k = nn.Linear(d_latent, d, bias=False)
            self.up_v = nn.Linear(d_latent, d, bias=False)
            self.o = nn.Linear(d, d, bias=False)

        def forward(self, x):
            B, T, D = x.shape
            c = self.down(x)
            q = self.q(x).view(B, T, self.h, self.hd).transpose(1, 2)
            k = self.up_k(c).view(B, T, self.h, self.hd).transpose(1, 2)
            v = self.up_v(c).view(B, T, self.h, self.hd).transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.o(y.transpose(1, 2).reshape(B, T, D)), c


    d = 64
    x = torch.randn(2, 16, d)

    # 1. GQA cache: decoding token by token equals the full causal pass.
    att = GQA(d, n_heads=8, n_kv=2, qk_norm=True)
    full = att(x)
    cache, steps = {}, []
    for t in range(16):
        steps.append(att(x[:, t:t + 1], cache=cache, start=t))
    print(f"[gqa] cached decode == full pass: {torch.allclose(full, torch.cat(steps, 1), atol=1e-5)}"
          f"  cache k {tuple(cache['k'].shape)}")

    # 2. QK-norm bounds the attention logits when weights grow.
    for qk in (False, True):
        a = GQA(d, 8, 2, qk_norm=qk)
        with torch.no_grad():
            a.q.weight.mul_(20); a.k.weight.mul_(20)
            B, T = x.shape[:2]
            q = a.qn(a.q(x).view(B, T, 8, 8)); k = a.kn(a.k(x).view(B, T, 2, 8))
            logit = (q[:, :, :2] * k).sum(-1).abs().max() / 8**0.5
        print(f"[qk-norm] {'on ' if qk else 'off'}  max |logit| with 20x weights: {logit:.1f}")

    # 3. RoPE: the score depends only on the distance between positions.
    qv, kv = torch.randn(1, 1, 8), torch.randn(1, 1, 8)
    s = lambda i, j: (rope(qv, torch.tensor([i])) * rope(kv, torch.tensor([j]))).sum().item()
    print(f"[rope] score(5,2)={s(5, 2):.4f}  score(105,102)={s(105, 102):.4f}")

    # 4. MLA: latent cache is far smaller than K+V; the K up-projection can be absorbed into the query.
    mla = MLA(d, n_heads=8, d_latent=16)
    y, c = mla(x)
    print(f"[mla] cache per token: latent {c.shape[-1]} floats vs K+V {2 * d} floats")
    q = mla.q(x[:, -1:])
    Wk = mla.up_k.weight                                              # (d, d_latent)
    score_naive = q @ mla.up_k(c).transpose(1, 2)
    score_absorbed = (q @ Wk) @ c.transpose(1, 2)                     # never materialise K
    print(f"[mla] absorbed scores == naive scores: {torch.allclose(score_naive, score_absorbed, atol=1e-4)}")

    # 5. Aux-loss-free balancing: a skewed router evens out as the bias updates.
    moe = MoE(d, 32, n_exp=8, top_k=2)
    with torch.no_grad():
        moe.router.weight[0] += 0.5 * x.reshape(-1, d).mean(0) / x.reshape(-1, d).mean(0).norm()
        moe.router.weight[1] += 0.5 * x.reshape(-1, d).mean(0) / x.reshape(-1, d).mean(0).norm()
    for it in range(301):
        moe(x)
        if it in (0, 100, 300):
            print(f"[moe] iter {it:3d}  tokens per expert {moe.load.int().tolist()}")
        moe.update_bias(0.01)

    # 6. Llama 4 interleaves NoPE layers (no position encoding) between RoPE layers.
    pattern = ["NoPE" if (i + 1) % 4 == 0 else "RoPE" for i in range(8)]
    print(f"[llama4] layer pattern: {' '.join(pattern)}")

    # 7. KV cache per token (bf16) at published configurations.
    configs = {"Llama-3-70B   GQA 64q/8kv  hd128 L80": 2 * 8 * 128 * 80,
               "Qwen3-32B     GQA 64q/8kv  hd128 L64": 2 * 8 * 128 * 64,
               "DeepSeek-V3   MLA 512+64       L61": (512 + 64) * 61,
               "MHA baseline  64 heads hd128   L80": 2 * 64 * 128 * 80}
    for k, n in configs.items():
        print(f"[kv] {k}: {n * 2 / 1024:7.1f} KiB/token")
    return


if __name__ == "__main__":
    app.run()
