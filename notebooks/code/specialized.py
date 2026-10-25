# sq-meta: {"title": "specialized", "emoji": "📓", "group": "Inference", "kind": "code", "desc": "Inference ch. 'Specialized Workloads': constrained decoding, multi-LoRA batching, batch-invariant numerics", "topic": ""}
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
    mo.md("# specialized\n\nInference ch. 'Specialized Workloads': constrained decoding, multi-LoRA batching, batch-invariant numerics\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Inference ch. 'Specialized Workloads': constrained decoding, multi-LoRA batching, batch-invariant numerics."""
    import torch

    from tiny_gpt import Cfg, TinyGPT

    # 1. Structured output: mask every token the grammar does not allow, then pick from the rest.
    words = ['{"label": "spam"}', '{"label": "ham"}', '{"label": "unsure"}']
    vocab = sorted(set("".join(words)))
    model = TinyGPT(Cfg(vocab=len(vocab)))


    def allowed(prefix):
        return {w[len(prefix)] for w in words if w.startswith(prefix) and len(w) > len(prefix)}


    out = ""
    while allowed(out):
        logits = model.full([vocab.index(ch) for ch in out] or [0])[-1]
        mask = torch.full_like(logits, float("-inf"))
        mask[[vocab.index(ch) for ch in allowed(out)]] = 0
        out += vocab[int((logits + mask).argmax())]
    free = "".join(vocab[int(t)] for t in model.full([0] * 8).argmax(-1)[:12])
    print(f"[grammar] unconstrained: {free!r}")
    print(f"[grammar] constrained:   {out!r}  valid: {out in words}")

    # 2. Multi-LoRA: one base matmul for the whole batch + a per-row low-rank delta (Punica / S-LoRA style).
    torch.manual_seed(0)
    d, r, n_ad = 64, 8, 4
    W = torch.randn(d, d, dtype=torch.float64)
    A = torch.randn(n_ad, r, d, dtype=torch.float64) * 0.1
    Bm = torch.randn(n_ad, d, r, dtype=torch.float64) * 0.1
    x = torch.randn(16, d, dtype=torch.float64)
    ad = torch.randint(0, n_ad, (16,))                       # which adapter each request uses
    batched = x @ W.T + torch.einsum("bdr,br->bd", Bm[ad], torch.einsum("brd,bd->br", A[ad], x))
    merged = torch.stack([x[i] @ (W + Bm[ad[i]] @ A[ad[i]]).T for i in range(16)])
    print(f"[multi-lora] {n_ad} adapters in one batch == per-request merged weights: {torch.allclose(batched, merged)}")

    # 3. Batch invariance: the same row reduced in a different order gives different bits.
    torch.manual_seed(0)
    row = torch.randn(4096, dtype=torch.float32) * 100
    sums = {chunks: torch.stack([c.sum() for c in row.chunk(chunks)]).sum().item() for chunks in (1, 4, 16, 64)}
    print(f"[determinism] float32 sum of one row by split count: " + ", ".join(f"{k}: {v:.6f}" for k, v in sums.items()))
    print(f"[determinism] distinct results: {len(set(sums.values()))}  -> a kernel whose split depends on batch size "
          f"gives batch-dependent outputs; fixing the split makes it batch-invariant")
    return


if __name__ == "__main__":
    app.run()
