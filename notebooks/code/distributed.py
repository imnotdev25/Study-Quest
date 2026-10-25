# sq-meta: {"title": "distributed", "emoji": "📓", "group": "Inference", "kind": "code", "desc": "Inference chs. 'Multi-GPU', 'MoE Inference', 'Long Context', 'Disaggregation': simulated parallelism", "topic": ""}
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
    mo.md("# distributed\n\nInference chs. 'Multi-GPU', 'MoE Inference', 'Long Context', 'Disaggregation': simulated parallelism\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Inference chs. 'Multi-GPU', 'MoE Inference', 'Long Context', 'Disaggregation': simulated parallelism
    and the bandwidth arithmetic behind each design."""
    import numpy as np
    import torch

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # 1. Tensor parallelism (Megatron MLP): column-split W1, row-split W2, one all-reduce (sum) at the end.
    d, f, tp = 64, 256, 4
    x, W1, W2 = torch.randn(8, d, dtype=torch.float64), torch.randn(f, d, dtype=torch.float64), torch.randn(d, f, dtype=torch.float64)
    full = torch.relu(x @ W1.T) @ W2.T
    parts = [torch.relu(x @ W1[r * f // tp:(r + 1) * f // tp].T) @ W2[:, r * f // tp:(r + 1) * f // tp].T for r in range(tp)]
    print(f"[tp] sum of {tp} shard outputs (all-reduce) == unsharded MLP: {torch.allclose(sum(parts), full)}")

    # Decode step for Llama-3.1-70B: weights read in parallel; 2 all-reduces per layer of B*d*2 bytes.
    P, L, D, B = 70.6e9, 80, 8192, 32
    for n in (2, 4, 8):
        t_w = 2 * P / n / 3.35e12
        ring = 2 * (n - 1) / n * B * D * 2                  # bytes each GPU sends per all-reduce
        t_c = 2 * L * (ring / 450e9 + 8e-6)                 # NVLink ~450 GB/s per direction + ~8 us latency
        print(f"[tp] TP={n}: weight read {t_w * 1e3:5.2f} ms + all-reduce {t_c * 1e3:4.2f} ms = {(t_w + t_c) * 1e3:5.2f} ms/step")

    # 2. Pipeline parallelism bubble: (p - 1) / (m + p - 1) with m micro-batches over p stages.
    print("[pp] bubble fraction  " + "  ".join(f"m={m}: {(4 - 1) / (m + 4 - 1):.0%}" for m in (1, 4, 16, 64)) + "  (p=4)")

    # 3. MoE expert parallelism: 256 experts on 32 GPUs, top-8 routing. The busiest GPU sets the step time.
    E, G, K = 256, 32, 8
    pop = rng.dirichlet(np.full(E, 2.0))                    # some experts are more popular
    for T in (64, 1024, 16384):
        load = np.zeros(G)
        for _ in range(T):
            load[(rng.choice(E, K, replace=False, p=pop)) // (E // G)] += 1
        print(f"[ep] {T:6d} tokens: busiest GPU / mean = {load.max() / load.mean():.2f}")
    tok_b = 7168 * 1                                        # DeepSeek-V3 hidden size, fp8 dispatch
    print(f"[ep] all-to-all per token: dispatch {K * tok_b / 1024:.0f} KiB (fp8) + combine {K * 7168 * 2 / 1024:.0f} KiB (bf16)")

    # 4. Context parallel / ring attention: each rank holds a KV slice; merge partial softmaxes with LSE.
    q, k, v = (torch.randn(n_, 32, dtype=torch.float64) for n_ in (4, 512, 512))
    ref = (q @ k.T / 32**0.5).softmax(-1) @ v
    m_, num, den = None, 0, 0
    for r in range(4):
        s = q @ k[r * 128:(r + 1) * 128].T / 32**0.5
        mr = s.amax(1, keepdim=True)
        m_new = mr if m_ is None else torch.maximum(m_, mr)
        scale_old = 0 if m_ is None else (m_ - m_new).exp()
        num = num * scale_old + (s - m_new).exp() @ v[r * 128:(r + 1) * 128]
        den = den * scale_old + (s - m_new).exp().sum(1, keepdim=True)
        m_ = m_new
    print(f"[ring] 4 KV shards merged == full attention: {torch.allclose(num / den, ref)}")

    # 5. Prefill/decode disaggregation: KV transfer vs prefill compute (H100 ~ 700 TFLOP/s effective).
    print("[pd] model, prompt          KV size    RDMA 50 GB/s   prefill compute")
    for name, P_, kvb, T in (("Llama-3.1-8B, 8k", 8e9, 131072, 8192), ("Llama-3.1-70B, 8k", 70.6e9, 327680, 8192),
                             ("DeepSeek-V3 MLA, 8k (37B active)", 37e9, 70272, 8192)):
        size = kvb * T
        print(f"[pd] {name:32s} {size / 2**30:5.2f} GiB  {size / 50e9 * 1e3:7.0f} ms   {2 * P_ * T / 700e12 * 1e3:6.0f} ms")

    # 6. Tiered KV: reload a 128k-token Llama-3.1-8B prefix vs recompute it.
    T = 131072
    size = 131072 * T
    recompute = (2 * 8e9 * T + 2 * 32 * T * T * 4096) / 700e12
    print(f"[tier] 128k prefix = {size / 2**30:.0f} GiB:  HBM-local 0 s | CPU over PCIe5 (~50 GB/s) {size / 50e9:.2f} s | "
          f"NVMe (~7 GB/s) {size / 7e9:.2f} s | recompute {recompute:.1f} s")
    return


if __name__ == "__main__":
    app.run()
