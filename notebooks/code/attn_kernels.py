# sq-meta: {"title": "attn_kernels", "emoji": "📓", "group": "Inference", "kind": "code", "desc": "Inference ch. 'Attention Optimization': tiled online-softmax attention (FlashAttention), split-KV decode", "topic": ""}
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
    mo.md("# attn_kernels\n\nInference ch. 'Attention Optimization': tiled online-softmax attention (FlashAttention), split-KV decode\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Inference ch. 'Attention Optimization': tiled online-softmax attention (FlashAttention), split-KV decode
    (flash-decoding), paged gather, HBM traffic, and decode arithmetic intensity of MHA/GQA/MLA."""
    import torch

    torch.manual_seed(0)


    def naive(q, k, v):
        s = q @ k.T / q.shape[-1] ** 0.5
        s = s.masked_fill(torch.ones_like(s, dtype=torch.bool).triu(1 + k.shape[0] - q.shape[0]), float("-inf"))
        return s.softmax(-1) @ v


    def flash(q, k, v, Br=64, Bc=64):
        """FlashAttention forward: never materialise the T x T score matrix. Causal."""
        T, d = q.shape
        out = torch.empty_like(q)
        for i in range(0, T, Br):
            qi = q[i:i + Br]
            m = torch.full((len(qi), 1), float("-inf"), dtype=q.dtype)   # running max
            l = torch.zeros(len(qi), 1, dtype=q.dtype)                    # running denominator
            acc = torch.zeros_like(qi)                                    # running numerator
            for j in range(0, i + Br, Bc):                                # skip tiles above the diagonal
                s = qi @ k[j:j + Bc].T / d**0.5
                rows, cols = torch.arange(i, i + len(qi))[:, None], torch.arange(j, j + s.shape[1])[None]
                s = s.masked_fill(cols > rows, float("-inf"))
                m_new = torch.maximum(m, s.amax(1, keepdim=True))
                p = (s - m_new).exp()
                corr = (m - m_new).exp()                                  # rescale what we had
                l = l * corr + p.sum(1, keepdim=True)
                acc = acc * corr + p @ v[j:j + Bc]
                m = m_new
            out[i:i + Br] = acc / l
        return out


    T, d = 512, 64
    q, k, v = (torch.randn(T, d, dtype=torch.float64) for _ in range(3))
    print(f"[flash] tiled == naive: {torch.allclose(flash(q, k, v), naive(q, k, v))}  (max err "
          f"{(flash(q, k, v) - naive(q, k, v)).abs().max():.1e})")

    # HBM traffic per head, bf16: naive writes and reads the T x T matrix; flash streams K, V per query tile.
    for T_ in (4096, 32768):
        naive_b = (4 * T_ * d + 2 * T_ * T_ * 2) * 2       # Q,K,V,O + S written & read, then P written & read
        flash_b = (2 * T_ * d + (T_ / 128) * 2 * T_ * d) * 2
        print(f"[hbm] T={T_:6d}: naive {naive_b / 2**20:8.1f} MiB  flash {flash_b / 2**20:7.1f} MiB  "
              f"({naive_b / flash_b:.0f}x less)")

    # Flash-decoding: one query, KV split into chunks processed in parallel, merged with log-sum-exp.
    qd = torch.randn(1, d, dtype=torch.float64)
    parts = []
    for j in range(0, T, 128):
        s = qd @ k[j:j + 128].T / d**0.5
        parts.append((s.amax(), (s - s.amax()).exp().sum(), (s - s.amax()).exp() @ v[j:j + 128]))
    M = max(p[0] for p in parts)
    num = sum(p[2] * (p[0] - M).exp() for p in parts)
    den = sum(p[1] * (p[0] - M).exp() for p in parts)
    print(f"[split-kv] {len(parts)} chunks merged == full decode attention: {torch.allclose(num / den, naive(qd, k, v))}")

    # Paged attention: the same K/V scattered over a block pool, read through a block table.
    BS = 16
    pool = torch.zeros(64, BS, d, dtype=torch.float64)
    table = torch.randperm(64)[:T // BS]
    pool[table] = k.view(-1, BS, d)
    print(f"[paged] gather through block table == contiguous K: {torch.equal(pool[table].flatten(0, 1), k)}")

    # Decode arithmetic intensity of attention (FLOPs per byte of KV read), bf16.
    print("[intensity] layout                 FLOP per KV byte")
    for name, qh, kvh, kvdim in (("MHA 32q/32kv", 32, 32, 128), ("GQA 32q/8kv", 32, 8, 128),
                                 ("MQA 32q/1kv", 32, 1, 128), ("MLA 128q, 576 latent", 128, 1, 576)):
        flops = 4 * qh * 128 if "MLA" not in name else 4 * qh * 512   # per cached token
        byts = (2 * kvh * kvdim if "MLA" not in name else kvdim) * 2
        print(f"[intensity] {name:22s} {flops / byts:6.1f}")
    return


if __name__ == "__main__":
    app.run()
