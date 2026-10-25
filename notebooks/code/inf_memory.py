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
    mo.md("# inf_memory\n\nInference ch. 'Prefill, Decode, KV Cache and Memory': memory math, the roofline, and timing a cache\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Inference ch. 'Prefill, Decode, KV Cache and Memory': memory math, the roofline, and timing a cache."""
    import time

    import torch

    from tiny_gpt import Cfg, TinyGPT, generate_cached, generate_naive

    GiB = 2**30
    # name: (params, layers, kv_heads, head_dim) ; MLA stores (latent + rope) per layer instead of 2*kv*hd
    MODELS = {"Llama-3.1-8B": (8.0e9, 32, 8, 128), "Qwen3-32B": (32.8e9, 64, 8, 128),
              "Llama-3.1-70B": (70.6e9, 80, 8, 128), "DeepSeek-V3 (MLA)": (671e9, 61, None, 512 + 64)}


    def kv_bytes_per_token(layers, kv, hd, bytes_=2):
        return layers * (hd if kv is None else 2 * kv * hd) * bytes_


    print("[mem] model               weights bf16   KV/token   KV tokens on 1xH100 (80 GB, 90%, 2 GiB act.)")
    for name, (P, L, kv, hd) in MODELS.items():
        w = P * 2
        per = kv_bytes_per_token(L, kv, hd)
        free = 0.9 * 80e9 - w - 2 * GiB
        fit = f"{int(free / per):,} tokens = {int(free / per / 4096)} x 4k requests" if free > 0 else "weights do not fit"
        print(f"[mem] {name:19s} {w / GiB:8.1f} GiB  {per / 1024:6.0f} KiB   {fit}")

    # Roofline: H100 SXM, dense bf16 ~989 TFLOP/s, HBM3 ~3.35 TB/s -> ridge at ~295 FLOP/byte.
    peak, bw = 989e12, 3.35e12
    print(f"[roof] ridge point {peak / bw:.0f} FLOP/byte")
    print("[roof] decode batch   FLOP/byte (weights only)   bound      ideal step, 8B bf16")
    P = 8.0e9
    for B in (1, 8, 64, 256, 1024):
        intensity = 2 * P * B / (2 * P)                      # 2PB FLOPs over 2P bytes of weights
        t = max(2 * P * B / peak, 2 * P / bw)
        print(f"[roof] {B:5d}        {intensity:8.0f}                  {'compute' if intensity > peak / bw else 'memory ':8s}  "
              f"{t * 1e3:6.2f} ms -> {B / t:9,.0f} tok/s")

    # Prefill vs decode on a tiny CPU model, and what the KV cache buys.
    m = TinyGPT(Cfg(vocab=256, d=256, n_layers=4, n_heads=8, n_kv=2, dtype=torch.float32))
    prompt = list(range(200))
    t0 = time.perf_counter(); m.full(prompt); t_pre = time.perf_counter() - t0
    t0 = time.perf_counter(); a = generate_cached(m, prompt, 64); t_c = time.perf_counter() - t0
    t0 = time.perf_counter(); b = generate_naive(m, prompt, 64); t_n = time.perf_counter() - t0
    print(f"[time] prefill 200 tokens in one pass: {t_pre * 1e3:.1f} ms ({t_pre / 200 * 1e3:.2f} ms/token)")
    print(f"[time] decode 64 tokens with KV cache: {t_c * 1e3:.0f} ms   without cache: {t_n * 1e3:.0f} ms   "
          f"same tokens: {a == b}")
    return


if __name__ == "__main__":
    app.run()
