# sq-meta: {"title": "prod", "emoji": "📓", "group": "Inference", "kind": "code", "desc": "Inference chs. 'Routing and Autoscaling', 'Economics': prefix-aware routing simulation and cost math", "topic": ""}
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
    mo.md("# prod\n\nInference chs. 'Routing and Autoscaling', 'Economics': prefix-aware routing simulation and cost math\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Inference chs. 'Routing and Autoscaling', 'Economics': prefix-aware routing simulation and cost math."""
    from collections import OrderedDict

    import numpy as np

    rng = np.random.default_rng(0)

    # Routing: 8 replicas, each with an LRU prefix cache. Requests continue Zipf-popular conversations.
    R, N_CONV, N_REQ, CACHE = 8, 400, 6000, 25
    ranks = np.arange(1, N_CONV + 1)
    popularity = 1 / ranks**1.1 / (1 / ranks**1.1).sum()
    conv_len = np.zeros(N_CONV)
    arrivals = np.cumsum(rng.exponential(1 / 40, N_REQ))     # 40 req/s
    convs = rng.choice(N_CONV, N_REQ, p=popularity)


    def simulate(policy):
        caches = [OrderedDict() for _ in range(R)]
        free_at = np.zeros(R)
        clen = np.zeros(N_CONV)
        ttft, hits, total, rr = [], 0, 0, 0
        for t, c in zip(arrivals, convs):
            prompt = clen[c] + rng.integers(100, 400)
            backlog = np.maximum(free_at - t, 0)
            if policy == "round robin":
                r = rr % R; rr += 1
            elif policy == "least loaded":
                r = int(backlog.argmin())
            else:                                               # prefix-aware with a load guard
                r = c % R
                if backlog[r] > backlog.min() + 0.5:
                    r = int(backlog.argmin())
            cached = caches[r].get(c, 0)                       # prefix this replica holds for c
            hits += cached; total += prompt
            service = (prompt - cached) * 25e-6 + 0.02          # prefill only what is not cached
            start = max(t, free_at[r])
            free_at[r] = start + service
            ttft.append(free_at[r] - t)
            caches[r][c] = prompt; caches[r].move_to_end(c)
            if len(caches[r]) > CACHE:
                caches[r].popitem(last=False)
            clen[c] = prompt + rng.integers(100, 300)
        ttft = np.array(ttft)
        return hits / total, np.median(ttft), np.percentile(ttft, 99)


    for pol in ("round robin", "least loaded", "prefix-aware"):
        h, p50, p99 = simulate(pol)
        print(f"[route] {pol:13s}  prefix hit rate {h:5.1%}  TTFT p50 {p50 * 1e3:6.1f} ms  p99 {p99 * 1e3:7.1f} ms")

    # Economics. GPU prices are illustrative on-demand $/hour; replace with your quotes.
    print("[cost] GPU      $/h   out tok/s   $ per 1M output tokens")
    for gpu, price, tps in (("H100", 3.0, 3750), ("H200", 3.8, 5200), ("L40S", 1.1, 1100)):
        print(f"[cost] {gpu:6s} {price:5.2f}  {tps:9,d}   {price / (tps * 3600) * 1e6:6.3f}")
    peak_rps, good_rps, headroom = 120, 11.7, 1.3
    n = int(np.ceil(peak_rps / good_rps * headroom))
    print(f"[plan] peak {peak_rps} req/s, {good_rps} SLO-good req/s per replica, {headroom}x headroom -> {n} replicas")
    print(f"[plan] daily cost at $3/h: ${n * 3 * 24:,.0f} if always on; with autoscaling to a 30% night trough "
          f"~${(n * 12 + 0.3 * n * 12) * 3:,.0f}")
    return


if __name__ == "__main__":
    app.run()
