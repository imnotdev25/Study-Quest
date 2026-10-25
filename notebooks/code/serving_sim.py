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
    mo.md("# serving_sim\n\nInference chs. 'Benchmark', 'Break One GPU', 'Continuous Batching': a discrete-time server simulator\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Inference chs. 'Benchmark', 'Break One GPU', 'Continuous Batching': a discrete-time server simulator.

    Step cost model for an 8B bf16 model on one H100 (rough constants, not a measurement):
        step = 6 ms (read weights) + 27 us per prefill token + 40 ns per KV token attended in decode.
    """
    import numpy as np

    W_MS, PRE_US, KV_NS = 6.0, 27.0, 40.0
    SLO_TTFT, SLO_TPOT = 1.0, 0.05            # seconds


    def workload(rate, n=600, seed=0):
        r = np.random.default_rng(seed)
        t = np.cumsum(r.exponential(1 / rate, n)) if rate else np.zeros(n)
        p = np.clip(r.lognormal(np.log(800), 0.6, n), 32, 4000).astype(int)
        o = np.clip(r.lognormal(np.log(200), 0.6, n), 8, 1000).astype(int)
        return t, p, o


    def simulate(mode, rate, max_batch=64, budget=2048, closed=None):
        arr, P, O = workload(rate)
        n = len(P)
        if closed:                                 # closed loop: `closed` users, each sends again when done
            arr = np.full(n, np.inf); arr[:closed] = 0
        now, nxt, wait, run = 0.0, 0, [], []
        first, last, gaps, finish = np.full(n, np.nan), np.zeros(n), np.zeros(n), np.full(n, np.nan)
        left_pre, left_out, issued = P.copy(), O.copy(), closed or n
        while np.isnan(finish).any():
            while nxt < n and arr[nxt] <= now:
                wait.append(nxt); nxt += 1
            if mode == "static":
                if not run and wait:
                    run, wait = wait[:max_batch], wait[max_batch:]
            else:
                while wait and len(run) < max_batch:
                    run.append(wait.pop(0))
            if not run:
                now = arr[nxt] if nxt < n and arr[nxt] < np.inf else now + 1e-3
                continue
            pre_tok, kv_tok, emit = 0, 0, []
            tok_budget = budget if mode == "chunked" else 10**9
            for i in run:
                if left_pre[i] > 0:
                    c = min(left_pre[i], tok_budget)
                    if c == 0:
                        continue
                    left_pre[i] -= c; pre_tok += c; tok_budget -= c
                    if left_pre[i] == 0:
                        emit.append(i)
                elif left_out[i] > 0:
                    kv_tok += P[i] + O[i] - left_out[i]; emit.append(i); tok_budget -= 1
            if mode == "static":                   # padded batch: finished slots still cost a decode
                kv_tok += sum(P[i] + O[i] for i in run if left_out[i] == 0)
            now += (W_MS + PRE_US * pre_tok / 1e3 + KV_NS * kv_tok / 1e6) / 1e3
            for i in emit:
                if np.isnan(first[i]):
                    first[i] = now
                else:
                    gaps[i] = max(gaps[i], now - last[i])
                last[i] = now
                left_out[i] -= 1
                if left_out[i] == 0:
                    finish[i] = now
                    if mode != "static":
                        run.remove(i)
                    if closed and issued < n:
                        arr[issued] = now; issued += 1
            if mode == "static" and all(left_out[i] == 0 for i in run):
                run = []
        if closed:
            pass
        ttft, e2e = first - arr, finish - arr
        tpot = (finish - first) / np.maximum(O - 1, 1)
        good = (ttft <= SLO_TTFT) & (tpot <= SLO_TPOT)
        span = finish.max() - arr.min()
        return dict(rps=n / span, tok=O.sum() / span, ttft50=np.median(ttft), ttft99=np.percentile(ttft, 99),
                    tpot50=np.median(tpot), itl99=np.percentile(gaps, 99), good=good.sum() / span,
                    frac=good.mean(), conc=(e2e.sum() / span), e2e=e2e.mean())


    print("[sim] mode        rate   req/s  out tok/s  TTFT p50  TTFT p99  TPOT p50  ITL p99  goodput (SLO ok)")
    for mode in ("static", "continuous", "chunked"):
        for rate in (4, 8, 14, 20):
            s = simulate(mode, rate)
            print(f"[sim] {mode:10s}  {rate:4d}  {s['rps']:6.2f}  {s['tok']:8.0f}  {s['ttft50']:7.2f}s  {s['ttft99']:7.2f}s"
                  f"  {s['tpot50'] * 1e3:6.1f}ms  {s['itl99'] * 1e3:5.0f}ms  {s['good']:5.2f}/s ({s['frac']:.0%})")

    s = simulate("chunked", 8)
    print(f"[little] L = lambda W: mean in system {s['conc']:.1f}  vs  rate x mean E2E {s['rps'] * s['e2e']:.1f}")

    print("[sweep] closed loop, chunked prefill: users -> throughput and latency")
    for users in (1, 4, 16, 32, 64, 128):
        s = simulate("chunked", 0, closed=users)
        print(f"[sweep] {users:4d} users  {s['tok']:7.0f} tok/s  TPOT p50 {s['tpot50'] * 1e3:6.1f} ms  TTFT p50 {s['ttft50']:6.2f} s")
    return


if __name__ == "__main__":
    app.run()
