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
    mo.md("# ag_eval\n\nAgents ch. 'Benchmarks': pass@k and pass^k estimators, compounding errors, benchmark noise,\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Agents ch. 'Benchmarks': pass@k and pass^k estimators, compounding errors, benchmark noise,
    a METR-style time-horizon fit, and cost per solved task."""
    import math

    import numpy as np

    rng = np.random.default_rng(0)

    # 1. pass@k from n samples per task with c correct. Unbiased: 1 - C(n-c, k)/C(n, k). Plug-in: 1 - (1-c/n)^k.
    tasks, n, k = 2000, 20, 5
    p = rng.beta(0.6, 0.9, tasks)                       # true per-task solve rates vary a lot
    c = rng.binomial(n, p)
    truth = np.mean(1 - (1 - p) ** k)
    unbiased = np.mean([1 - math.comb(n - ci, k) / math.comb(n, k) for ci in c])
    plug = np.mean(1 - (1 - c / n) ** k)
    print(f"[pass@k] k={k}, n={n}: truth {truth:.4f}   unbiased estimator {unbiased:.4f}   plug-in {plug:.4f} (biased low)")

    # 2. pass^k (tau-bench): all k independent tries succeed. Unbiased: C(c, k)/C(n, k).
    print("[pass^k] k   pass@k   pass^k     (same agent: pass@1 = %.3f)" % p.mean())
    for kk in (1, 2, 4, 8):
        pa = np.mean([1 - math.comb(n - ci, kk) / math.comb(n, kk) for ci in c])
        pw = np.mean([math.comb(ci, kk) / math.comb(n, kk) for ci in c])
        print(f"[pass^k] {kk}   {pa:.3f}    {pw:.3f}")

    # 3. Compounding: a task of s steps, each right with prob 0.95. With a check that catches a wrong step
    #    with prob d and up to 3 attempts, the effective per-step success rises.
    q, d = 0.95, 0.9
    # A step fails if a wrong attempt slips past the check, or all r attempts are wrong and caught.
    miss = lambda r: sum(((1 - q) * d) ** i for i in range(r)) * (1 - q) * (1 - d) + ((1 - q) * d) ** r
    print("[compound] steps   no checks   check+retry (3 tries, 90% detection)")
    for s in (10, 30, 100):
        print(f"[compound] {s:4d}     {q ** s:.3f}       {(1 - miss(3)) ** s:.3f}")

    # 4. Benchmark noise: SWE-bench Verified has 500 tasks.
    for acc in (0.5, 0.7):
        print(f"[noise] {acc:.0%} on 500 tasks: 95% CI +- {1.96 * math.sqrt(acc * (1 - acc) / 500) * 100:.1f} points")

    # 5. METR-style time horizon: P(success) is logistic in log2(human minutes). Fit and find the 50% point.
    dur = 2 ** rng.uniform(0, math.log2(960), 400)                   # 1 minute to 16 hours
    x = np.log2(dur)
    y = (rng.random(400) < 1 / (1 + np.exp(0.8 * (x - math.log2(120))))).astype(float)


    def fit(x, y):
        X = np.c_[np.ones_like(x), x]
        w = np.zeros(2)
        for _ in range(25):
            pr = 1 / (1 + np.exp(-X @ w))
            w -= np.linalg.solve(X.T @ (X * (pr * (1 - pr))[:, None]) + 1e-9 * np.eye(2), X.T @ (pr - y))
        return 2 ** (-w[0] / w[1])


    h = fit(x, y)
    boots = [fit(x[i], y[i]) for i in (rng.integers(0, 400, 400) for _ in range(300))]
    lo, hi = np.percentile(boots, [5, 95])
    print(f"[horizon] true 50% horizon 120 min; fitted {h:.0f} min (90% bootstrap CI {lo:.0f}-{hi:.0f})")

    # 6. Cost per solved task: retry until the harness's verifier passes (up to 4 tries).
    for name, cost, sr in (("large model", 1.20, 0.70), ("small model", 0.15, 0.35)):
        solved = 1 - (1 - sr) ** 4
        exp_cost = sum(cost * (1 - sr) ** i for i in range(4))
        print(f"[cost] {name}: ${cost:.2f}/attempt, {sr:.0%} per try -> {solved:.0%} solved within 4 tries, "
              f"${exp_cost / solved:.2f} per solved task")
    return


if __name__ == "__main__":
    app.run()
