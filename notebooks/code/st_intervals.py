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
    mo.md("# st_intervals\n\nInference: confidence intervals and whether they really cover 95%\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Inference: confidence intervals and whether they really cover 95%."""
    import math

    import numpy as np

    rng = np.random.default_rng(0)
    T975 = {5: 2.776, 10: 2.262, 30: 2.045}                       # t quantiles for df = n - 1 (n = 5, 10, 30)

    # 1. Mean of a normal, sigma unknown: z-interval with s undercovers for small n, t-interval is exact.
    print("[ci] n    z-interval coverage   t-interval coverage   (target 0.95)")
    for n in (5, 10, 30):
        x = rng.normal(0, 1, (100_000, n))
        m, s = x.mean(1), x.std(1, ddof=1)
        cz = np.mean(np.abs(m) <= 1.96 * s / math.sqrt(n))
        ct = np.mean(np.abs(m) <= T975[n] * s / math.sqrt(n))
        print(f"[ci] {n:2d}   {cz:.4f}                {ct:.4f}")

    # 2. A proportion near 0 or 1 (eval accuracy 95% on 40 items): Wald undercovers, Wilson holds up.
    p, n = 0.95, 40
    k = rng.binomial(n, p, 200_000)
    ph = k / n
    wald = np.abs(ph - p) <= 1.96 * np.sqrt(ph * (1 - ph) / n)
    z = 1.96
    center = (ph + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    wilson = np.abs(center - p) <= half
    print(f"[prop] p=0.95, n=40: Wald coverage {wald.mean():.4f}   Wilson coverage {wilson.mean():.4f}")

    # 3. Bootstrap percentile interval for a median (no formula needed). Check coverage by repetition.
    true_median = math.log(2)                                      # Exponential(1)
    hits = 0
    for _ in range(1000):
        x = rng.exponential(1, 60)
        boots = np.median(rng.choice(x, (2000, 60)), axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        hits += lo <= true_median <= hi
    print(f"[boot] percentile bootstrap CI for a median, n=60: coverage {hits / 1000:.3f}")

    # 4. Sample size for a margin of error: n = z^2 p(1-p) / E^2.
    for E in (0.05, 0.02, 0.01):
        print(f"[size] +-{E:.0%} at 95%, worst case p=0.5: n = {math.ceil(1.96**2 * 0.25 / E**2):,}")
    return


if __name__ == "__main__":
    app.run()
