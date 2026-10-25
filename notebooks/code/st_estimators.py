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
    mo.md("# st_estimators\n\nInference: sampling behaviour (LLN, CLT), bias, variance and MSE of estimators\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Inference: sampling behaviour (LLN, CLT), bias, variance and MSE of estimators."""
    import math

    import numpy as np

    rng = np.random.default_rng(0)

    # 1. Law of large numbers: running mean of die rolls.
    rolls = rng.integers(1, 7, 100_000)
    for n in (10, 1000, 100_000):
        print(f"[lln] n = {n:7,d}   mean {rolls[:n].mean():.4f}   (E = 3.5)")

    # 2. CLT: standardised means of a skewed Exponential(1). Skewness of the mean is 2/sqrt(n).
    Phi = lambda z: 0.5 * math.erfc(-z / math.sqrt(2))
    for n in (1, 5, 30, 200):
        m = rng.exponential(1, (100_000, n)).mean(1)
        z = (m - 1) * math.sqrt(n)
        print(f"[clt] n = {n:3d}   sd of mean {m.std():.4f} (1/sqrt n = {1 / math.sqrt(n):.4f})  "
              f"P(Z <= -1) {np.mean(z <= -1):.4f} (normal {Phi(-1):.4f})  skew {np.mean(((z - z.mean()) / z.std())**3):.3f}")

    # 3. Variance estimators: divide by n-1 (unbiased), n (MLE), n+1 (lowest MSE for normals).
    n, sigma2 = 10, 4.0
    x = rng.normal(0, 2, (400_000, n))
    ss = ((x - x.mean(1, keepdims=True)) ** 2).sum(1)
    print("[var-est] divisor   bias      variance   MSE")
    for div in (n - 1, n, n + 1):
        e = ss / div
        print(f"[var-est] n{div - n:+d}       {e.mean() - sigma2:+.4f}   {e.var():.4f}     {np.mean((e - sigma2) ** 2):.4f}")

    # 4. Consistency: the sample median of a normal has variance pi/(2n) sigma^2, i.e. 57% more than the mean.
    x = rng.normal(0, 1, (50_000, 101))
    print(f"[eff] n=101: Var(mean) {x.mean(1).var():.5f}  Var(median) {np.median(x, 1).var():.5f}  "
          f"ratio {np.median(x, 1).var() / x.mean(1).var():.3f} (pi/2 = {math.pi / 2:.3f})")
    return


if __name__ == "__main__":
    app.run()
