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
    mo.md("# pr_dists\n\nProbability: the standard discrete and continuous distributions, theory vs simulation\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Probability: the standard discrete and continuous distributions, theory vs simulation."""
    import math

    import numpy as np

    rng = np.random.default_rng(0)
    N = 1_000_000

    table = [  # name, sampler, theoretical mean, theoretical variance
        ("Bernoulli(0.3)", lambda: rng.random(N) < 0.3, 0.3, 0.21),
        ("Binomial(20, 0.3)", lambda: rng.binomial(20, 0.3, N), 6.0, 4.2),
        ("Geometric(0.25) trials", lambda: rng.geometric(0.25, N), 4.0, 12.0),
        ("Poisson(4)", lambda: rng.poisson(4, N), 4.0, 4.0),
        ("Uniform(2, 8)", lambda: rng.uniform(2, 8, N), 5.0, 3.0),
        ("Exponential(rate 0.5)", lambda: rng.exponential(2, N), 2.0, 4.0),
        ("Normal(1, 9)", lambda: rng.normal(1, 3, N), 1.0, 9.0),
        ("Gamma(k=3, theta=2)", lambda: rng.gamma(3, 2, N), 6.0, 12.0),
        ("Beta(2, 5)", lambda: rng.beta(2, 5, N), 2 / 7, 10 / (49 * 8)),
        ("Laplace(0, b=1)", lambda: rng.laplace(0, 1, N), 0.0, 2.0),
    ]
    print("[dist] distribution              mean (theory)        variance (theory)")
    for name, sample, m, v in table:
        s = sample().astype(float)
        print(f"[dist] {name:24s} {s.mean():7.4f} ({m:7.4f})   {s.var():8.4f} ({v:8.4f})")

    # Poisson as the limit of Binomial(n, lam/n).
    lam = 3.0
    pois = np.array([math.exp(-lam) * lam**k / math.factorial(k) for k in range(40)])
    for n in (10, 100, 1000):
        p = lam / n
        binom = np.array([math.comb(n, k) * p**k * (1 - p) ** (n - k) if k <= n else 0.0 for k in range(40)])
        print(f"[limit] Binomial({n}, 3/{n}) vs Poisson(3): total variation {0.5 * np.abs(binom - pois).sum():.5f}")

    # Sums: Exponential sums are Gamma; Bernoulli sums are Binomial; Normal sums stay Normal.
    s = rng.exponential(2, (N, 3)).sum(1)
    print(f"[sums] sum of 3 Exponential(mean 2): mean {s.mean():.3f}, var {s.var():.3f}  -> Gamma(3, 2): 6, 12")
    return


if __name__ == "__main__":
    app.run()
