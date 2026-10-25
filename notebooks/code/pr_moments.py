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
    mo.md("# pr_moments\n\nProbability: expectation, variance, Jensen, tail bounds, moment generating functions\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Probability: expectation, variance, Jensen, tail bounds, moment generating functions."""
    import math

    import numpy as np

    rng = np.random.default_rng(0)
    N = 2_000_000

    X, Y = rng.exponential(2.0, N), rng.uniform(0, 6, N)          # independent; E 2, 3; Var 4, 3
    print(f"[linear] E[3X + 2Y + 1] = {np.mean(3 * X + 2 * Y + 1):.3f}   formula 3*2 + 2*3 + 1 = 13")
    print(f"[var] Var(X + Y) = {np.var(X + Y):.3f} = Var X + Var Y = {4 + 3}   Var(3X) = {np.var(3 * X):.2f} = 9 Var X = 36")
    print(f"[var] Var X = E[X^2] - (E X)^2: {np.mean(X**2) - np.mean(X)**2:.3f}")

    G = rng.gamma(3.0, 1.0, N)
    print(f"[jensen] log is concave: E[log G] = {np.mean(np.log(G)):.4f} <= log E[G] = {np.log(np.mean(G)):.4f}")

    # Tail of Exponential(mean 1): exact e^{-t}; Markov 1/t; Chebyshev (one-sided) <= 1/(t-1)^2.
    print("[tail] t   exact P(X >= t)   Markov 1/t   Chebyshev 1/(t-1)^2")
    for t in (2, 4, 8):
        print(f"[tail] {t}   {math.exp(-t):.5f}           {1 / t:.4f}       {1 / (t - 1) ** 2:.4f}")

    # MGF of Poisson(lam): M(s) = exp(lam (e^s - 1)). M'(0) = E X, M''(0) = E X^2.
    lam, h = 3.0, 1e-4
    M = lambda s: math.exp(lam * (math.exp(s) - 1))
    m1, m2 = (M(h) - M(-h)) / (2 * h), (M(h) - 2 * M(0) + M(-h)) / h**2
    P = rng.poisson(lam, N)
    print(f"[mgf] M'(0) = {m1:.4f} (E X = {P.mean():.4f})   M''(0) = {m2:.4f} (E X^2 = {np.mean(P.astype(float)**2):.4f})")

    # Skewness and kurtosis distinguish shapes that share a mean and variance.
    for name, s in (("normal", rng.normal(0, 1, N)), ("laplace", rng.laplace(0, 1 / math.sqrt(2), N)),
                    ("exponential-1", rng.exponential(1, N) - 1)):
        z = (s - s.mean()) / s.std()
        print(f"[shape] {name:13s} mean {s.mean():+.3f} var {s.var():.3f} skew {np.mean(z**3):+.3f} excess kurt {np.mean(z**4) - 3:+.3f}")
    return


if __name__ == "__main__":
    app.run()
