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
    mo.md("# pr_continuous\n\nProbability: continuous random variables and transformations (inverse CDF, Box-Muller,\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Probability: continuous random variables and transformations (inverse CDF, Box-Muller,
    change of variables, Gumbel-max, and the reparameterisation vs score-function gradient)."""
    import numpy as np

    rng = np.random.default_rng(0)
    N = 1_000_000

    # 1. Exponential(rate 2): CDF 1 - e^{-2x}; memoryless P(X > s + t | X > s) = P(X > t).
    x = rng.exponential(1 / 2, N)
    print(f"[exp] P(X <= 0.5) sim {np.mean(x <= 0.5):.4f}  CDF {1 - np.exp(-1):.4f}")
    print(f"[exp] memoryless: P(X > 1.3 | X > 1) = {np.mean(x[x > 1] > 1.3):.4f}  P(X > 0.3) = {np.mean(x > 0.3):.4f}")

    # 2. Inverse-CDF sampling: if U ~ U(0,1) then F^{-1}(U) ~ F.  Exponential: -ln(1-U)/rate.
    u = rng.random(N)
    xi = -np.log(1 - u) / 2
    print(f"[inverse-cdf] mean {xi.mean():.4f} (0.5)  var {xi.var():.4f} (0.25)")

    # 3. Change of variables: Y = X^2 with X ~ U(0,1) has density f_Y(y) = 1 / (2 sqrt y).
    y = rng.random(N) ** 2
    edges = np.array([0.01, 0.1, 0.4, 0.9])
    for lo, hi in zip(edges[:-1], edges[1:]):
        print(f"[transform] P({lo} < Y < {hi}): sim {np.mean((y > lo) & (y < hi)):.4f}  "
              f"integral of 1/(2 sqrt y) = sqrt(hi) - sqrt(lo) = {np.sqrt(hi) - np.sqrt(lo):.4f}")

    # 4. Box-Muller: two uniforms -> two independent N(0,1), via the polar Jacobian.
    u1, u2 = rng.random(N), rng.random(N)
    r = np.sqrt(-2 * np.log(u1))
    z1, z2 = r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)
    print(f"[box-muller] mean {z1.mean():+.4f}  var {z1.var():.4f}  corr(z1, z2) {np.corrcoef(z1, z2)[0, 1]:+.4f}"
          f"  P(|z|<1.96) {np.mean(np.abs(z1) < 1.96):.4f}")

    # 5. Gumbel-max: argmax(logits + Gumbel noise) is an exact sample from softmax(logits).
    logits = np.array([2.0, 1.0, 0.0, -1.0])
    g = -np.log(-np.log(rng.random((N, 4))))
    counts = np.bincount((logits + g).argmax(1), minlength=4) / N
    print(f"[gumbel] frequencies {np.round(counts, 4)}  softmax {np.round(np.exp(logits) / np.exp(logits).sum(), 4)}")

    # 6. Gradient of E_{x ~ N(mu, 1)}[x^2] = 2 mu, two ways (mu = 1.5 -> 3).
    mu, n = 1.5, 1000
    eps = rng.normal(size=(2000, n))
    pathwise = (2 * (mu + eps)).mean(1)                    # reparameterise x = mu + eps, differentiate x^2
    score = ((mu + eps) ** 2 * eps).mean(1)                # REINFORCE: f(x) d/dmu log p(x) = x^2 (x - mu)
    print(f"[grad] pathwise estimate {pathwise.mean():.3f} +- {pathwise.std():.3f}   "
          f"score-function {score.mean():.3f} +- {score.std():.3f}   (true 3.000)")
    return


if __name__ == "__main__":
    app.run()
