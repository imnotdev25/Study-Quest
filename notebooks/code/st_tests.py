# sq-meta: {"title": "st_tests", "emoji": "📓", "group": "Probability & Statistics", "kind": "code", "desc": "Inference: comparing two models on one eval set, multiple comparisons, Hoeffding, importance sampling", "topic": ""}
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
    mo.md("# st_tests\n\nInference: comparing two models on one eval set, multiple comparisons, Hoeffding, importance sampling\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Inference: comparing two models on one eval set, multiple comparisons, Hoeffding, importance sampling."""
    import math

    import numpy as np

    rng = np.random.default_rng(0)
    Phi = lambda z: 0.5 * math.erfc(-z / math.sqrt(2))

    # 1. Two models on the same 500 questions. Their correctness is correlated (shared easy/hard items).
    n = 500
    difficulty = rng.normal(size=n)
    a = rng.random(n) < 1 / (1 + np.exp(-(difficulty + 0.75)))
    b = rng.random(n) < 1 / (1 + np.exp(-(difficulty + 0.45)))
    diff = a.astype(float) - b
    se_unpaired = math.sqrt(a.var(ddof=1) / n + b.var(ddof=1) / n)
    se_paired = diff.std(ddof=1) / math.sqrt(n)
    z_u, z_p = diff.mean() / se_unpaired, diff.mean() / se_paired
    print(f"[ab] accuracy A {a.mean():.3f}  B {b.mean():.3f}  gap {diff.mean():+.3f}")
    print(f"[ab] unpaired SE {se_unpaired:.4f}  z {z_u:.2f}  p {2 * (1 - Phi(abs(z_u))):.4f}")
    print(f"[ab] paired   SE {se_paired:.4f}  z {z_p:.2f}  p {2 * (1 - Phi(abs(z_p))):.4f}   (same data, smaller SE)")

    # Permutation test: under H0 the A/B labels are exchangeable within each question, so flip signs.
    flips = rng.choice([-1, 1], size=(20_000, n))
    null = (flips * diff).mean(1)
    print(f"[perm] paired sign-flip permutation p-value {np.mean(np.abs(null) >= abs(diff.mean())):.4f}")

    # 2. Multiple comparisons: 20 benchmarks, no real difference anywhere.
    trials = 5_000
    pvals = rng.random((trials, 20))                              # p-values are uniform under H0
    print(f"[multi] P(at least one p < 0.05 among 20 null tests) = {np.mean((pvals < 0.05).any(1)):.3f} "
          f"(theory {1 - 0.95**20:.3f});  with Bonferroni (p < 0.0025): {np.mean((pvals < 0.05 / 20).any(1)):.3f}")

    # 3. Hoeffding: P(|mean - mu| >= eps) <= 2 exp(-2 n eps^2). Questions needed for +-2 points at 95%.
    eps, delta = 0.02, 0.05
    print(f"[hoeffding] n >= ln(2/delta) / (2 eps^2) = {math.ceil(math.log(2 / delta) / (2 * eps**2)):,} questions "
          f"(no assumptions); normal approximation at p=0.5: {math.ceil(1.96**2 * 0.25 / eps**2):,}")

    # 4. Importance sampling: P(Z > 4) for Z ~ N(0,1) = 3.17e-5.
    exact = 1 - Phi(4)
    z = rng.normal(size=100_000)
    w_samples = rng.normal(4, 1, 100_000)                         # proposal q = N(4, 1)
    w = np.exp(-w_samples**2 / 2) / np.exp(-(w_samples - 4) ** 2 / 2)   # p/q
    is_est = np.mean((w_samples > 4) * w)
    print(f"[is] exact {exact:.3e}   naive MC (100k) {np.mean(z > 4):.3e}   importance sampling (100k) {is_est:.3e}")
    for shift in (0.5, 1.0, 2.0, 3.0):                            # policy drift -> effective sample size collapses
        s = rng.normal(0, 1, 10_000)
        ww = np.exp(-(s - shift) ** 2 / 2 + s**2 / 2)             # target N(shift,1), samples from N(0,1)
        print(f"[ess] target shifted by {shift}: ESS = (sum w)^2 / sum w^2 = {ww.sum()**2 / (ww**2).sum():7.0f} of 10,000"
              f"   (theory {10_000 * math.exp(-shift**2):.0f})")
    return


if __name__ == "__main__":
    app.run()
