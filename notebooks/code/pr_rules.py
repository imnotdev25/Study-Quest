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
    mo.md("# pr_rules\n\nProbability: the rules, Bayes' theorem, and Bayesian updating\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Probability: the rules, Bayes' theorem, and Bayesian updating."""
    import numpy as np

    rng = np.random.default_rng(0)

    # 1. Bayes on a diagnostic test: P(disease | positive).
    prev, sens, spec = 0.01, 0.95, 0.95
    p_pos = sens * prev + (1 - spec) * (1 - prev)                 # law of total probability
    post1 = sens * prev / p_pos
    p_pos2 = sens * post1 + (1 - spec) * (1 - post1)
    post2 = sens * post1 / p_pos2                                 # yesterday's posterior is today's prior
    print(f"[bayes] P(+) = {p_pos:.4f}   P(D | +) = {post1:.4f}   after a second independent + : {post2:.4f}")
    n = 2_000_000
    d = rng.random(n) < prev
    t = np.where(d, rng.random(n) < sens, rng.random(n) < 1 - spec)
    print(f"[bayes] simulation of {n:,} people: P(D | +) = {d[t].mean():.4f}")

    # 2. Beta-Bernoulli: prior Beta(a, b), after h heads and t tails the posterior is Beta(a + h, b + t).
    a, b = 2.0, 2.0
    flips = rng.random(40) < 0.7
    for k in (0, 5, 20, 40):
        h = flips[:k].sum(); A, B = a + h, b + k - h
        mean, sd = A / (A + B), np.sqrt(A * B / ((A + B) ** 2 * (A + B + 1)))
        print(f"[beta] after {k:2d} flips ({h:2d} heads): Beta({A:.0f},{B:.0f})  mean {mean:.3f}  sd {sd:.3f}")
    A_seq, B_seq = a, b
    for f in flips:                                               # one flip at a time
        A_seq, B_seq = A_seq + f, B_seq + (1 - f)
    print(f"[beta] sequential updates == one batch update: {(A_seq, B_seq) == (a + flips.sum(), b + 40 - flips.sum())}")

    # 3. Grid posterior for the same coin (no conjugacy needed): prior x likelihood, normalise.
    theta = np.linspace(0.001, 0.999, 999)
    prior = theta ** (a - 1) * (1 - theta) ** (b - 1)
    like = theta ** flips.sum() * (1 - theta) ** (40 - flips.sum())
    post = prior * like; post /= post.sum()
    print(f"[grid] posterior mean on a grid {(theta * post).sum():.4f}  vs conjugate {(a + flips.sum()) / (a + b + 40):.4f}")

    # 4. Monty Hall: switching wins 2/3.
    car, pick = rng.integers(3, size=100_000), rng.integers(3, size=100_000)
    print(f"[monty] stay wins {(car == pick).mean():.3f}   switch wins {(car != pick).mean():.3f}")
    return


if __name__ == "__main__":
    app.run()
