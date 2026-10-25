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
    mo.md("# st_likelihood\n\nInference: maximum likelihood, Fisher information, MAP, and logistic regression by Newton's method\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Inference: maximum likelihood, Fisher information, MAP, and logistic regression by Newton's method."""
    import numpy as np

    rng = np.random.default_rng(0)

    # 1. Bernoulli MLE = sample mean; Gaussian MLE = sample mean and the 1/n variance.
    x = rng.random(200) < 0.3
    grid = np.linspace(0.01, 0.99, 9801)
    ll = x.sum() * np.log(grid) + (len(x) - x.sum()) * np.log(1 - grid)
    print(f"[mle] Bernoulli: grid argmax {grid[ll.argmax()]:.4f}  closed form mean {x.mean():.4f}")
    g = rng.normal(5, 2, 500)
    print(f"[mle] Gaussian: mu-hat {g.mean():.4f}  sigma2-hat (1/n) {((g - g.mean())**2).mean():.4f}  true 5, 4")

    # 2. Fisher information and the sampling spread of the MLE: Var(p-hat) ~ 1 / (n I(p)), I(p) = 1/(p(1-p)).
    p, n = 0.3, 200
    est = (rng.random((20_000, n)) < p).mean(1)
    print(f"[fisher] Var of MLE across 20,000 datasets {est.var():.6f}   Cramer-Rao 1/(n I) = {p * (1 - p) / n:.6f}")

    # 3. Exponential rate MLE 1/mean is biased in small samples: E = n/(n-1) lam.
    lam = 2.0
    for n in (5, 50):
        e = 1 / rng.exponential(1 / lam, (200_000, n)).mean(1)
        print(f"[bias] n={n:2d}: mean of rate MLE {e.mean():.4f}   theory n/(n-1) lam = {n / (n - 1) * lam:.4f}")

    # 4. MAP with a Beta(a, b) prior = (h + a - 1) / (n + a + b - 2): shrinks small samples toward the prior.
    a = b = 5
    for h, n in ((3, 3), (30, 50), (300, 500)):
        print(f"[map] {h:3d}/{n:3d} heads: MLE {h / n:.3f}   MAP {(h + a - 1) / (n + a + b - 2):.3f}")

    # 5. Logistic regression: NLL = binary cross-entropy. Gradient X^T (p - y), Hessian X^T diag(p(1-p)) X.
    X = np.c_[np.ones(1000), rng.normal(size=(1000, 2))]
    w_true = np.array([-0.5, 2.0, -1.0])
    y = (rng.random(1000) < 1 / (1 + np.exp(-X @ w_true))).astype(float)
    w = np.zeros(3)
    for it in range(8):
        pr = 1 / (1 + np.exp(-X @ w))
        grad, H = X.T @ (pr - y), X.T @ (X * (pr * (1 - pr))[:, None])
        w -= np.linalg.solve(H, grad)
        nll = -(y * np.log(pr) + (1 - y) * np.log(1 - pr)).mean()
        if it in (0, 1, 2, 5, 7):
            print(f"[logreg] Newton iter {it}: mean BCE {nll:.6f}  |grad| {np.linalg.norm(grad):.2e}")
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    print(f"[logreg] w-hat {np.round(w, 3)}  true {w_true}  standard errors from the inverse Hessian {np.round(se, 3)}")
    return


if __name__ == "__main__":
    app.run()
