# sq-meta: {"title": "math_kit", "emoji": "📓", "group": "Math & Optimisation", "kind": "code", "desc": "Toolkit ch. 1-2: the math fundamentals, each checked numerically", "topic": ""}
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
    mo.md("# math_kit\n\nToolkit ch. 1-2: the math fundamentals, each checked numerically\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Toolkit ch. 1-2: the math fundamentals, each checked numerically."""
    import numpy as np

    rng = np.random.default_rng(0)

    # 1. Derivatives and gradients: central differences agree with the analytic gradient.
    f = lambda x: np.sum(x**2 * np.sin(x))
    grad_f = lambda x: 2 * x * np.sin(x) + x**2 * np.cos(x)
    x = rng.normal(size=5)
    h = 1e-5
    num = np.array([(f(x + h * e) - f(x - h * e)) / (2 * h) for e in np.eye(5)])
    print(f"[grad] max |numeric - analytic| = {np.abs(num - grad_f(x)).max():.1e}")

    # 2. Chain rule and Jacobian: y = tanh(Wx), J = diag(1 - y^2) W.
    W = rng.normal(size=(3, 4))
    y = np.tanh(W @ x[:4])
    J = (1 - y**2)[:, None] * W
    J_num = np.stack([(np.tanh(W @ (x[:4] + h * e)) - np.tanh(W @ (x[:4] - h * e))) / (2 * h)
                      for e in np.eye(4)], axis=1)
    print(f"[jacobian] shape {J.shape}, max err {np.abs(J - J_num).max():.1e}")

    # 3. Hadamard product: d(sum(a*b))/da = b, element by element.
    a, b = rng.normal(size=4), rng.normal(size=4)
    g_num = np.array([(np.sum((a + h * e) * b) - np.sum((a - h * e) * b)) / (2 * h) for e in np.eye(4)])
    print(f"[hadamard] a*b = {np.round(a * b, 3)}  numeric grad wrt a == b: {np.allclose(g_num, b)}")

    # 4. Entropy, cross-entropy, KL:  CE(p, q) = H(p) + KL(p || q).
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.4, 0.4, 0.2])
    H = -(p * np.log(p)).sum()
    CE = -(p * np.log(q)).sum()
    KL = (p * np.log(p / q)).sum()
    print(f"[info] H(p)={H:.4f}  CE(p,q)={CE:.4f}  KL(p||q)={KL:.4f}  H+KL={H + KL:.4f}")
    print(f"[info] KL(q||p)={(q * np.log(q / p)).sum():.4f}  (KL is not symmetric)")

    # 5. Monte Carlo KL estimators used in RLHF (samples from p, r = q/p):
    #    k1 = -log r (unbiased, can go negative), k3 = (r - 1) - log r (unbiased, always >= 0).
    s = rng.choice(3, size=200_000, p=p)
    r = q[s] / p[s]
    k1, k3 = -np.log(r), (r - 1) - np.log(r)
    print(f"[kl-est] k1 mean {k1.mean():.4f} std {k1.std():.3f} | k3 mean {k3.mean():.4f} std {k3.std():.3f}")

    # 6. SVD and Eckart-Young: the best rank-k approximation keeps the top k singular values,
    #    and its Frobenius error is sqrt(sum of the dropped sigma^2).
    M = rng.normal(size=(50, 4)) @ rng.normal(size=(4, 40)) + 0.05 * rng.normal(size=(50, 40))
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    for k in (1, 4, 10):
        Mk = (U[:, :k] * S[:k]) @ Vt[:k]
        print(f"[svd] rank {k:2d}: error {np.linalg.norm(M - Mk):7.3f}  predicted {np.sqrt((S[k:]**2).sum()):7.3f}")
    print(f"[svd] top 6 singular values {np.round(S[:6], 2)}  (rank-4 signal + noise)")

    # 7. EMA: m_t = beta m_{t-1} + (1 - beta) x_t. Bias correction divides by 1 - beta^t.
    beta, m = 0.9, 0.0
    for t, xt in enumerate([5.0] * 5, start=1):
        m = beta * m + (1 - beta) * xt
        print(f"[ema] t={t}  raw {m:.3f}  corrected {m / (1 - beta**t):.3f}")

    # 8. L1 vs L2: minimise 0.5 (w - c)^2 + penalty.
    #    L2 (lam/2 w^2): w = c / (1 + lam)  shrinks everything.  L1 (lam |w|): soft threshold, zeros small weights.
    c, lam = np.array([3.0, 0.5, -0.2, 1.2]), 0.6
    print(f"[norms] c        {c}")
    print(f"[norms] ridge    {np.round(c / (1 + lam), 3)}")
    print(f"[norms] lasso    {np.round(np.sign(c) * np.maximum(np.abs(c) - lam, 0), 3)}")
    return


if __name__ == "__main__":
    app.run()
