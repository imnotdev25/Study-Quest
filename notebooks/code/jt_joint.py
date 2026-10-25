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
    mo.md("# jt_joint\n\nJoint RVs: two discrete, two continuous, moments of joint distributions, covariance and correlation\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Joint RVs: two discrete, two continuous, moments of joint distributions, covariance and correlation."""
    import numpy as np

    rng = np.random.default_rng(0)

    # 1. Two discrete variables: a joint table, marginals, conditionals, independence, mutual information.
    P = np.array([[0.20, 0.10, 0.05],
                  [0.05, 0.25, 0.35]])                             # rows X in {0,1}, columns Y in {0,1,2}
    px, py = P.sum(1), P.sum(0)
    print(f"[discrete] P(X) = {px}   P(Y) = {py}   P(Y | X=1) = {np.round(P[1] / px[1], 3)}")
    print(f"[discrete] independent? max |P(x,y) - P(x)P(y)| = {np.abs(P - np.outer(px, py)).max():.3f}")
    MI = (P * np.log(P / np.outer(px, py))).sum()
    print(f"[discrete] mutual information I(X;Y) = KL(P_xy || P_x P_y) = {MI:.4f} nats")
    x_vals, y_vals = np.array([0, 1]), np.array([0, 1, 2])
    EX, EY = px @ x_vals, py @ y_vals
    EXY = (P * np.outer(x_vals, y_vals)).sum()
    print(f"[discrete] E X {EX:.2f}  E Y {EY:.2f}  E XY {EXY:.2f}  Cov = E XY - E X E Y = {EXY - EX * EY:.4f}")

    # 2. Two continuous variables: f(x, y) = x + y on the unit square.
    #    Marginal f_X(x) = x + 1/2;  E X = 7/12;  E XY = 1/3;  Cov = 1/3 - 49/144 = -1/144.
    n = 600
    c = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(c, c, indexing="ij")
    f = X + Y
    dA = 1 / n**2
    print(f"[continuous] total mass {f.sum() * dA:.6f}   E X {(X * f).sum() * dA:.6f} (7/12 = {7 / 12:.6f})")
    print(f"[continuous] Cov(X,Y) {(X * Y * f).sum() * dA - ((X * f).sum() * dA) ** 2:.6f}  (-1/144 = {-1 / 144:.6f})")
    u = rng.random((3_000_000, 3))
    keep = u[:, 2] * 2 < u[:, 0] + u[:, 1]                         # rejection sampling from f (max 2)
    s = u[keep, :2]
    print(f"[continuous] rejection-sampled {len(s):,} points: Cov {np.cov(s.T)[0, 1]:.6f}")

    # 3. Iterated expectation and total variance: X | K ~ N(mu_K, s_K^2), K in {0, 1} w.p. {0.3, 0.7}.
    w, mu, sd = np.array([0.3, 0.7]), np.array([-2.0, 3.0]), np.array([1.0, 0.5])
    K = rng.random(2_000_000) < w[1]
    Xm = np.where(K, rng.normal(mu[1], sd[1], K.size), rng.normal(mu[0], sd[0], K.size))
    EX = w @ mu
    VarX = w @ sd**2 + w @ (mu - EX) ** 2                          # E[Var(X|K)] + Var(E[X|K])
    print(f"[total] E X = E[E[X|K]] = {EX:.3f} (sim {Xm.mean():.3f});  Var X = {w @ sd**2:.3f} + {w @ (mu - EX) ** 2:.3f}"
          f" = {VarX:.3f} (sim {Xm.var():.3f})")

    # 4. Correlation: scale-free, bounded by 1, and blind to nonlinear dependence.
    Z = rng.normal(size=1_000_000)
    W = 0.6 * Z + 0.8 * rng.normal(size=Z.size)
    print(f"[corr] corr(Z, W) {np.corrcoef(Z, W)[0, 1]:.4f}   corr(5Z + 2, -3W) {np.corrcoef(5 * Z + 2, -3 * W)[0, 1]:.4f}")
    print(f"[corr] Y = Z^2: corr(Z, Y) {np.corrcoef(Z, Z**2)[0, 1]:+.4f} but E[Y | |Z|>2] = {np.mean(Z[np.abs(Z) > 2] ** 2):.2f}"
          f" vs E[Y] = {np.mean(Z**2):.2f}  (uncorrelated, not independent)")
    return


if __name__ == "__main__":
    app.run()
