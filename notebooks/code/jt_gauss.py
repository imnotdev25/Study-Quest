# sq-meta: {"title": "jt_gauss", "emoji": "📓", "group": "Probability & Statistics", "kind": "code", "desc": "Joint RVs: covariance matrices and Gaussian combinations (sums, linear maps, conditioning)", "topic": ""}
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
    mo.md("# jt_gauss\n\nJoint RVs: covariance matrices and Gaussian combinations (sums, linear maps, conditioning)\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Joint RVs: covariance matrices and Gaussian combinations (sums, linear maps, conditioning)."""
    import numpy as np

    rng = np.random.default_rng(0)
    N = 1_000_000

    # 1. Sum of independent normals: means add, variances add.
    s = rng.normal(1, 2, N) + rng.normal(-3, 1.5, N)
    print(f"[sum] N(1,4) + N(-3,2.25): mean {s.mean():.3f} (-2), var {s.var():.3f} (6.25)")

    # 2. Correlated sum: Var(X + Y) = Var X + Var Y + 2 Cov(X, Y).
    Sigma = np.array([[4.0, 2.4, 0.0], [2.4, 9.0, -1.5], [0.0, -1.5, 1.0]])
    L = np.linalg.cholesky(Sigma)                                  # Sigma = L L^T
    mu = np.array([1.0, 0.0, 2.0])
    Z = mu + rng.normal(size=(N, 3)) @ L.T                         # x = mu + L z
    print(f"[mvn] sample covariance from Cholesky sampling:\n{np.round(np.cov(Z.T), 2)}")
    print(f"[mvn] Var(X1 + X2) = {np.var(Z[:, 0] + Z[:, 1]):.3f}   formula 4 + 9 + 2(2.4) = {4 + 9 + 4.8}")

    # 3. Linear map: AX + b ~ N(A mu + b, A Sigma A^T).
    A = np.array([[1.0, -1.0, 0.0], [0.5, 0.5, 2.0]])
    Y = Z @ A.T
    print(f"[linear] cov(AX) sample vs A Sigma A^T  max diff {np.abs(np.cov(Y.T) - A @ Sigma @ A.T).max():.3f}")

    # 4. Eigen-decomposition: principal axes of the density ellipse; PSD means all eigenvalues >= 0.
    ev, V = np.linalg.eigh(Sigma[:2, :2])
    print(f"[eig] Sigma[:2,:2] eigenvalues {np.round(ev, 3)}, principal axis {np.round(V[:, 1], 3)}")

    # 5. Conditioning: X1 | X2 = x2 ~ N(mu1 + S12/S22 (x2 - mu2), S11 - S12^2/S22).
    x2 = 3.0
    cm = mu[0] + Sigma[0, 1] / Sigma[1, 1] * (x2 - mu[1])
    cv = Sigma[0, 0] - Sigma[0, 1] ** 2 / Sigma[1, 1]
    near = Z[np.abs(Z[:, 1] - x2) < 0.05, 0]
    print(f"[cond] E[X1 | X2=3] formula {cm:.3f}, sample {near.mean():.3f}   Var formula {cv:.3f}, sample {near.var():.3f}")

    # 6. Product of Gaussian densities (combining two noisy measurements): precision-weighted average.
    m1, v1, m2, v2 = 10.0, 4.0, 14.0, 1.0
    v = 1 / (1 / v1 + 1 / v2); m = v * (m1 / v1 + m2 / v2)
    print(f"[fuse] N(10, 4) x N(14, 1) -> N({m:.2f}, {v:.2f})   (the precise sensor gets 4x the weight)")
    return


if __name__ == "__main__":
    app.run()
