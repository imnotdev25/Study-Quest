# sq-meta: {"title": "mf_jacobian", "emoji": "📓", "group": "Math & Optimisation", "kind": "code", "desc": "Math I: vector-valued maps, Jacobians and local (Taylor) approximation", "topic": ""}
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
    mo.md("# mf_jacobian\n\nMath I: vector-valued maps, Jacobians and local (Taylor) approximation\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Math I: vector-valued maps, Jacobians and local (Taylor) approximation."""
    import numpy as np

    rng = np.random.default_rng(0)


    def jac_num(f, x, h=1e-6):
        """Central-difference Jacobian: column j is (f(x + h e_j) - f(x - h e_j)) / 2h."""
        return np.stack([(f(x + h * e) - f(x - h * e)) / (2 * h) for e in np.eye(len(x))], axis=1)


    # 1. Polar -> Cartesian, R^2 -> R^2. The Jacobian determinant r is the area-scaling factor.
    polar = lambda u: np.array([u[0] * np.cos(u[1]), u[0] * np.sin(u[1])])
    u = np.array([2.0, 0.7])
    J = np.array([[np.cos(u[1]), -u[0] * np.sin(u[1])], [np.sin(u[1]), u[0] * np.cos(u[1])]])
    print(f"[jac] polar map at r=2, theta=0.7: analytic == numeric {np.allclose(J, jac_num(polar, u))}, "
          f"det J = {np.linalg.det(J):.4f} (= r)")

    # 2. A map R^2 -> R^3 and its linear approximation f(x0 + d) ~ f(x0) + J d. Error shrinks like |d|^2.
    f = lambda x: np.array([np.sin(x[0]) * x[1], np.exp(x[0] - x[1]), x[0] ** 2 + x[1] ** 3])
    x0 = np.array([0.3, -0.5])
    J0 = jac_num(f, x0)
    d = rng.normal(size=2); d /= np.linalg.norm(d)
    print(f"[jac] f: R^2 -> R^3, Jacobian shape {J0.shape}")
    for s in (1e-1, 1e-2, 1e-3):
        err = np.linalg.norm(f(x0 + s * d) - f(x0) - J0 @ (s * d))
        print(f"[taylor-1] |d| = {s:.0e}   error {err:.2e}   error / |d|^2 = {err / s**2:.3f}")

    # 3. Second order for a scalar function: log-sum-exp. Gradient = softmax, Hessian = diag(p) - p p^T.
    lse = lambda z: z.max() + np.log(np.exp(z - z.max()).sum())
    softmax = lambda z: np.exp(z - lse(z))
    z0 = rng.normal(size=4)
    p = softmax(z0)
    H = np.diag(p) - np.outer(p, p)
    print(f"[hessian] LSE Hessian == diag(p) - pp^T: {np.allclose(H, jac_num(softmax, z0), atol=1e-8)}, "
          f"eigenvalues {np.round(np.linalg.eigvalsh(H), 4)}")
    print(f"[hessian] H @ ones = {np.round(H @ np.ones(4), 12)}  (adding a constant to all logits changes nothing)")
    d = rng.normal(size=4); d /= np.linalg.norm(d)
    for s in (1e-1, 1e-2):
        e1 = abs(lse(z0 + s * d) - lse(z0) - p @ (s * d))
        e2 = abs(lse(z0 + s * d) - lse(z0) - p @ (s * d) - 0.5 * (s * d) @ H @ (s * d))
        print(f"[taylor-2] |d| = {s:.0e}   1st-order error {e1:.2e}   2nd-order error {e2:.2e}")

    # 4. Chain rule for maps: J_{g o f}(x) = J_g(f(x)) J_f(x)   (2x3) @ (3x2) -> (2x2)
    g = lambda y: np.array([y[0] * y[1], y[2] - y[0]])
    print(f"[chain] J(g o f) == J_g J_f: {np.allclose(jac_num(lambda x: g(f(x)), x0), jac_num(g, f(x0)) @ J0, atol=1e-6)}")
    return


if __name__ == "__main__":
    app.run()
