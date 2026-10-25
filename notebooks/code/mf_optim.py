# sq-meta: {"title": "mf_optim", "emoji": "📓", "group": "Math & Optimisation", "kind": "code", "desc": "Math I: unconstrained and constrained optimisation, with the two constrained problems behind", "topic": ""}
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
    mo.md("# mf_optim\n\nMath I: unconstrained and constrained optimisation, with the two constrained problems behind\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Math I: unconstrained and constrained optimisation, with the two constrained problems behind
    softmax (max entropy) and RLHF (KL-regularised reward maximisation)."""
    import numpy as np

    rng = np.random.default_rng(0)

    # 1. Gradient descent on f(x) = 1/2 x^T A x - b^T x. Speed is set by the condition number kappa.
    print("[gd] kappa   GD steps to 1e-6   theory ~ kappa*ln(1e6)   Newton steps")
    for kappa in (10, 100, 1000):
        A = np.diag(np.linspace(1, kappa, 10)); b = np.ones(10); x_star = np.linalg.solve(A, b)
        x = np.zeros(10)
        for k in range(1, 100_000):
            x -= (A @ x - b) / kappa                          # step 1/L, L = largest eigenvalue
            if np.linalg.norm(x - x_star) < 1e-6 * np.linalg.norm(x_star):
                break
        xn = np.zeros(10) - np.linalg.solve(A, A @ np.zeros(10) - b)   # one Newton step
        print(f"[gd] {kappa:5d}   {k:8d}            {kappa * np.log(1e6):8.0f}              "
              f"{1 if np.allclose(xn, x_star) else '>1'}")

    # 2. Second-derivative test: f = x^3 - 3x + y^2 has critical points (1, 0) and (-1, 0).
    for pt in ((1.0, 0.0), (-1.0, 0.0)):
        ev = np.linalg.eigvalsh(np.diag([6 * pt[0], 2.0]))
        kind = "minimum" if (ev > 0).all() else "saddle" if (ev < 0).any() and (ev > 0).any() else "maximum"
        print(f"[hessian] critical point {pt}: Hessian eigenvalues {ev} -> {kind}")

    # 3. Max entropy subject to sum p = 1 and E_p[energy] = mu  ->  Gibbs / softmax p ~ exp(-lam * energy).
    E, mu = np.array([0.0, 1.0, 2.0, 3.0]), 1.0
    gibbs = lambda lam: np.exp(-lam * E) / np.exp(-lam * E).sum()
    lo, hi = -20.0, 20.0
    for _ in range(100):                                       # mean energy decreases in lam: bisect
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gibbs(mid) @ E > mu else (lo, mid)
    p = gibbs(lo)
    H = lambda q: -(q * np.log(q)).sum()
    null = np.linalg.svd(np.vstack([np.ones(4), E]))[2][2:]   # directions keeping both constraints
    worse = all(H(p + 0.01 * (rng.normal(size=2) @ null)) < H(p) for _ in range(1000))
    print(f"[maxent] lambda = {lo:.4f}, p = {np.round(p, 4)}, mean energy {p @ E:.4f}, entropy {H(p):.4f}")
    print(f"[maxent] 1000 random feasible perturbations all have lower entropy: {worse}")

    # 4. RLHF objective: max_pi E_pi[r] - beta KL(pi || pi_ref)  ->  pi* ~ pi_ref exp(r / beta).
    K, beta = 6, 0.5
    pi_ref, r = rng.dirichlet(np.ones(K)), rng.normal(size=K)
    J = lambda pi: pi @ r - beta * (pi * np.log(pi / pi_ref)).sum()
    pi_star = pi_ref * np.exp(r / beta); pi_star /= pi_star.sum()
    best_random = max(J(rng.dirichlet(np.ones(K))) for _ in range(20_000))
    print(f"[rlhf] J(pi*) = {J(pi_star):.4f} = beta log Z = {beta * np.log((pi_ref * np.exp(r / beta)).sum()):.4f};"
          f" best of 20,000 random policies {best_random:.4f}")

    # 5. KKT: min |x - c|^2 s.t. x >= 0. Solution max(c, 0); multipliers mu = 2 max(-c, 0) >= 0, mu * x = 0.
    c = np.array([1.5, -0.7, 0.2, -2.0])
    x = np.zeros(4)
    for _ in range(200):
        x = np.maximum(x - 0.1 * 2 * (x - c), 0)                 # projected gradient descent
    mu_kkt = 2 * (x - c)                                         # stationarity: 2(x - c) - mu = 0
    print(f"[kkt] projected GD x = {np.round(x, 4)} (= max(c,0)), multipliers {np.round(mu_kkt, 4) + 0.0}, "
          f"complementary slackness mu*x = {np.round(mu_kkt * x, 6) + 0.0}")
    return


if __name__ == "__main__":
    app.run()
