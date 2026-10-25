# sq-meta: {"title": "mf_integrals", "emoji": "📓", "group": "Math & Optimisation", "kind": "code", "desc": "Math I: Riemann sums, double integrals, Fubini, polar change of variables, Monte Carlo", "topic": ""}
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
    mo.md("# mf_integrals\n\nMath I: Riemann sums, double integrals, Fubini, polar change of variables, Monte Carlo\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Math I: Riemann sums, double integrals, Fubini, polar change of variables, Monte Carlo."""
    import numpy as np

    rng = np.random.default_rng(0)

    # 1. From partitions to area: integral of x^2 on [0, 1] is 1/3.
    f = lambda x: x**2
    print("[riemann] n       left error   right error  midpoint error")
    for n in (10, 100, 1000):
        x, h = np.linspace(0, 1, n + 1), 1 / n
        L, R, M = (f(x[:-1]) * h).sum(), (f(x[1:]) * h).sum(), (f((x[:-1] + x[1:]) / 2) * h).sum()
        print(f"[riemann] {n:5d}   {L - 1/3:+.2e}    {R - 1/3:+.2e}    {M - 1/3:+.2e}")

    # 2. Volume: double midpoint sum of f(x, y) = x y over the unit square (exact 1/4).
    n = 400
    c = (np.arange(n) + 0.5) / n
    Xg, Yg = np.meshgrid(c, c, indexing="ij")
    print(f"[volume] sum over {n}x{n} cells of x*y*dA = {(Xg * Yg).sum() / n**2:.6f}  (exact 0.25)")

    # 3. Fubini on the triangle 0 <= y <= x <= 1, f = x y: both orders give 1/8.
    inner_y = lambda x: x * x**2 / 2                 # integral_0^x x y dy
    inner_x = lambda y: y * (1 - y**2) / 2           # integral_y^1 x y dx
    print(f"[fubini] dy then dx: {(inner_y(c) / n).sum():.6f}   dx then dy: {(inner_x(c) / n).sum():.6f}   (exact 0.125)")

    # 4. The Gaussian integral via polar coordinates: double integral of exp(-(x^2+y^2)) = pi.
    g = np.linspace(-6, 6, 1201); h = g[1] - g[0]
    G = np.exp(-(g[:, None] ** 2 + g[None, :] ** 2)).sum() * h * h
    print(f"[gauss] grid double integral {G:.6f} vs pi {np.pi:.6f}  -> integral of exp(-x^2) = sqrt(pi) = {np.sqrt(G):.6f}")

    # 5. Monte Carlo: area of the unit disk = pi; error falls like 1/sqrt(N).
    for N in (100, 10_000, 1_000_000):
        u = rng.uniform(-1, 1, (N, 2))
        est = 4 * ((u**2).sum(1) < 1).mean()
        print(f"[mc] N = {N:9,d}   pi estimate {est:.4f}   error {abs(est - np.pi):.4f}   1/sqrt(N) = {N**-0.5:.4f}")

    # 6. Change of variables in a probability: P(X^2 + Y^2 < 1) for independent N(0,1) is 1 - e^{-1/2}.
    z = rng.normal(size=(1_000_000, 2))
    print(f"[polar] P(X^2+Y^2<1): simulated {((z**2).sum(1) < 1).mean():.4f}   exact {1 - np.exp(-0.5):.4f}")
    return


if __name__ == "__main__":
    app.run()
