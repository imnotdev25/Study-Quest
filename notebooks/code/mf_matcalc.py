# sq-meta: {"title": "mf_matcalc", "emoji": "📓", "group": "Math & Optimisation", "kind": "code", "desc": "Math I: matrix calculus identities, each checked against autograd", "topic": ""}
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
    mo.md("# mf_matcalc\n\nMath I: matrix calculus identities, each checked against autograd\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Math I: matrix calculus identities, each checked against autograd."""
    import torch

    torch.manual_seed(0)
    D = torch.float64


    def grad_of(fn, X):
        X = X.clone().requires_grad_(True)
        fn(X).backward()
        return X.grad


    n, m = 4, 3
    a, b, x = torch.randn(n, dtype=D), torch.randn(m, dtype=D), torch.randn(n, dtype=D)
    X = torch.randn(n, m, dtype=D)
    A = torch.randn(n, n, dtype=D)
    B = torch.randn(m, n, dtype=D)
    S = torch.randn(n, n, dtype=D); S = S @ S.T + n * torch.eye(n, dtype=D)       # symmetric positive definite
    Xd, Y, W = torch.randn(20, n, dtype=D), torch.randn(20, 2, dtype=D), torch.randn(n, 2, dtype=D)

    checks = {
        "d(a^T X b)/dX       = a b^T": (grad_of(lambda X: a @ X @ b, X), torch.outer(a, b)),
        "d(x^T A x)/dx       = (A + A^T) x": (grad_of(lambda x: x @ A @ x, x), (A + A.T) @ x),
        "d tr(A X B)/dX      = A^T B^T": (grad_of(lambda X: torch.trace(A @ X @ B), X), A.T @ B.T),
        "d log det S/dS      = S^-T": (grad_of(torch.logdet, S), torch.linalg.inv(S).T),
        "d tr(S^-1 A)/dS     = -S^-T A^T S^-T": (grad_of(lambda S: torch.trace(torch.linalg.inv(S) @ A), S),
                                                 -torch.linalg.inv(S).T @ A.T @ torch.linalg.inv(S).T),
        "d |XW - Y|_F^2/dW   = 2 X^T (XW - Y)": (grad_of(lambda W: ((Xd @ W - Y) ** 2).sum(), W), 2 * Xd.T @ (Xd @ W - Y)),
    }
    for name, (auto, formula) in checks.items():
        print(f"[matcalc] {name:38s} {torch.allclose(auto, formula)}")

    z = torch.randn(5, dtype=D)
    p = z.softmax(0)
    Jsm = torch.autograd.functional.jacobian(lambda z: z.softmax(0), z)
    print(f"[matcalc] softmax Jacobian = diag(p) - p p^T       {torch.allclose(Jsm, torch.diag(p) - torch.outer(p, p))}")

    # Setting the least-squares gradient to zero gives the normal equations X^T X W = X^T Y.
    W_star = torch.linalg.solve(Xd.T @ Xd, Xd.T @ Y)
    print(f"[lstsq] normal equations == torch.linalg.lstsq: {torch.allclose(W_star, torch.linalg.lstsq(Xd, Y).solution)}, "
          f"gradient there {grad_of(lambda W: ((Xd @ W - Y) ** 2).sum(), W_star).abs().max():.1e}")
    return


if __name__ == "__main__":
    app.run()
