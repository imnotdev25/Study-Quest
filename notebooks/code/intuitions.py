# sq-meta: {"title": "intuitions", "emoji": "📓", "group": "Math & Optimisation", "kind": "code", "desc": "Toolkit ch. 2: dot products, softmax temperature, broadcasting", "topic": ""}
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
    mo.md("# intuitions\n\nToolkit ch. 2: dot products, softmax temperature, broadcasting\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Toolkit ch. 2: dot products, softmax temperature, broadcasting."""
    import numpy as np

    rng = np.random.default_rng(1)

    # Dot product vs cosine: the dot product also rewards length.
    q = np.array([1.0, 2.0, 0.0])
    K = np.array([[1.0, 2.0, 0.1],    # same direction
                  [3.0, 6.0, 0.0],    # same direction, 3x longer
                  [-2.0, 1.0, 0.0]])  # orthogonal
    dots = K @ q
    cos = dots / (np.linalg.norm(K, axis=1) * np.linalg.norm(q))
    print(f"[dot] q.k    = {np.round(dots, 2)}")
    print(f"[dot] cosine = {np.round(cos, 3)}")


    def softmax(z, T=1.0):
        z = np.asarray(z) / T
        e = np.exp(z - z.max())          # subtract max: same result, no overflow
        return e / e.sum()


    z = np.array([2.0, 1.0, 0.1])
    for T in (0.25, 1.0, 4.0):
        print(f"[softmax] T={T:<4}  {np.round(softmax(z, T), 3)}")
    print(f"[softmax] shift-invariant: {np.allclose(softmax(z), softmax(z + 1000))}")

    # Why attention divides by sqrt(d): q.k for random unit-variance vectors has variance d.
    for d in (16, 256):
        s = (rng.normal(size=(10_000, d)) * rng.normal(size=(10_000, d))).sum(1)
        print(f"[scale] d={d:3d}  std(q.k)={s.std():6.2f}  std(q.k/sqrt d)={(s / np.sqrt(d)).std():.2f}")

    # Broadcasting: all pairwise squared distances, no loops.
    A, B = rng.normal(size=(5, 3)), rng.normal(size=(4, 3))
    D1 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)              # (5,1,3)-(1,4,3) -> (5,4,3)
    D2 = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T  # (5,1)+(1,4)-(5,4)
    print(f"[broadcast] D shape {D1.shape}, two methods agree: {np.allclose(D1, D2)}")
    return


if __name__ == "__main__":
    app.run()
