# sq-meta: {"title": "nn_scratch", "emoji": "📓", "group": "Transformers & LLMs", "kind": "code", "desc": "Toolkit ch. 4: a neural network from scratch in NumPy, with RMSNorm, Adam and an LR schedule", "topic": ""}
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
    mo.md("# nn_scratch\n\nToolkit ch. 4: a neural network from scratch in NumPy, with RMSNorm, Adam and an LR schedule\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Toolkit ch. 4: a neural network from scratch in NumPy, with RMSNorm, Adam and an LR schedule."""
    import numpy as np

    rng = np.random.default_rng(0)


    class Linear:
        def __init__(self, din, dout):
            self.W = rng.normal(size=(din, dout)) * np.sqrt(2 / din)   # He init for ReLU
            self.b = np.zeros(dout)

        def forward(self, x):
            self.x = x
            return x @ self.W + self.b

        def backward(self, g):
            self.grads = [self.x.T @ g, g.sum(0)]
            return g @ self.W.T

        @property
        def params(self):
            return [self.W, self.b]


    class RMSNorm:
        def __init__(self, d, eps=1e-6):
            self.w, self.eps = np.ones(d), eps

        def forward(self, x):
            self.r = np.sqrt((x**2).mean(-1, keepdims=True) + self.eps)
            self.xh = x / self.r
            return self.xh * self.w

        def backward(self, g):
            self.grads = [(g * self.xh).sum(0)]
            gx = g * self.w                                     # gradient wrt x-hat
            return (gx - self.xh * (gx * self.xh).mean(-1, keepdims=True)) / self.r

        @property
        def params(self):
            return [self.w]


    class ReLU:
        params = []

        def forward(self, x):
            self.m = x > 0
            return x * self.m

        def backward(self, g):
            self.grads = []
            return g * self.m


    def softmax_ce(z, y):
        z = z - z.max(1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        loss = -np.log(p[np.arange(len(y)), y]).mean()
        g = p.copy()
        g[np.arange(len(y)), y] -= 1                            # dL/dz = p - onehot(y)
        return loss, g / len(y)


    class Net:
        def __init__(self, sizes):
            self.layers = []
            for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
                self.layers.append(Linear(a, b))
                if i < len(sizes) - 2:
                    self.layers += [RMSNorm(b), ReLU()]

        def forward(self, x):
            for l in self.layers:
                x = l.forward(x)
            return x

        def backward(self, g):
            for l in reversed(self.layers):
                g = l.backward(g)

        def params_grads(self):
            return [(p, g) for l in self.layers for p, g in zip(l.params, l.grads)]


    class Adam:
        def __init__(self, net, b1=0.9, b2=0.999, eps=1e-8):
            self.b1, self.b2, self.eps, self.t = b1, b2, eps, 0
            params = [p for l in net.layers for p in l.params]
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        def step(self, net, lr):
            self.t += 1
            for i, (p, g) in enumerate(net.params_grads()):
                self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g          # EMA of gradient
                self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g      # EMA of squared gradient
                mh = self.m[i] / (1 - self.b1**self.t)
                vh = self.v[i] / (1 - self.b2**self.t)
                p -= lr * mh / (np.sqrt(vh) + self.eps)


    def lr_at(step, total, peak, warmup):
        """Linear warmup, then cosine decay to 10% of peak."""
        if step < warmup:
            return peak * (step + 1) / warmup
        t = (step - warmup) / max(1, total - warmup)
        return peak * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * t)))


    def spirals(n=200, k=3):
        X, y = [], []
        for c in range(k):
            r = np.linspace(0.05, 1, n)
            t = np.linspace(c * 4, (c + 1) * 4, n) + rng.normal(scale=0.2, size=n)
            X.append(np.c_[r * np.sin(t), r * np.cos(t)])
            y += [c] * n
        return np.vstack(X), np.array(y)


    X, y = spirals()

    # Gradient check on a small net: analytic backward vs central differences.
    net = Net([2, 8, 3])
    loss, g = softmax_ce(net.forward(X[:16]), y[:16])
    net.backward(g)
    worst = 0.0
    for p, gp in net.params_grads():
        for idx in [tuple(rng.integers(s) for s in p.shape) for _ in range(5)]:
            old = p[idx]
            p[idx] = old + 1e-5; lp = softmax_ce(net.forward(X[:16]), y[:16])[0]
            p[idx] = old - 1e-5; lm = softmax_ce(net.forward(X[:16]), y[:16])[0]
            p[idx] = old
            num = (lp - lm) / 2e-5
            worst = max(worst, abs(num - gp[idx]) / max(1e-8, abs(num) + abs(gp[idx])))
    print(f"[gradcheck] worst relative error {worst:.1e}")

    # Train: Adam + warmup/cosine vs plain SGD at a fixed rate.
    for name in ("sgd", "adam"):
        rng = np.random.default_rng(1)
        net = Net([2, 64, 64, 3])
        opt = Adam(net)
        total = 600
        for step in range(total):
            loss, g = softmax_ce(net.forward(X), y)
            net.backward(g)
            if name == "adam":
                opt.step(net, lr_at(step, total, peak=1e-2, warmup=100))
            else:
                for p, gp in net.params_grads():
                    p -= 0.1 * gp
            if step in (0, 25, 75, 150, total - 1):
                acc = (net.forward(X).argmax(1) == y).mean()
                print(f"[train:{name:4s}] step {step:4d}  loss {loss:.3f}  acc {acc:.2f}")
    return


if __name__ == "__main__":
    app.run()
