# sq-meta: {"title": "spec_decode", "emoji": "📓", "group": "Inference", "kind": "code", "desc": "Inference ch. 'Speculative Decoding': exactness of speculative sampling, expected speedup,", "topic": ""}
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
    mo.md("# spec_decode\n\nInference ch. 'Speculative Decoding': exactness of speculative sampling, expected speedup,\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Inference ch. 'Speculative Decoding': exactness of speculative sampling, expected speedup,
    tree drafts, and model-free drafting (prompt lookup vs suffix-tree style) on an agent-like workload."""
    import random
    from collections import defaultdict

    import numpy as np

    rng = np.random.default_rng(0)

    # 1. Speculative sampling is exact: accept x~q with prob min(1, p/q), else resample from (p - q)+.
    V = 8
    p = rng.dirichlet(np.ones(V))                       # target
    q = 0.6 * p + 0.4 * rng.dirichlet(np.ones(V))       # draft, close but not equal
    N, out, acc = 200_000, np.zeros(V), 0
    resid = np.maximum(p - q, 0); resid /= resid.sum()
    for x in rng.choice(V, N, p=q):
        if rng.random() < min(1, p[x] / q[x]):
            out[x] += 1; acc += 1
        else:
            out[rng.choice(V, p=resid)] += 1
    tv = 0.5 * np.abs(p - q).sum()
    print(f"[exact] TV(empirical, target) = {0.5 * np.abs(out / N - p).sum():.4f}  (sampling noise ~0.003)")
    print(f"[exact] acceptance {acc / N:.4f}  vs  1 - TV(p, q) = {1 - tv:.4f}")

    # 2. Expected tokens per target pass with k drafts and per-token acceptance a: (1 - a^(k+1)) / (1 - a).
    print("[speedup] a     k=1   k=3   k=5   k=8    best k (draft cost c = 0.05 of target)")
    for a in (0.6, 0.75, 0.9):
        tok = {k: (1 - a ** (k + 1)) / (1 - a) for k in (1, 3, 5, 8)}
        sp = {k: (1 - a ** (k + 1)) / (1 - a) / (1 + 0.05 * k) for k in range(1, 16)}
        kb = max(sp, key=sp.get)
        print(f"[speedup] {a:.2f}  " + "  ".join(f"{tok[k]:4.2f}" for k in (1, 3, 5, 8))
              + f"    k={kb}, speedup {sp[kb]:.2f}x")

    # 3. Tree drafts: top-2 candidates at each of 3 levels (Medusa / EAGLE-2 style). If the target token is
    #    the draft's top-1 with prob 0.6 and its top-2 with prob 0.2, a chain accepts on top-1 only.
    a1, a2, depth = 0.6, 0.2, 3
    chain = sum(a1**i for i in range(1, depth + 1))
    tree = sum((a1 + a2) ** i for i in range(1, depth + 1))
    print(f"[tree] expected accepted drafts, depth {depth}: chain {chain:.2f}  tree (2 per level, "
          f"{2 + 4 + 8} nodes) {tree:.2f}")

    # 4. Model-free drafting on an agent-like workload (characters as tokens, greedy target = the text).
    random.seed(0)
    tools = ["search", "read_file", "run_tests"]


    def episode():
        s = ""
        for _ in range(6):
            t = random.choice(tools)
            arg = random.choice(["src/app.py", "tests/test_api.py", "README.md", "pricing bug", "login flow"])
            s += f'Thought: I should call {t}.\n{{"tool": "{t}", "args": {{"query": "{arg}"}}}}\n'
            s += random.choice(["Result: ok\n", "Result: 2 failures\n", "Result: found 3 matches\n"])
        return s


    history = [episode() for _ in range(30)]


    def run(text, draft_fn, k_max):
        """Count target passes to emit `text` when each pass verifies a draft and adds one free token."""
        i, passes, accepted = 0, 0, 0
        while i < len(text):
            d = draft_fn(text[:i], k_max)
            n = 0
            while n < len(d) and i + n < len(text) and d[n] == text[i + n]:
                n += 1
            i += n + 1; passes += 1; accepted += n
        return passes, accepted


    def prompt_lookup(ctx, k, n=3):
        """Find the last earlier occurrence of the final n tokens in this request; copy what followed."""
        if len(ctx) < n:
            return ""
        j = ctx.rfind(ctx[-n:], 0, len(ctx) - 1)
        return ctx[j + n:j + n + k] if j >= 0 else ""


    class SuffixIndex:
        """Suffix-decoding style: index past outputs by context; draft length grows with match length."""

        def __init__(self, texts, n=8):
            self.n, self.next = n, defaultdict(lambda: defaultdict(int))
            for t in texts:
                for i in range(len(t)):
                    for m in range(1, n + 1):
                        if i - m >= 0:
                            self.next[t[i - m:i]][t[i]] += 1

        def draft(self, ctx, k_max):
            out = ""
            for _ in range(k_max):
                c = ctx + out
                for m in range(self.n, 0, -1):          # longest matching context wins
                    nxt = self.next.get(c[-m:])
                    if nxt:
                        tok, cnt = max(nxt.items(), key=lambda kv: kv[1])
                        if cnt / sum(nxt.values()) < 0.5 or m < 3:   # stop when the continuation is uncertain
                            return out
                        out += tok
                        break
                else:
                    return out
            return out


    idx = SuffixIndex(history[:20])
    test = history[20:]
    total = sum(len(t) for t in test)
    for name, fn, k in (("none", lambda c, k: "", 0), ("prompt lookup", prompt_lookup, 8),
                        ("suffix index", idx.draft, 32)):
        passes = sum(run(t, fn, k)[0] for t in test)
        print(f"[model-free] {name:13s}: {total / passes:5.2f} tokens per target pass  ({passes} passes for {total} tokens)")
    return


if __name__ == "__main__":
    app.run()
