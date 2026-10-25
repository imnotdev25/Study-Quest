# sq-meta: {"title": "ag_context", "emoji": "📓", "group": "Agents & Evals", "kind": "code", "desc": "Agents ch. 'Context Engineering': BM25 retrieval, budget packing, compaction, and prompt-cache cost", "topic": ""}
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
    mo.md("# ag_context\n\nAgents ch. 'Context Engineering': BM25 retrieval, budget packing, compaction, and prompt-cache cost\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Agents ch. 'Context Engineering': BM25 retrieval, budget packing, compaction, and prompt-cache cost."""
    import math
    import re
    from collections import Counter

    import numpy as np

    rng = np.random.default_rng(0)
    tokens = lambda s: max(1, len(s) // 4)          # rough rule of thumb: ~4 characters per token in English

    DOCS = [
        "Run the unit tests with make test; integration tests need DATABASE_URL set.",
        "Database migrations live in db/migrations and run with make migrate.",
        "In tests, migrations run automatically against a throwaway Postgres container.",
        "Deploys go through the release pipeline; never push to main directly.",
        "Auth uses short-lived JWTs signed with the key in the secrets manager.",
        "Code style: ruff for linting, black for formatting, type hints on public functions.",
        "The frontend is a Vite app in web/; run npm run dev for a local server.",
        "Feature flags are read from config/flags.yaml at startup.",
        "Logs ship to the central collector; use structured logging with request ids.",
        "To add a migration, run make new-migration name=<slug> and edit the generated file.",
    ]


    class BM25:
        def __init__(self, docs, k1=1.5, b=0.75):
            self.docs = [re.findall(r"[a-z0-9_]+", d.lower()) for d in docs]
            self.k1, self.b, self.N = k1, b, len(docs)
            self.avg = sum(map(len, self.docs)) / self.N
            self.df = Counter(t for d in self.docs for t in set(d))

        def scores(self, query):
            q = re.findall(r"[a-z0-9_]+", query.lower())
            out = []
            for d in self.docs:
                tf, s = Counter(d), 0.0
                for t in q:
                    if t in tf:
                        idf = math.log((self.N - self.df[t] + 0.5) / (self.df[t] + 0.5) + 1)
                        s += idf * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
                out.append(s)
            return np.array(out)


    query = "how do database migrations run in tests"
    sc = BM25(DOCS).scores(query)
    for i in sc.argsort()[::-1][:3]:
        print(f"[bm25] {sc[i]:.2f}  {DOCS[i]}")

    # Packing a context window: fixed parts first, then retrieved chunks by score, then recent history.
    budget = 900
    fixed = {"system prompt": 250, "tool definitions": 300, "AGENTS.md": 120}
    used = sum(fixed.values())
    picked = []
    for i in sc.argsort()[::-1]:
        if sc[i] > 0 and used + tokens(DOCS[i]) <= budget - 150:          # reserve 150 for recent history
            picked.append(i); used += tokens(DOCS[i])
    print(f"[pack] budget {budget}: fixed {sum(fixed.values())} + {len(picked)} retrieved chunks -> {used}, "
          f"{budget - used} left for the latest turns")

    # Compaction over a 40-step agent run. Tool outputs are long; old ones are rarely needed verbatim.
    steps = 40
    tool_out = np.clip(rng.lognormal(np.log(500), 0.8, steps), 50, 6000).astype(int)
    think = 80


    def run(policy):
        hist, total_in, peak = [], 0, 0
        base = sum(fixed.values())
        for t in range(steps):
            hist.append([think, tool_out[t]])
            if policy in ("clear", "clear+summary"):
                for h in hist[:-5]:
                    h[1] = min(h[1], 30)                                    # replace old tool output by a stub
            ctx = base + sum(a + b for a, b in hist)
            if policy == "clear+summary" and ctx > 4000:
                hist = [[400, 0]] + hist[-5:]                               # summarise all but the last 5 steps
                ctx = base + sum(a + b for a, b in hist)
            peak, total_in = max(peak, ctx), total_in + ctx
        return peak, total_in


    for pol in ("none", "clear", "clear+summary"):
        peak, tot = run(pol)
        print(f"[compact] {pol:13s}: peak context {peak:6,} tokens   input tokens summed over 40 calls {tot:9,}")

    # Prompt caching. Relative prices: uncached input 1.0, cache write 1.25, cache read 0.1 (check your provider).
    prefix, per_turn, turns = 20_000, 1_500, 30
    no_cache = sum(prefix + t * per_turn for t in range(1, turns + 1))
    with_cache = 1.25 * (prefix + per_turn) + sum(0.1 * (prefix + (t - 1) * per_turn) + 1.25 * per_turn
                                                  for t in range(2, turns + 1))
    busted = sum(1.25 * (prefix + t * per_turn) for t in range(1, turns + 1))  # a timestamp at the top changes every call
    print(f"[cache] 30 turns, 20k-token prefix: no cache {no_cache:,.0f} units | stable prefix cached {with_cache:,.0f} "
          f"({with_cache / no_cache:.0%}) | cache busted every turn {busted:,.0f} ({busted / no_cache:.0%})")
    return


if __name__ == "__main__":
    app.run()
