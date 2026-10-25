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
    mo.md("# mini_engine\n\nInference ch. 'Build a Mini Inference Engine': paged KV, continuous batching with a token budget,\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Inference ch. 'Build a Mini Inference Engine': paged KV, continuous batching with a token budget,
    chunked prefill, prefix caching, preemption by recompute, and streaming. Checked against naive greedy."""
    import math
    from collections import OrderedDict, deque

    import torch

    from tiny_gpt import Cfg, TinyGPT, generate_naive


    class BlockPool:
        """Fixed pool of KV blocks. Blocks with ref 0 that hold a cached prefix stay reusable (LRU)."""

        def __init__(self, cfg, n_blocks, bs):
            shape = (cfg.n_layers, n_blocks, bs, cfg.n_kv, cfg.hd)
            self.k, self.v = torch.zeros(shape, dtype=cfg.dtype), torch.zeros(shape, dtype=cfg.dtype)
            self.free = deque(range(n_blocks))
            self.ref = [0] * n_blocks
            self.hash_of = [None] * n_blocks
            self.cached = {}                 # prefix hash -> block
            self.evictable = OrderedDict()   # cached blocks nobody is using, oldest first
            self.evictions = 0

        def available(self):
            return len(self.free) + len(self.evictable)

        def alloc(self):
            if self.free:
                b = self.free.popleft()
            else:
                b, _ = self.evictable.popitem(last=False)
                del self.cached[self.hash_of[b]]
                self.hash_of[b] = None
                self.evictions += 1
            self.ref[b] = 1
            return b

        def release(self, b):
            self.ref[b] -= 1
            if self.ref[b] == 0:
                if self.hash_of[b] is not None:
                    self.evictable[b] = None
                else:
                    self.free.append(b)

        def lookup(self, h):
            b = self.cached.get(h)
            if b is not None:
                if self.ref[b] == 0:
                    del self.evictable[b]
                self.ref[b] += 1
            return b

        def register(self, b, h):
            if h not in self.cached and self.hash_of[b] is None:
                self.cached[h], self.hash_of[b] = b, h


    class Seq:
        def __init__(self, rid, prompt, max_new):
            self.id, self.tokens, self.max_new = rid, list(prompt), max_new
            self.out, self.blocks, self.computed, self.registered = [], [], 0, 0


    class Engine:
        def __init__(self, model, n_blocks=24, bs=8, budget=32, max_seqs=8, prefix_cache=True):
            self.m, self.bs, self.budget, self.max_seqs, self.pc = model, bs, budget, max_seqs, prefix_cache
            self.pool = BlockPool(model.c, n_blocks, bs)
            self.waiting, self.running, self.done = deque(), [], {}
            self.stats = dict(steps=0, computed=0, prefix_hit=0, preempted=0, max_batch=0)

        def add(self, rid, prompt, max_new):
            self.waiting.append(Seq(rid, prompt, max_new))

        def _hash(self, seq, i):                     # identity of block i = all tokens up to its end
            return hash(tuple(seq.tokens[:(i + 1) * self.bs]))

        def _grow(self, seq, n):
            need = math.ceil((seq.computed + n) / self.bs) - len(seq.blocks)
            if need > self.pool.available():
                return False
            seq.blocks += [self.pool.alloc() for _ in range(need)]
            return True

        def _reset(self, seq):
            for b in reversed(seq.blocks):           # tail blocks become the first to be evicted
                self.pool.release(b)
            seq.blocks, seq.computed, seq.registered = [], 0, 0

        def _match_prefix(self, seq):
            if not self.pc:
                return
            for i in range((len(seq.tokens) - 1) // self.bs):   # always leave >= 1 token to compute
                b = self.pool.lookup(self._hash(seq, i))
                if b is None:
                    break
                seq.blocks.append(b)
            seq.computed = seq.registered = len(seq.blocks)
            seq.computed *= self.bs
            self.stats["prefix_hit"] += seq.computed

        def step(self):
            budget, sched = self.budget, []
            i = 0
            while i < len(self.running) and budget > 0:            # running sequences first, oldest first
                seq = self.running[i]
                n = min(len(seq.tokens) - seq.computed, budget)
                while not self._grow(seq, n):                      # out of blocks: preempt the newest
                    victim = self.running.pop()
                    self._reset(victim)
                    self.waiting.appendleft(victim)
                    self.stats["preempted"] += 1
                    if victim is seq:
                        break
                else:
                    sched.append((seq, n)); budget -= n; i += 1
                    continue
                break
            while self.waiting and budget > 0 and len(self.running) < self.max_seqs:
                seq = self.waiting[0]
                self._match_prefix(seq)
                n = min(len(seq.tokens) - seq.computed, budget)
                if not self._grow(seq, n):
                    self._reset(seq)
                    break
                self.running.append(self.waiting.popleft())
                sched.append((seq, n)); budget -= n
            if not sched:
                return {}

            bs, pool = self.bs, self.pool
            starts = [s.computed for s, _ in sched]

            def kv_get(i, l):
                s, p = sched[i][0], starts[i]
                idx = torch.tensor(s.blocks)
                return (pool.k[l, idx].flatten(0, 1)[:p], pool.v[l, idx].flatten(0, 1)[:p])

            def kv_put(i, l, k, v):
                s = sched[i][0]
                for j in range(len(k)):
                    p = starts[i] + j
                    pool.k[l, s.blocks[p // bs], p % bs] = k[j]
                    pool.v[l, s.blocks[p // bs], p % bs] = v[j]

            segs = [torch.tensor(s.tokens[s.computed:s.computed + n]) for s, n in sched]
            logits = self.m(segs, starts, kv_get, kv_put)
            self.stats["steps"] += 1
            self.stats["computed"] += sum(n for _, n in sched)
            self.stats["max_batch"] = max(self.stats["max_batch"], len(sched))

            streamed = {}
            for (s, n), lg in zip(sched, logits):
                s.computed += n
                while self.pc and (s.registered + 1) * bs <= s.computed:
                    pool.register(s.blocks[s.registered], self._hash(s, s.registered))
                    s.registered += 1
                if s.computed == len(s.tokens):                   # caught up: sample the next token
                    tok = int(lg[-1].argmax())
                    s.tokens.append(tok); s.out.append(tok); streamed[s.id] = tok
                    if len(s.out) == s.max_new:
                        self.running.remove(s); self._reset(s); self.done[s.id] = s.out
            return streamed

        def busy(self):
            return bool(self.waiting or self.running)


    model = TinyGPT(Cfg())
    g = torch.Generator().manual_seed(1)
    system = torch.randint(0, 64, (24,), generator=g).tolist()          # shared system prompt
    reqs = [(r, system + torch.randint(0, 64, (int(torch.randint(4, 13, (1,), generator=g)),), generator=g).tolist(),
             int(torch.randint(12, 25, (1,), generator=g))) for r in range(8)]
    reference = {r: generate_naive(model, p, n) for r, p, n in reqs}

    for pc in (False, True):
        eng = Engine(model, n_blocks=24, bs=8, budget=32, prefix_cache=pc)
        stream0 = []
        for r, p, n in reqs[:4]:
            eng.add(r, p, n)
        step = 0
        while eng.busy():
            if step in (3, 6):                                           # later arrivals join mid-flight
                for r, p, n in reqs[4:6] if step == 3 else reqs[6:]:
                    eng.add(r, p, n)
            out = eng.step()
            if 0 in out and len(stream0) < 6:
                stream0.append(f"s{step}:{out[0]}")
            step += 1
        ok = all(eng.done[r] == reference[r] for r, _, _ in reqs)
        st = eng.stats
        print(f"[engine] prefix_cache={pc!s:5}  matches naive greedy: {ok}  steps {st['steps']}  "
              f"tokens computed {st['computed']}  prefix hits {st['prefix_hit']}  preemptions {st['preempted']}  "
              f"evictions {eng.pool.evictions}  max batch {st['max_batch']}")
    print(f"[engine] streaming, request 0 (step:token): {' '.join(stream0)} ...")
    total_prompt = sum(len(p) for _, p, _ in reqs)
    total_out = sum(n for _, _, n in reqs)
    print(f"[engine] workload: {len(reqs)} requests, {total_prompt} prompt + {total_out} output tokens, "
          f"KV pool 24 blocks x 8 = 192 token slots")
    return


if __name__ == "__main__":
    app.run()
