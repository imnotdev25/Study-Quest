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
    mo.md("# paged_kv\n\nInference chs. 'PagedAttention' and 'Prefix Optimization': fragmentation, sharing, prefix-cache hit rates\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Inference chs. 'PagedAttention' and 'Prefix Optimization': fragmentation, sharing, prefix-cache hit rates."""
    import math
    from collections import OrderedDict

    import numpy as np

    rng = np.random.default_rng(0)

    # 1. Contiguous vs paged allocation for the same requests.
    MAX_LEN, BS = 4096, 16
    lens = np.clip(rng.lognormal(np.log(700), 0.8, 256), 16, MAX_LEN).astype(int)
    used = lens.sum()
    contig = len(lens) * MAX_LEN                         # reserve max length per request
    paged = sum(math.ceil(n / BS) * BS for n in lens)    # waste only inside the last block
    print(f"[frag] {len(lens)} requests, mean length {lens.mean():.0f}, max reservation {MAX_LEN}")
    print(f"[frag] contiguous: {used / contig:5.1%} of reserved KV slots hold tokens")
    print(f"[frag] paged ({BS}-token blocks): {used / paged:5.1%}  (waste {paged - used:,} slots, < {BS} per request)")
    budget = 400_000
    print(f"[frag] requests that fit in {budget:,} KV slots: contiguous {budget // MAX_LEN}, "
          f"paged {np.searchsorted(np.cumsum(np.ceil(lens / BS) * BS), budget)}")

    # 2. Parallel sampling: n samples share the prompt's blocks (ref counts), copy-on-write at the end.
    prompt, out, n = 2000, 300, 8
    no_share = n * math.ceil((prompt + out) / BS)
    share = math.ceil(prompt / BS) + n * (math.ceil((prompt + out) / BS) - prompt // BS)
    print(f"[share] {n} samples of a {prompt}-token prompt: {no_share} blocks unshared, {share} shared "
          f"({1 - share / no_share:.0%} saved)")


    # 3. Prefix caching on multi-turn chat: block-hash LRU (vLLM style) vs exact token trie (radix style).
    def chats(n_conv=40, turns=6, n_sys=3):
        systems = [list(rng.integers(0, 50_000, 600)) for _ in range(n_sys)]
        convs = [list(systems[rng.integers(n_sys)]) for _ in range(n_conv)]
        reqs = []
        for t in range(turns):
            for c in rng.permutation(n_conv):
                convs[c] += list(rng.integers(0, 50_000, rng.integers(30, 200)))      # user turn
                reqs.append(list(convs[c]))
                convs[c] += list(rng.integers(0, 50_000, rng.integers(50, 300)))      # assistant reply
        return reqs


    reqs = chats()
    total = sum(len(r) for r in reqs)
    for cap_blocks in (2_000, 8_000, 40_000):
        lru, hit, hit_trie = OrderedDict(), 0, 0
        for r in reqs:
            h, matched = None, 0
            keys = []
            for i in range(len(r) // BS):
                h = hash((h, tuple(r[i * BS:(i + 1) * BS])))
                keys.append(h)
            for k in keys:
                if k in lru:
                    matched += BS; lru.move_to_end(k)
                else:
                    break
            hit += matched
            for k in keys:
                lru[k] = None; lru.move_to_end(k)
            while len(lru) > cap_blocks:
                lru.popitem(last=False)
        print(f"[prefix] cache {cap_blocks:6,} blocks ({cap_blocks * BS:,} tokens): hit rate {hit / total:5.1%}")
    print(f"[prefix] workload: {len(reqs)} requests, {total:,} prompt tokens; ideal (infinite cache) "
          f"is bounded by the share of each prompt seen before")
    return


if __name__ == "__main__":
    app.run()
