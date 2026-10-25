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
    mo.md("# tensors\n\nToolkit ch. 3: PyTorch tensor operations, plus seven transformer-flavoured tasks\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Toolkit ch. 3: PyTorch tensor operations, plus seven transformer-flavoured tasks."""
    import torch

    torch.manual_seed(0)

    # Creation and special tensors.
    print("[create]", torch.arange(0, 10, 3).tolist(), torch.linspace(0, 1, 5).tolist())
    print("[create] eye(3) diag:", torch.eye(3).diagonal().tolist(), " zeros_like:", torch.zeros_like(torch.eye(2)).shape)

    # Strides: view needs contiguous memory, reshape copies when it must.
    x = torch.arange(12).reshape(3, 4)
    xt = x.T
    print(f"[stride] x {tuple(x.shape)} stride {x.stride()} | x.T stride {xt.stride()} contiguous={xt.is_contiguous()}")
    try:
        xt.view(12)
    except RuntimeError:
        print("[stride] x.T.view(12) -> RuntimeError; x.T.reshape(12) ->", xt.reshape(12)[:6].tolist())

    # squeeze / unsqueeze / flatten.
    v = torch.randn(4)
    print(f"[shape] unsqueeze(0) {tuple(v.unsqueeze(0).shape)}  unsqueeze(1) {tuple(v.unsqueeze(1).shape)}"
          f"  squeeze {tuple(v[None, :, None].squeeze().shape)}  flatten {tuple(torch.randn(2, 3, 4).flatten(1).shape)}")

    # cat vs stack.
    a, b = torch.randn(2, 3), torch.randn(2, 3)
    print(f"[join] cat dim0 {tuple(torch.cat([a, b]).shape)}  stack dim0 {tuple(torch.stack([a, b]).shape)}")

    # Indexing: gather picks the logit of each target token -> cross-entropy by hand.
    logits, y = torch.randn(4, 10), torch.tensor([3, 0, 9, 1])
    nll = -(logits.log_softmax(-1).gather(1, y[:, None]).squeeze(1)).mean()
    print(f"[gather] manual CE {nll:.5f}  F.cross_entropy {torch.nn.functional.cross_entropy(logits, y):.5f}")

    # Batched matmul broadcasts the batch dims: (2,1,3,4) @ (5,4,6) -> (2,5,3,6).
    print("[matmul]", tuple((torch.randn(2, 1, 3, 4) @ torch.randn(5, 4, 6)).shape))

    # Seven tasks.
    T = 5
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))                        # 1 causal mask
    onehot = torch.nn.functional.one_hot(torch.tensor([2, 0, 1]), 4)              # 2 one-hot
    outer = torch.einsum("bi,bj->bij", torch.randn(3, 4), torch.randn(3, 2))       # 3 batched outer product
    z = torch.randn(2, 8)
    kth = z.topk(3, dim=-1).values[:, -1:]
    topk = z.masked_fill(z < kth, float("-inf"))                                 # 4 top-k logits mask
    lens = torch.tensor([3, 5])
    pad = torch.arange(T)[None, :] < lens[:, None]                               # 5 padding mask
    Q, K, V = (torch.randn(2, 4, T, 8) for _ in range(3))
    att = torch.einsum("bhqd,bhkd->bhqk", Q, K) / 8**0.5
    att = att.masked_fill(~causal, float("-inf")).softmax(-1)
    out = torch.einsum("bhqk,bhkd->bhqd", att, V)                                 # 6 attention with einsum
    ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
    kv = torch.randn(2, 2, T, 8).repeat_interleave(2, dim=1)                      # 7 GQA: 2 kv heads -> 4
    print(f"[tasks] causal row2 {causal[2].int().tolist()}  onehot {onehot[0].tolist()}  outer {tuple(outer.shape)}")
    print(f"[tasks] top-k kept per row {(topk > float('-inf')).sum(1).tolist()}  pad {pad.int().tolist()}")
    print(f"[tasks] einsum attention == SDPA: {torch.allclose(out, ref, atol=1e-5)}  GQA kv {tuple(kv.shape)}")
    return


if __name__ == "__main__":
    app.run()
