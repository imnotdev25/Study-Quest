#!/usr/bin/env python3
"""Write marimo notebooks into study_app/notebooks/ and a manifest at public/content/notebooks.json.

* interactive/: hand-written notebooks with sliders and plots
* code/:        every runnable llm_book/code script as an editable notebook (one cell = the script)

Usage: ../.venv/bin/python scripts/build_notebooks.py
"""
import json
import pathlib
import re
import textwrap

APP = pathlib.Path(__file__).resolve().parents[1]
ROOT = APP.parent
NB = APP / "notebooks"
HEAD = 'import marimo\n\n__generated_with = "0.25.1"\napp = marimo.App(width="medium")\n\n'
TAIL = '\n\nif __name__ == "__main__":\n    app.run()\n'


def cell(args, body, returns=()):
    sig = ", ".join(sorted(args))
    ret = f"    return ({', '.join(returns)},)\n" if returns else "    return\n"
    return f"@app.cell\ndef _({sig}):\n{textwrap.indent(textwrap.dedent(body).strip(), '    ')}\n{ret}\n\n"


IMPORTS = cell([], """
import marimo as mo
import numpy as np
import matplotlib.pyplot as plt
""", ["mo", "np", "plt"])

INTERACTIVE = {}

INTERACTIVE["bandit_casino"] = ("🎰", "Bandit Casino", "Play three slot machines yourself, then race epsilon-greedy, greedy and UCB.", "bandit", [
    IMPORTS,
    cell(["mo"], '''
mo.md("""
# 🎰 The Bandit Casino
Three machines, hidden payouts. Pull arms below and try to find the best one. Every pull you spend on a bad
arm is **regret**. Then scroll down and watch three algorithms play the same game.
""")
'''),
    cell(["mo"], "get_pulls, set_pulls = mo.state([])", ["get_pulls", "set_pulls"]),
    cell(["mo", "np", "set_pulls"], '''
casino_means = [0.30, 0.55, 0.45]
_rng = np.random.default_rng()
def _pull(i):
    set_pulls(lambda p: p + [(i, float(_rng.normal(casino_means[i], 0.3)))])
    return 0
arms = mo.ui.array([mo.ui.button(label=f"Pull {c}", on_click=lambda _, i=i: _pull(i)) for i, c in enumerate("ABC")])
reset = mo.ui.button(label="Reset", on_click=lambda _: set_pulls([]) or 0)
mo.hstack([*arms, reset], justify="start")
''', ["arms", "casino_means", "reset"]),
    cell(["casino_means", "get_pulls", "mo", "np"], '''
_p = get_pulls()
_rows = []
for _i, _c in enumerate("ABC"):
    _r = [r for a, r in _p if a == _i]
    _rows.append({"arm": _c, "pulls": len(_r), "mean reward": round(float(np.mean(_r)), 3) if _r else "-"})
_regret = sum(max(casino_means) - casino_means[a] for a, _ in _p)
_msg = "Pull some arms!" if not _p else f"**{len(_p)} pulls, regret so far {_regret:.2f}.**"
if len(_p) >= 20:
    _msg += f" The true means were {casino_means}: arm B was best."
mo.vstack([mo.ui.table(_rows, selection=None), mo.md(_msg)])
'''),
    cell(["mo"], '''
eps = mo.ui.slider(0.0, 0.5, step=0.01, value=0.1, label="epsilon")
steps = mo.ui.slider(200, 5000, step=100, value=2000, label="steps")
runs = mo.ui.slider(10, 300, step=10, value=100, label="runs to average")
mo.hstack([eps, steps, runs])
''', ["eps", "runs", "steps"]),
    cell(["eps", "np", "plt", "runs", "steps"], '''
_means = np.array([0.30, 0.55, 0.45])
def _play(kind, rng):
    Q, N, opt = np.zeros(3), np.zeros(3), np.zeros(steps.value)
    for t in range(steps.value):
        if kind == "ucb":
            a = int(np.argmax(Q + 2 * np.sqrt(np.log(t + 1) / (N + 1e-9)))) if t >= 3 else t
        elif kind == "eps" and rng.random() < eps.value:
            a = int(rng.integers(3))
        else:
            a = int(np.argmax(Q))
        r = rng.normal(_means[a], 0.3)
        N[a] += 1; Q[a] += (r - Q[a]) / N[a]; opt[t] = a == 1
    return opt
_fig, _ax = plt.subplots(figsize=(7, 3.2))
for _kind, _lab in (("greedy", "greedy"), ("eps", f"epsilon-greedy ({eps.value})"), ("ucb", "UCB")):
    _rng = np.random.default_rng(0)
    _ax.plot(np.mean([_play(_kind, _rng) for _ in range(runs.value)], 0), label=_lab)
_ax.set_xlabel("step"); _ax.set_ylabel("% optimal arm"); _ax.legend(); _ax.set_ylim(0, 1)
_fig
'''),
])

INTERACTIVE["softmax_temperature"] = ("🌡️", "Softmax and Temperature", "See how temperature reshapes next-token probabilities and entropy.", "softmax", [
    IMPORTS,
    cell(["mo"], '''
logits_text = mo.ui.text(value="2.0, 1.0, 0.1, -1.0, 3.0", label="logits")
temp = mo.ui.slider(0.05, 5.0, step=0.05, value=1.0, label="temperature")
top_p = mo.ui.slider(0.05, 1.0, step=0.05, value=1.0, label="top-p")
mo.vstack([mo.md("# 🌡️ Softmax, temperature and top-p"), logits_text, mo.hstack([temp, top_p])])
''', ["logits_text", "temp", "top_p"]),
    cell(["logits_text", "mo", "np", "plt", "temp", "top_p"], '''
_z = np.array([float(x) for x in logits_text.value.split(",") if x.strip()])
_p = np.exp((_z - _z.max()) / temp.value); _p /= _p.sum()
_order = np.argsort(-_p); _keep = _order[:np.searchsorted(np.cumsum(_p[_order]), top_p.value) + 1]
_q = np.zeros_like(_p); _q[_keep] = _p[_keep]; _q /= _q.sum()
_H = -(_q[_q > 0] * np.log(_q[_q > 0])).sum()
_fig, _ax = plt.subplots(figsize=(7, 3))
_ax.bar(range(len(_z)), _p, color="#c7d2fe", label="softmax(z / T)")
_ax.bar(range(len(_z)), _q, color="#6366f1", width=0.5, label="after top-p")
_ax.set_xticks(range(len(_z)), [f"tok{i}\\nz={v:g}" for i, v in enumerate(_z)]); _ax.set_ylim(0, 1); _ax.legend()
mo.vstack([_fig, mo.md(f"Entropy **{_H:.3f} nats** (max {np.log(len(_z)):.3f}). Low T: argmax. High T: uniform.")])
'''),
])

INTERACTIVE["ppo_clip"] = ("✂️", "PPO Clipping", "Drag epsilon and see where PPO's gradient switches off.", "PPO", [
    IMPORTS,
    cell(["mo"], '''
clip = mo.ui.slider(0.05, 0.5, step=0.01, value=0.2, label="clip epsilon")
adv = mo.ui.dropdown({"A > 0 (good action)": 1.0, "A < 0 (bad action)": -1.0}, value="A > 0 (good action)", label="advantage")
mo.vstack([mo.md("# ✂️ The PPO clipped objective"), mo.hstack([clip, adv])])
''', ["adv", "clip"]),
    cell(["adv", "clip", "mo", "np", "plt"], '''
_r = np.linspace(0.3, 2.0, 400); _A = adv.value; _e = clip.value
_obj = np.minimum(_r * _A, np.clip(_r, 1 - _e, 1 + _e) * _A)
_grad = np.gradient(_obj, _r)
_fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(9, 3))
_a1.plot(_r, _r * _A, "--", color="gray", label="r A (unclipped)"); _a1.plot(_r, _obj, lw=2.5, label="PPO objective")
_a1.axvspan(1 - _e, 1 + _e, color="#e0e7ff"); _a1.set_xlabel("ratio r = pi_new / pi_old"); _a1.legend(fontsize=8)
_a2.plot(_r, _grad, color="#dc2626"); _a2.set_title("d objective / d r"); _a2.set_xlabel("r")
mo.vstack([_fig, mo.md("Shaded band: the trust region. Outside it on the side that would *help*, the gradient is 0. "
                       "On the side that would hurt (A < 0, r > 1+eps), it stays on: PPO keeps fixing mistakes.")])
'''),
])

INTERACTIVE["kv_cache_calculator"] = ("🧮", "KV Cache Calculator", "How many users fit on one GPU? Weights, KV cache and precision.", "KV cache", [
    IMPORTS,
    cell(["mo"], '''
preset = mo.ui.dropdown({"Llama-3.1-8B": (8.0, 32, 8, 128), "Qwen3-32B": (32.8, 64, 8, 128), "Llama-3.1-70B": (70.6, 80, 8, 128)},
                        value="Llama-3.1-8B", label="model")
ctx = mo.ui.slider(1024, 131072, step=1024, value=8192, label="context per request")
gpu = mo.ui.slider(24, 192, step=8, value=80, label="GPU memory (GB)")
wdt = mo.ui.dropdown({"bf16": 2, "fp8": 1, "int4": 0.53}, value="bf16", label="weights")
kdt = mo.ui.dropdown({"bf16": 2, "fp8": 1}, value="bf16", label="KV cache")
mo.vstack([mo.md("# 🧮 KV cache calculator"), mo.hstack([preset, gpu]), mo.hstack([ctx, wdt, kdt])])
''', ["ctx", "gpu", "kdt", "preset", "wdt"]),
    cell(["ctx", "gpu", "kdt", "mo", "plt", "preset", "wdt"], '''
_P, _L, _kv, _hd = preset.value
_w = _P * 1e9 * wdt.value / 1e9
_per_tok = 2 * _L * _kv * _hd * kdt.value
_free = 0.9 * gpu.value - _w - 2
_users = max(0, int(_free * 1e9 / (_per_tok * ctx.value)))
_fig, _ax = plt.subplots(figsize=(7, 1.4))
_ax.barh([0], [_w], color="#6366f1", label="weights"); _ax.barh([0], [max(_free, 0)], left=[_w], color="#a5b4fc", label="KV cache")
_ax.set_xlim(0, gpu.value); _ax.set_yticks([]); _ax.set_xlabel("GB"); _ax.legend(loc="lower right", fontsize=8)
_txt = (f"KV per token **{_per_tok / 1024:.0f} KiB**, per request **{_per_tok * ctx.value / 2**30:.2f} GiB**. "
        + (f"Room for **{_users} concurrent requests** at this context." if _free > 0 else "**Weights alone don't fit.** Try fp8 or int4."))
mo.vstack([_fig, mo.md(_txt)])
'''),
])

INTERACTIVE["speculative_decoding"] = ("🏎️", "Speculative Decoding", "Acceptance rate, draft length and the real speedup.", "speculative decoding", [
    IMPORTS,
    cell(["mo"], '''
acc = mo.ui.slider(0.3, 0.98, step=0.01, value=0.8, label="per-token acceptance")
cost = mo.ui.slider(0.0, 0.3, step=0.01, value=0.05, label="draft cost / target cost")
mo.vstack([mo.md("# 🏎️ Speculative decoding speedup"), mo.hstack([acc, cost])])
''', ["acc", "cost"]),
    cell(["acc", "cost", "mo", "np", "plt"], '''
_k = np.arange(1, 21); _a, _c = acc.value, cost.value
_tok = (1 - _a ** (_k + 1)) / (1 - _a)
_sp = _tok / (1 + _c * _k)
_fig, _ax = plt.subplots(figsize=(7, 3))
_ax.plot(_k, _tok, "o-", label="tokens per target pass"); _ax.plot(_k, _sp, "s-", label="wall-clock speedup")
_ax.set_xlabel("draft length k"); _ax.legend(); _ax.grid(alpha=0.3)
mo.vstack([_fig, mo.md(f"Best k = **{_k[_sp.argmax()]}**, speedup **{_sp.max():.2f}x**. "
                       "Longer drafts stop paying once a^k is small.")])
'''),
])

INTERACTIVE["gradient_descent"] = ("⛰️", "Gradient Descent Valley", "Condition number vs learning rate on a quadratic bowl.", "optimisation", [
    IMPORTS,
    cell(["mo"], '''
kappa = mo.ui.slider(1, 50, step=1, value=10, label="condition number kappa")
lr = mo.ui.slider(0.01, 2.2, step=0.01, value=0.9, label="learning rate x (1/L)")
mom = mo.ui.slider(0.0, 0.95, step=0.05, value=0.0, label="momentum")
mo.vstack([mo.md("# ⛰️ Gradient descent in a valley"), mo.hstack([kappa, lr, mom])])
''', ["kappa", "lr", "mom"]),
    cell(["kappa", "lr", "mo", "mom", "np", "plt"], '''
_A = np.diag([1.0, float(kappa.value)]); _eta = lr.value / kappa.value
_x, _v, _path = np.array([4.0, 1.5]), np.zeros(2), []
for _ in range(60):
    _path.append(_x.copy()); _v = mom.value * _v - _eta * (_A @ _x); _x = _x + _v
_path = np.array(_path)
_g = np.linspace(-5, 5, 200); _X, _Y = np.meshgrid(_g, _g)
_fig, _ax = plt.subplots(figsize=(6, 3.6))
_ax.contour(_X, _Y, 0.5 * (_X ** 2 + kappa.value * _Y ** 2), levels=15, cmap="Blues")
_ax.plot(_path[:, 0], _path[:, 1], "o-", color="#dc2626", ms=3); _ax.set_xlim(-5, 5); _ax.set_ylim(-2.5, 2.5)
_d = np.linalg.norm(_path[-1])
mo.vstack([_fig, mo.md(f"Distance to the minimum after 60 steps: **{_d:.2e}**. "
                       + ("Diverged: lr above 2/L." if _d > 10 else "Raise kappa and watch it zig-zag; add momentum to fix it."))])
'''),
])

INTERACTIVE["bayes_coin"] = ("🪙", "Bayesian Coin", "Prior + flips = posterior. Watch the Beta distribution sharpen.", "Bayes", [
    IMPORTS,
    cell(["mo"], '''
a = mo.ui.slider(0.5, 20, step=0.5, value=2, label="prior a (imaginary heads)")
b = mo.ui.slider(0.5, 20, step=0.5, value=2, label="prior b (imaginary tails)")
heads = mo.ui.slider(0, 100, step=1, value=7, label="observed heads")
tails = mo.ui.slider(0, 100, step=1, value=3, label="observed tails")
mo.vstack([mo.md("# 🪙 Beta-Bernoulli updating"), mo.hstack([a, b]), mo.hstack([heads, tails])])
''', ["a", "b", "heads", "tails"]),
    cell(["a", "b", "heads", "mo", "np", "plt", "tails"], '''
from math import lgamma
_t = np.linspace(0.001, 0.999, 400)
def _beta(x, p, q):
    return np.exp((p - 1) * np.log(x) + (q - 1) * np.log(1 - x) - (lgamma(p) + lgamma(q) - lgamma(p + q)))
_A, _B = a.value + heads.value, b.value + tails.value
_fig, _ax = plt.subplots(figsize=(7, 3))
_ax.plot(_t, _beta(_t, a.value, b.value), "--", color="gray", label=f"prior Beta({a.value:g},{b.value:g})")
_ax.plot(_t, _beta(_t, _A, _B), lw=2.5, color="#6366f1", label=f"posterior Beta({_A:g},{_B:g})")
_ax.axvline(heads.value / max(1, heads.value + tails.value), color="#dc2626", ls=":", label="MLE h/n"); _ax.legend()
mo.vstack([_fig, mo.md(f"Posterior mean **{_A / (_A + _B):.3f}**: the prior pulls the MLE toward {a.value / (a.value + b.value):.2f} "
                       "until the data outweighs it.")])
'''),
])

INTERACTIVE["gridworld_values"] = ("🗺️", "Gridworld Value Iteration", "Change gamma, slip and step cost; watch values and the policy update.", "value iteration", [
    IMPORTS,
    cell(["mo"], '''
gw_gamma = mo.ui.slider(0.5, 0.99, step=0.01, value=0.9, label="gamma")
gw_slip = mo.ui.slider(0.0, 0.4, step=0.05, value=0.1, label="slip probability")
gw_step = mo.ui.slider(-0.3, 0.0, step=0.01, value=-0.04, label="reward per step")
mo.vstack([mo.md("# 🗺️ Gridworld value iteration\\nGoal +1 (top right), pit -1 below it, grey cells are walls."),
           mo.hstack([gw_gamma, gw_slip, gw_step])])
''', ["gw_gamma", "gw_slip", "gw_step"]),
    cell(["gw_gamma", "gw_slip", "gw_step", "mo", "np", "plt"], '''
_N, _goal, _pit, _wall = 5, (0, 4), (1, 4), {(1, 1), (2, 1), (3, 3)}
_acts, _arrows = [(-1, 0), (1, 0), (0, -1), (0, 1)], "↑↓←→"
def _move(_s, _a):
    _r, _c = _s[0] + _a[0], _s[1] + _a[1]
    return _s if not (0 <= _r < _N and 0 <= _c < _N) or (_r, _c) in _wall else (_r, _c)
def _qs(_V, _s):
    _out = []
    for _a in _acts:
        _branches = [(1 - gw_slip.value, _a), (gw_slip.value / 2, (_a[1], _a[0])), (gw_slip.value / 2, (-_a[1], -_a[0]))]
        _out.append(sum(_p * (gw_step.value + gw_gamma.value * _V[_move(_s, _b)]) for _p, _b in _branches))
    return _out
_V, _iters = np.zeros((_N, _N)), 0
for _iters in range(1, 500):
    _V2 = _V.copy()
    for _r in range(_N):
        for _c in range(_N):
            if (_r, _c) not in _wall and (_r, _c) not in (_goal, _pit):
                _V2[_r, _c] = max(_qs(_V, (_r, _c)))
    _V2[_goal], _V2[_pit] = 1.0, -1.0
    _done = np.abs(_V2 - _V).max() < 1e-6
    _V = _V2
    if _done:
        break
_fig, _ax = plt.subplots(figsize=(5, 5))
_show = _V.copy()
for _w in _wall:
    _show[_w] = np.nan
_ax.imshow(_show, cmap="RdYlGn", vmin=-1, vmax=1)
for _r in range(_N):
    for _c in range(_N):
        if (_r, _c) in _wall:
            continue
        _lab = "GOAL" if (_r, _c) == _goal else "PIT" if (_r, _c) == _pit else f"{_V[_r, _c]:.2f}\\n{_arrows[int(np.argmax(_qs(_V, (_r, _c))))]}"
        _ax.text(_c, _r, _lab, ha="center", va="center", fontsize=9)
_ax.set_xticks([]); _ax.set_yticks([])
mo.vstack([_fig, mo.md(f"Converged in **{_iters}** sweeps. Make steps expensive and the agent risks the pit to finish fast; "
                       "raise slip and it takes the safe long way round.")])
'''),
])

INTERACTIVE["attention_heatmap"] = ("🔦", "Attention Heatmap", "Type a sentence and see who attends to whom, with and without a causal mask.", "attention", [
    IMPORTS,
    cell(["mo"], '''
att_text = mo.ui.text(value="the cat sat on the mat because the cat was tired", label="sentence", full_width=True)
att_causal = mo.ui.checkbox(value=True, label="causal mask")
att_sharp = mo.ui.slider(0.2, 4.0, step=0.1, value=1.5, label="sharpness (1 / temperature)")
mo.vstack([mo.md("# 🔦 Attention heatmap\\nEach word gets a fixed random vector, so repeated words match strongly."),
           att_text, mo.hstack([att_causal, att_sharp])])
''', ["att_causal", "att_sharp", "att_text"]),
    cell(["att_causal", "att_sharp", "att_text", "mo", "np", "plt"], '''
_words = att_text.value.split()[:24] or ["empty"]
def _vec(_w):
    return np.random.default_rng(sum(ord(_ch) * 31 ** _i for _i, _ch in enumerate(_w)) % 2**32).normal(size=16)
_E = np.stack([_vec(_w) for _w in _words])
_S = _E @ _E.T / 4.0 * att_sharp.value
if att_causal.value:
    _S = np.where(np.tril(np.ones_like(_S, dtype=bool)), _S, -np.inf)
_P = np.exp(_S - _S.max(1, keepdims=True)); _P /= _P.sum(1, keepdims=True)
_fig, _ax = plt.subplots(figsize=(6.5, 5.5))
_ax.imshow(_P, cmap="Purples", vmin=0, vmax=1)
_ax.set_xticks(range(len(_words)), _words, rotation=60, ha="right"); _ax.set_yticks(range(len(_words)), _words)
_ax.set_xlabel("key (attended to)"); _ax.set_ylabel("query")
mo.vstack([_fig, mo.md("Each row sums to 1. With the mask, row *i* only sees columns ≤ *i*; the first word can only look at itself.")])
'''),
])

INTERACTIVE["central_limit"] = ("🔔", "Central Limit Theorem", "Average n draws from a skewed distribution and watch a bell curve appear.", "central limit", [
    IMPORTS,
    cell(["mo"], '''
clt_dist = mo.ui.dropdown(["exponential", "uniform", "bernoulli(0.1)", "die roll"], value="exponential", label="distribution")
clt_n = mo.ui.slider(1, 100, step=1, value=5, label="n draws per average")
mo.vstack([mo.md("# 🔔 The central limit theorem"), mo.hstack([clt_dist, clt_n])])
''', ["clt_dist", "clt_n"]),
    cell(["clt_dist", "clt_n", "mo", "np", "plt"], '''
_rng = np.random.default_rng(0)
_draw = {"exponential": lambda s: _rng.exponential(1, s), "uniform": lambda s: _rng.uniform(0, 1, s),
         "bernoulli(0.1)": lambda s: (_rng.random(s) < 0.1).astype(float), "die roll": lambda s: _rng.integers(1, 7, s).astype(float)}[clt_dist.value]
_m = _draw((20000, clt_n.value)).mean(1)
_mu, _sd = _m.mean(), _m.std()
_x = np.linspace(_m.min(), _m.max(), 300)
_fig, _ax = plt.subplots(figsize=(7, 3.2))
_ax.hist(_m, bins=60, density=True, color="#c4b5fd")
_ax.plot(_x, np.exp(-(_x - _mu) ** 2 / (2 * _sd ** 2)) / np.sqrt(2 * np.pi * _sd ** 2), color="#6d28d9", lw=2)
_skew = np.mean(((_m - _mu) / _sd) ** 3)
mo.vstack([_fig, mo.md(f"Mean {_mu:.3f}, sd {_sd:.4f}, skewness **{_skew:+.2f}** (0 for a perfect bell). The sd shrinks like 1/sqrt(n).")])
'''),
])

INTERACTIVE["kl_estimators"] = ("📏", "KL Estimators (k1 vs k3)", "The two Monte Carlo KL estimators used in RLHF, side by side.", "KL", [
    IMPORTS,
    cell(["mo"], '''
kl_shift = mo.ui.slider(0.05, 2.0, step=0.05, value=0.5, label="distance between policies (mean shift)")
kl_n = mo.ui.slider(10, 2000, step=10, value=200, label="samples per estimate")
mo.vstack([mo.md("# 📏 Estimating KL(p‖q) from samples\\np = N(0,1) is the policy, q = N(shift,1) the reference. True KL = shift²/2."),
           mo.hstack([kl_shift, kl_n])])
''', ["kl_n", "kl_shift"]),
    cell(["kl_n", "kl_shift", "mo", "np", "plt"], '''
_rng = np.random.default_rng(1)
_x = _rng.normal(0, 1, (3000, kl_n.value))
_logr = -0.5 * (_x - kl_shift.value) ** 2 + 0.5 * _x ** 2          # log q(x)/p(x)
_k1 = (-_logr).mean(1)
_k3 = (np.exp(_logr) - 1 - _logr).mean(1)
_true = kl_shift.value ** 2 / 2
_fig, _ax = plt.subplots(figsize=(7, 3.2))
_ax.hist(_k1, bins=60, alpha=0.6, label=f"k1 = -log r   (sd {_k1.std():.3f})", color="#f59e0b")
_ax.hist(_k3, bins=60, alpha=0.6, label=f"k3 = r-1-log r (sd {_k3.std():.3f})", color="#7c3aed")
_ax.axvline(_true, color="black", ls="--", label=f"true KL {_true:.3f}"); _ax.legend(fontsize=8)
mo.vstack([_fig, mo.md("Both are unbiased. k3 is never negative and has lower variance when the policies are close, "
                       "which is why GRPO-style trainers use it for the KL penalty.")])
'''),
])

INTERACTIVE["low_rank_svd"] = ("🧩", "Low-Rank Approximation (SVD)", "Keep k singular values of an image: the idea behind LoRA and compression.", "SVD", [
    IMPORTS,
    cell(["mo"], '''
svd_k = mo.ui.slider(1, 60, step=1, value=5, label="rank k")
svd_noise = mo.ui.slider(0.0, 0.5, step=0.05, value=0.1, label="noise")
mo.vstack([mo.md("# 🧩 Best rank-k approximation"), mo.hstack([svd_k, svd_noise])])
''', ["svd_k", "svd_noise"]),
    cell(["mo", "np", "plt", "svd_k", "svd_noise"], '''
_g = np.linspace(-1, 1, 80)
_X, _Y = np.meshgrid(_g, _g)
_img = (np.exp(-((_X - 0.3) ** 2 + _Y ** 2) * 6) + 0.6 * ((_X ** 2 + (_Y + 0.4) ** 2) < 0.12) + 0.4 * np.sin(6 * _X) * (_Y > 0.5))
_img = _img + svd_noise.value * np.random.default_rng(0).normal(size=_img.shape)
_U, _S, _Vt = np.linalg.svd(_img)
_k = svd_k.value
_approx = (_U[:, :_k] * _S[:_k]) @ _Vt[:_k]
_err = np.linalg.norm(_img - _approx) / np.linalg.norm(_img)
_fig, _axs = plt.subplots(1, 3, figsize=(10, 3.2))
_axs[0].imshow(_img, cmap="magma"); _axs[0].set_title("original (80×80)")
_axs[1].imshow(_approx, cmap="magma"); _axs[1].set_title(f"rank {_k}")
_axs[2].semilogy(_S, "o-", ms=3); _axs[2].axvline(_k - 0.5, color="red"); _axs[2].set_title("singular values")
for _a in _axs[:2]:
    _a.set_xticks([]); _a.set_yticks([])
mo.vstack([_fig, mo.md(f"Relative error **{_err:.3f}** using {_k * 161} numbers instead of 6400 "
                       f"({_k * 161 / 6400:.0%}). LoRA bets that a fine-tune's weight *update* has a spectrum like this.")])
'''),
])

INTERACTIVE["grpo_advantages"] = ("👥", "GRPO Group Advantages", "Group size, solve rate, and how often a group carries no learning signal.", "GRPO", [
    IMPORTS,
    cell(["mo"], '''
gr_G = mo.ui.slider(2, 32, step=1, value=8, label="group size G")
gr_p = mo.ui.slider(0.01, 0.99, step=0.01, value=0.2, label="prompt solve rate p")
mo.vstack([mo.md("# 👥 GRPO: advantages within a group\\nA = (r - mean) / std over G answers to the same prompt."),
           mo.hstack([gr_G, gr_p])])
''', ["gr_G", "gr_p"]),
    cell(["gr_G", "gr_p", "mo", "np", "plt"], '''
_rng = np.random.default_rng(3)
_r = (_rng.random(gr_G.value) < gr_p.value).astype(float)
_adv = (_r - _r.mean()) / (_r.std() + 1e-6)
_Gs = np.arange(2, 33)
_dead = gr_p.value ** _Gs + (1 - gr_p.value) ** _Gs
_fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.2))
_a1.bar(range(gr_G.value), _adv, color=["#16a34a" if _x > 0 else "#dc2626" for _x in _adv])
_a1.set_title("one sampled group: advantage per answer"); _a1.set_xlabel("answer")
_a2.plot(_Gs, _dead, "o-", ms=3); _a2.axvline(gr_G.value, color="red", ls=":")
_a2.set_title("P(all answers same) = no gradient"); _a2.set_xlabel("G"); _a2.set_ylim(0, 1)
mo.vstack([_fig, mo.md(f"At G={gr_G.value}, p={gr_p.value:.2f}: **{_dead[gr_G.value - 2]:.0%}** of prompts give zero signal. "
                       "That's why DAPO filters out all-correct and all-wrong groups.")])
'''),
])

INTERACTIVE["pass_at_k"] = ("🎯", "pass@k vs pass^k", "Same agent, two very different reliability numbers.", "pass@k", [
    IMPORTS,
    cell(["mo"], '''
pk_p = mo.ui.slider(0.05, 0.95, step=0.05, value=0.4, label="per-attempt success p")
mo.vstack([mo.md("# 🎯 pass@k vs pass^k"), pk_p])
''', ["pk_p"]),
    cell(["mo", "np", "plt", "pk_p"], '''
_k = np.arange(1, 17)
_fig, _ax = plt.subplots(figsize=(7, 3.2))
_ax.plot(_k, 1 - (1 - pk_p.value) ** _k, "o-", label="pass@k: at least one of k works")
_ax.plot(_k, pk_p.value ** _k, "s-", label="pass^k: all k work")
_ax.set_xlabel("k"); _ax.set_ylim(0, 1); _ax.legend(); _ax.grid(alpha=0.3)
mo.vstack([_fig, mo.md("Benchmarks love pass@k. Users live with pass^k: an agent that works 90% of the time fails one task in ten.")])
'''),
])

GROUPS = {
    "bandit_casino": "Reinforcement Learning", "ppo_clip": "Reinforcement Learning", "gridworld_values": "Reinforcement Learning",
    "grpo_advantages": "Reinforcement Learning", "kl_estimators": "Reinforcement Learning",
    "softmax_temperature": "Transformers & LLMs", "attention_heatmap": "Transformers & LLMs",
    "kv_cache_calculator": "Inference", "speculative_decoding": "Inference",
    "gradient_descent": "Math & Optimisation", "low_rank_svd": "Math & Optimisation",
    "bayes_coin": "Probability & Statistics", "central_limit": "Probability & Statistics", "pass_at_k": "Agents & Evals",
}


def code_group(stem):
    if stem.startswith("mf_") or stem in ("math_kit", "intuitions"):
        return "Math & Optimisation"
    if stem.startswith(("pr_", "jt_", "st_")):
        return "Probability & Statistics"
    if stem.startswith("ag_"):
        return "Agents & Evals"
    if stem in ("rl_bridge",):
        return "Reinforcement Learning"
    if stem in ("inf_memory", "serving_sim", "paged_kv", "quant", "attn_kernels", "spec_decode", "distributed", "prod",
                "specialized", "mini_engine"):
        return "Inference"
    return "Transformers & LLMs"


def meta_line(**m):
    return "# sq-meta: " + json.dumps(m, ensure_ascii=False) + "\n"


# Scripts that can't run as notebooks: slow (bonus), need a server (serve_app), TensorFlow/GPU/API key, helpers.
SKIP = {"bonus", "serve_app", "keras_basics", "triton_kernels", "ag_api", "verify", "tiny_gpt"}


def code_notebook(path):
    src = path.read_text()
    doc = re.match(r'\s*"""(.*?)"""', src, re.S)
    title = (doc[1].strip().splitlines()[0] if doc else path.stem).rstrip(".")
    setup = cell([], f'''
import marimo as mo
import sys as _sys
_sys.path.insert(0, {str(path.parent)!r})
''', ["mo"])
    intro = cell(["mo"], f'mo.md({json.dumps("# " + path.stem + chr(10) + chr(10) + title + chr(10) + chr(10) + "Edit anything and re-run. Printed output appears below the cell.")})')
    body = f"@app.cell\ndef _():\n{textwrap.indent(src.rstrip(), '    ')}\n    return\n\n\n"
    return HEAD + setup + intro + body + TAIL.lstrip("\n"), title


def main():
    manifest = []
    (NB / "interactive").mkdir(parents=True, exist_ok=True)
    (NB / "code").mkdir(parents=True, exist_ok=True)
    for name, (emoji, title, desc, topic, cells) in INTERACTIVE.items():
        m = dict(title=title, emoji=emoji, group=GROUPS.get(name, "Other"), kind="interactive", desc=desc, topic=topic)
        (NB / "interactive" / f"{name}.py").write_text(meta_line(**m) + HEAD + "".join(cells) + TAIL.lstrip("\n"))
        manifest.append(dict(file=f"interactive/{name}.py", **m))
    for path in sorted((ROOT / "llm_book/code").glob("*.py")):
        if path.stem in SKIP:
            continue
        text, title = code_notebook(path)
        m = dict(title=path.stem, emoji="📓", group=code_group(path.stem), kind="code", desc=title, topic="")
        (NB / "code" / f"{path.stem}.py").write_text(meta_line(**m) + text)
        manifest.append(dict(file=f"code/{path.stem}.py", **m))
    (APP / "public/content/notebooks.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"{sum(m['kind'] == 'interactive' for m in manifest)} interactive + {sum(m['kind'] == 'code' for m in manifest)} code notebooks")


if __name__ == "__main__":
    main()
