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
    mo.md("# rl_bridge\n\nToolkit ch. 5: agents and environments, REINFORCE, DQN, and the PPO/GRPO/GSPO ratios\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Toolkit ch. 5: agents and environments, REINFORCE, DQN, and the PPO/GRPO/GSPO ratios."""
    import numpy as np
    import torch

    rng = np.random.default_rng(0)
    torch.manual_seed(0)


    class Corridor:
        """States 0..N-1. Start at 0, reward +1 at N-1, -0.01 per step. Actions: 0 left, 1 right."""

        def __init__(self, n=6, max_steps=30):
            self.n, self.max_steps = n, max_steps

        def reset(self):
            self.s, self.t = 0, 0
            return self.s

        def step(self, a):
            self.s = max(0, min(self.n - 1, self.s + (1 if a == 1 else -1)))
            self.t += 1
            done = self.s == self.n - 1 or self.t >= self.max_steps
            return self.s, (1.0 if self.s == self.n - 1 else -0.01), done


    env = Corridor()

    # REINFORCE with a tabular softmax policy and a running-mean baseline.
    theta, baseline, gamma = np.zeros((env.n, 2)), 0.0, 0.99
    for ep in range(301):
        s, done, traj = env.reset(), False, []
        while not done:
            p = np.exp(theta[s] - theta[s].max()); p /= p.sum()
            a = rng.choice(2, p=p)
            s2, r, done = env.step(a)
            traj.append((s, a, r, p)); s = s2
        G, ret = 0.0, sum(r for *_, r, _ in [(x[0], x[1], x[2], x[3]) for x in traj])
        for s, a, r, p in reversed(traj):
            G = r + gamma * G
            grad = -p; grad[a] += 1                 # d log pi(a|s) / d theta[s] = onehot(a) - pi
            theta[s] += 0.1 * (G - baseline) * grad
        baseline = 0.9 * baseline + 0.1 * ret
        if ep % 100 == 0:
            print(f"[reinforce] ep {ep:3d}  steps {len(traj):2d}  return {ret:+.2f}")

    # DQN: Q-network on one-hot states, replay buffer, target network, epsilon-greedy.
    qnet = torch.nn.Sequential(torch.nn.Linear(env.n, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
    target = torch.nn.Sequential(torch.nn.Linear(env.n, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
    target.load_state_dict(qnet.state_dict())
    opt = torch.optim.Adam(qnet.parameters(), lr=1e-2)
    onehot = torch.eye(env.n)
    buf, step = [], 0
    for ep in range(151):
        s, done, n = env.reset(), False, 0
        eps = max(0.05, 1 - ep / 75)
        while not done:
            a = rng.integers(2) if rng.random() < eps else int(qnet(onehot[s]).argmax())
            s2, r, done = env.step(a)
            buf.append((s, a, r, s2, float(done))); s = s2; n += 1; step += 1
            if len(buf) >= 64:
                idx = rng.integers(len(buf), size=64)
                S, A, R, S2, D = (torch.tensor(c) for c in zip(*[buf[i] for i in idx]))
                q = qnet(onehot[S]).gather(1, A[:, None]).squeeze(1)
                with torch.no_grad():
                    y = R.float() + gamma * (1 - D.float()) * target(onehot[S2]).max(1).values
                loss = torch.nn.functional.smooth_l1_loss(q, y)
                opt.zero_grad(); loss.backward(); opt.step()
            if step % 100 == 0:
                target.load_state_dict(qnet.state_dict())
        if ep % 50 == 0:
            print(f"[dqn] ep {ep:3d}  eps {eps:.2f}  steps {n:2d}")
    print("[dqn] greedy action per state:", qnet(onehot).argmax(1).tolist())

    # PPO clipping: gradient of min(r A, clip(r) A) wrt log-ratio.
    for A in (+1.0, -1.0):
        rows = []
        for r0 in (0.7, 1.0, 1.3):
            lr = torch.tensor(np.log(r0), requires_grad=True)
            r = lr.exp()
            obj = torch.minimum(r * A, r.clamp(0.8, 1.2) * A)
            obj.backward()
            rows.append(f"r={r0}: grad {lr.grad.item():+.2f}")
        print(f"[ppo] A={A:+.0f}  " + "  ".join(rows))

    # GRPO vs GSPO on one group of 4 sampled answers, one token has drifted a lot.
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)       # group-normalised advantage
    logp_old = torch.randn(4, 10) - 2
    logp_new = logp_old + 0.02 * torch.randn(4, 10)
    logp_new[0, 3] += 1.5                                            # one outlier token in answer 0
    tok_ratio = (logp_new - logp_old).exp()
    seq_ratio = (logp_new - logp_old).mean(1).exp()                  # GSPO: geometric mean over tokens
    print(f"[grpo] advantages {[round(a, 3) for a in adv.tolist()]}")
    print(f"[grpo] answer 0 token ratios max {tok_ratio[0].max():.2f}  -> that token is clipped at 1.2")
    print(f"[gspo] sequence ratios {[round(x, 3) for x in seq_ratio.tolist()]}")
    return


if __name__ == "__main__":
    app.run()
