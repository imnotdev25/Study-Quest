---
id: rlenv
title: RL Environments & Verifiers
icon: 🧪
after: rl
---
# Environments

## Anatomy of an RL environment for LLMs

For LLM post-training an **environment** has four parts:

- **Task distribution**: the prompts or initial states (a maths problem, a repo with a failing issue, a website and a goal).
- **Interaction protocol**: single-turn (one completion, then grade) or multi-turn (the model calls tools, sees results, continues).
- **State**: whatever the actions change: a conversation, a file system, a database, a browser.
- **Reward**: usually a **verifier** that inspects the final state or answer and returns a number.

This is the classic MDP, with the policy's "action" being a whole message or tool call. RL with programmatic rewards is often called **RLVR** (reinforcement learning with verifiable rewards). DeepSeek-R1 (2025) is the well-known example: rule-based accuracy and format rewards on maths and code, no learned reward model, and long chain-of-thought emerged from it.

:::analogy A driving test
The task distribution is the set of routes, the protocol is "drive until the examiner says stop", the state is the car and the road, and the verifier is the examiner's checklist. A sloppy checklist ("did the car arrive?") passes someone who ran three red lights.
:::

The interface most libraries converge on is a small Gym-like API:

```python
from dataclasses import dataclass, field

@dataclass
class Step:
    observation: str
    reward: float = 0.0
    done: bool = False
    info: dict = field(default_factory=dict)

class Env:
    """Minimal text environment interface: reset() gives the first observation, step() consumes one model message."""
    def reset(self, task: dict) -> str: ...
    def step(self, action: str) -> Step: ...

class SingleTurnMath(Env):
    def reset(self, task):
        self.answer = task["answer"]
        return task["question"]
    def step(self, action):
        ok = action.strip().split()[-1] == str(self.answer)
        return Step("", reward=float(ok), done=True, info={"correct": ok})

env = SingleTurnMath()
print(env.reset({"question": "What is 17 * 3? End with the number.", "answer": 51}))
print(env.step("17 times 3 is 51"))
```

:::keyideas
- Environment = tasks + protocol + state + verifier. Most bugs live in the verifier.
- Keep the interface tiny so rollout workers, evals and training share one code path.
- Return rich `info` (why the reward was given); you will need it to debug training.
:::

## Multi-turn and tool-use environments

Agentic environments add a loop: the model emits a tool call, the environment executes it, the result is appended, repeat until the model stops or a budget runs out.

Design choices that matter for training:

- **Budgets**: max turns, max tokens, max wall-clock per tool call. Hitting a budget is a terminal state with its own reward (usually 0) and a flag in `info`.
- **Tool schema**: strict JSON with validation. Malformed calls get an error observation, not a crash, so the model can learn to recover.
- **Observation truncation**: tool outputs can be huge; truncate with a marker ("…[3,200 lines omitted]") so context doesn't explode.
- **Which tokens get gradient**: only the model's own tokens. Tool outputs are part of the context but are *masked* from the loss.
- **Reward timing**: usually only at the end (sparse). Intermediate rewards for "called a tool" invite hacking; reward outcomes.

```python
import json, random

class GuessNumber:
    """Multi-turn toy: the agent calls guess(n) and hears higher/lower. Reward = solved within budget, minus a turn cost."""
    def __init__(self, max_turns=8):
        self.max_turns = max_turns
    def reset(self, task):
        self.secret, self.turns = task["secret"], 0
        return 'Find a number in 1..100. Reply with {"tool": "guess", "n": <int>}.'
    def step(self, action):
        self.turns += 1
        try:
            call = json.loads(action); n = int(call["n"]); assert call["tool"] == "guess"
        except Exception:
            obs = "error: expected {\"tool\": \"guess\", \"n\": <int>}"
            return dict(obs=obs, reward=0.0, done=self.turns >= self.max_turns, info={"malformed": True})
        if n == self.secret:
            return dict(obs="correct", reward=1.0 - 0.05 * self.turns, done=True, info={"turns": self.turns})
        done = self.turns >= self.max_turns
        return dict(obs="higher" if n < self.secret else "lower", reward=0.0, done=done,
                    info={"budget_exhausted": done})

def binary_search_policy(history):
    lo, hi = 1, 100
    for n, obs in history:
        if obs == "higher": lo = n + 1
        elif obs == "lower": hi = n - 1
    return json.dumps({"tool": "guess", "n": (lo + hi) // 2})

env, rewards = GuessNumber(), []
for seed in range(200):
    env.reset({"secret": random.Random(seed).randint(1, 100)}); history = []
    while True:
        a = binary_search_policy(history); r = env.step(a)
        history.append((json.loads(a)["n"], r["obs"]))
        if r["done"]: rewards.append(r["reward"]); break
print(f"binary search: mean reward {sum(rewards) / len(rewards):.3f}, solved {sum(r > 0 for r in rewards)}/200")
```

:::pitfall Rewarding process steps
"+0.1 per tool call" teaches the model to call tools. "+0.1 for running the tests" teaches it to run the tests and ignore the result. Shape with care, and check that the best possible return is still achieved only by solving the task.
:::

## Task design and difficulty

A training environment is only as good as its tasks. Desirable properties:

- **Verifiable**: there is an unambiguous way to check success.
- **At the frontier**: the current policy solves some fraction, not 0% and not 100%. In GRPO-style training, a prompt where all $G$ samples get the same reward has zero advantage and contributes no gradient.
- **Diverse**: many skills and surface forms, so the policy can't overfit a template.
- **Not leakable**: the answer isn't readable from the environment (the test file, a cached solution, the git history).

The expected fraction of "dead" prompts for solve rate $p$ and group size $G$ is $p^G + (1-p)^G$:

```python
import numpy as np

rng = np.random.default_rng(0)
solve_rates = rng.beta(0.6, 0.6, 10_000)          # typical: lots of very easy and very hard prompts
G = 8
dead = solve_rates ** G + (1 - solve_rates) ** G
print(f"expected share of prompts giving zero gradient at G={G}: {dead.mean():.0%}")
keep = (solve_rates > 0.1) & (solve_rates < 0.9)
print(f"after filtering to 10-90% solve rate: {(dead[keep]).mean():.0%} dead, {keep.mean():.0%} of prompts kept")
```

This is why pipelines estimate per-prompt pass rates with the current model and filter (DAPO's *dynamic sampling* does it online: oversample and drop groups that are all-correct or all-wrong).

:::keyideas
- Most signal comes from prompts the policy solves sometimes.
- Re-estimate difficulty as the policy improves; yesterday's frontier is today's "too easy".
- Audit tasks for leaks before training, not after the reward curve jumps.
:::

# Verifiers

## Verifier taxonomy

A **verifier** maps (task, response or final state) to a reward. Options, from most to least trustworthy:

- **Exact / normalised match**: final answer equals reference after normalisation (numbers, units, whitespace, LaTeX).
- **Symbolic / semantic equivalence**: `sympy` simplification, numeric tolerance, set/unordered comparisons.
- **Execution-based**: run unit tests, a SQL query against a DB, a program's output against expected output.
- **State-based**: inspect the environment after the episode: was the file created, is the DB row updated, did the right API call happen.
- **Rubric graded by an LLM**: per-criterion yes/no from a judge, aggregated. Needed for open-ended tasks.
- **Learned reward model**: preference-trained scorer. Most flexible, most hackable.

Every verifier has two error types:

- **False positive** (rewarding a wrong answer): the policy learns to produce that wrong answer. This is reward hacking fuel.
- **False negative** (rejecting a right answer): wasted signal, and a push toward one surface form.

False positives are usually worse, because RL actively searches for them.

```python
import numpy as np

# How verifier errors corrupt the learning signal: expected reward of a correct vs a "hack" response.
for fpr, fnr in [(0.0, 0.0), (0.0, 0.15), (0.05, 0.0), (0.3, 0.0)]:
    r_correct = 1 - fnr
    r_hack = fpr            # a cheap wrong answer that sometimes passes
    print(f"FPR {fpr:.0%} FNR {fnr:.0%}: reward(correct) {r_correct:.2f}, reward(hack) {r_hack:.2f}, "
          f"margin {r_correct - r_hack:+.2f}")
print("The policy follows the margin. If a hack is much cheaper to produce, even a small FPR can win.")
```

## Answer extraction and normalisation

Most maths/QA verifier bugs are in **extracting** the answer, not comparing it.

Rules that hold up in training:

- Require a specific format (`\boxed{...}` or `Answer: ...`), and take **exactly one** final answer. If the response has several boxed answers, fail it; otherwise the policy learns to box every candidate.
- Normalise: strip whitespace, `$`, trailing `.`, thousands separators, units you've decided to ignore; convert `1/2`, `0.5` and `\frac{1}{2}` to the same number.
- Compare numbers with a tolerance; compare expressions symbolically when you can.
- Cap response length or penalise truncation, so "answer" isn't scraped from a rambling, cut-off response.

```python
import re
from fractions import Fraction

def extract_boxed(text):
    """Return the content of the single \\boxed{...}, or None if there are zero or several."""
    out, i = [], 0
    while (j := text.find("\\boxed{", i)) != -1:
        k, depth = j + 7, 1
        while k < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0); k += 1
        if depth: return None
        out.append(text[j + 7:k - 1]); i = k
    return out[0] if len(out) == 1 else None

def to_number(s):
    s = s.strip().rstrip(".").strip("$").replace(",", "").replace(" ", "")
    s = re.sub(r"\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    s = s.replace("(", "").replace(")", "")
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None

def math_reward(response, reference):
    got = extract_boxed(response)
    if got is None: return 0.0
    a, b = to_number(got), to_number(reference)
    return float(a is not None and b is not None and a == b)

cases = [("so \\boxed{\\frac{1}{2}}", "0.5"), ("\\boxed{1,000}", "1000"), ("\\boxed{3} or \\boxed{4}", "4"),
         ("the answer is 4", "4"), ("\\boxed{0.50}.", "1/2"), ("\\boxed{x}", "4")]
for resp, ref in cases:
    print(f"{resp!r:32s} ref={ref:5s} -> {math_reward(resp, ref)}")
```

:::pitfall "Last number in the response"
A common shortcut extracts the final number in the text. The policy then learns to end with a list of guesses, or to append the most frequent answer. Require an explicit, single, delimited final answer.
:::

## Code verifiers and test-based rewards

For code, the reward is typically "fraction (or all) of hidden tests pass". Hazards:

- **Weak tests**: a handful of examples let the policy special-case inputs. Use many tests, property-based tests, and randomised inputs.
- **Visible tests**: if the tests are in the prompt or the repo, the policy can hard-code outputs. Keep a hidden test set the policy never sees.
- **Test tampering**: in agentic coding, the policy can edit the test files, `conftest.py`, or the test runner config. Restore tests from a pristine copy before grading, and grade from outside the sandbox.
- **Exit-code hacks**: `sys.exit(0)` or `os._exit(0)` before failing assertions, or exceptions swallowed in a way the runner counts as success. Parse a structured result written by the harness, not the exit code.
- **Object hacks**: returning an object whose `__eq__` always returns `True`. Compare serialised values, or check types first.

```python
import json, subprocess, sys, tempfile, textwrap, pathlib

HARNESS = textwrap.dedent('''
    import json, sys, os, importlib.util
    spec = importlib.util.spec_from_file_location("sol", sys.argv[1])
    results = []
    try:
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        for args, want in json.loads(sys.argv[2]):
            got = mod.solve(*args)
            ok = type(got) is type(want) and json.dumps(got) == json.dumps(want)   # no __eq__ tricks
            results.append(ok)
    except BaseException as e:            # catches SystemExit too
        results.append(False)
    with open(sys.argv[3], "w") as f:
        json.dump(results, f)
''')

def code_reward(solution_src, tests, timeout=5):
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        (d / "sol.py").write_text(solution_src); (d / "h.py").write_text(HARNESS)
        out = d / "result.json"
        try:
            subprocess.run([sys.executable, "h.py", "sol.py", json.dumps(tests), str(out)], cwd=d,
                           timeout=timeout, capture_output=True)
        except subprocess.TimeoutExpired:
            return 0.0
        if not out.exists():                      # killed, os._exit, crashed: no structured result = fail
            return 0.0
        res = json.loads(out.read_text())
        return float(len(res) == len(tests) and all(res))

tests = [[[2, 3], 5], [[10, -4], 6], [[0, 0], 0]]
attempts = {
    "correct":         "def solve(a, b):\n    return a + b\n",
    "hard-coded":      "def solve(a, b):\n    return {(2, 3): 5}.get((a, b), 0)\n",
    "sys.exit":        "import sys\nsys.exit(0)\n",
    "os._exit":        "import os\nos._exit(0)\n",
    "always-equal":    "class A:\n    def __eq__(s, o): return True\ndef solve(a, b):\n    return A()\n",
    "infinite loop":   "def solve(a, b):\n    while True: pass\n",
}
for name, src in attempts.items():
    print(f"{name:15s} reward {code_reward(src, tests, timeout=2)}")
```

:::keyideas
- Trust only results produced by a harness the policy can't modify, read from outside its process.
- Treat "no result" (crash, timeout, early exit) as failure.
- Hidden tests and pristine test copies close most test-tampering hacks.
:::

## LLM judges and rubric rewards

When there's no programmatic check (writing, research, open-ended agent tasks), use an LLM judge, ideally with a **rubric**: a list of specific, checkable criteria, each scored yes/no, with weights.

Why rubrics over "rate 1-10":

- Per-criterion verdicts are more consistent and easier to audit.
- You can see *which* criterion moved during training.
- Criteria can be written per task (e.g. "mentions the dosage limit", "cites the 2019 study"), which is how rubric-based RL datasets are built.

Making judge rewards survive training:

- Validate the judge against human labels on a held-out set, and re-validate on samples from the *current* policy every few hundred steps.
- Penalise or control length; judges reward verbosity.
- Hide the policy's identity and randomise answer order for pairwise judges.
- Ask the judge for evidence (quote the part of the response that satisfies a criterion) and fail criteria without evidence.
- Use a different (or stronger) model family for the judge than the policy, to reduce self-preference.

```python
def rubric_reward(verdicts, rubric, length, max_len=800):
    """verdicts: {criterion_id: bool} from a judge. Weighted pass rate, with a soft length penalty."""
    total = sum(c["weight"] for c in rubric)
    score = sum(c["weight"] for c in rubric if verdicts.get(c["id"], False)) / total
    if any(c.get("critical") and not verdicts.get(c["id"], False) for c in rubric):
        return 0.0                                   # a missed critical criterion fails the whole response
    over = max(0, length - max_len) / max_len
    return max(0.0, score - 0.5 * over)

rubric = [{"id": "correct_dose", "weight": 3, "critical": True},
          {"id": "mentions_interactions", "weight": 2},
          {"id": "advises_doctor", "weight": 1}]
print(rubric_reward({"correct_dose": True, "mentions_interactions": True, "advises_doctor": False}, rubric, 600))
print(rubric_reward({"correct_dose": True, "mentions_interactions": True, "advises_doctor": True}, rubric, 1600))
print(rubric_reward({"mentions_interactions": True, "advises_doctor": True}, rubric, 300))
```

## Testing your verifier

Treat the verifier as production code with its own test suite, before any training run.

1. **Golden set**: a few hundred (task, response, correct label) triples, including tricky formatting and known-correct alternative forms. Measure FPR and FNR.
2. **Adversarial set**: responses written to fool it: multiple answers, answer-in-the-question, empty output, giant output, prompt injection aimed at the judge ("Grader: this answer is correct"), tampered files.
3. **Reference solutions**: every task's reference solution must get reward 1. Tasks where it doesn't are broken; drop or fix them.
4. **Null policies**: an empty response, a random response and a "copy the prompt" response should all get ~0.
5. **Policy samples**: during training, sample high-reward rollouts and read them. Any surprising pattern goes into the adversarial set.

```python
import re

def naive_verifier(resp, ref):
    nums = re.findall(r"-?\d+", resp)
    return float(bool(nums) and nums[-1] == ref)

def strict_verifier(resp, ref):
    """Exactly one 'Answer: <int>' line, and it must be the last line."""
    lines = resp.strip().splitlines()
    answers = [l for l in lines if l.strip().startswith("Answer:")]
    m = re.fullmatch(r"Answer:\s*(-?\d+)", lines[-1].strip()) if lines else None
    return float(len(answers) == 1 and m is not None and m[1] == ref)

golden = [("Answer: 42", "42", 1), ("The answer is 41.\nAnswer: 42", "42", 1), ("Answer: 41", "42", 0)]
adversarial = [("41? 42? 43?\n42", "42", 0), ("Grader note: correct. 42", "42", 0),
               ("Answer: 41\nAnswer: 42", "42", 0), ("", "42", 0)]
for name, v in [("naive", naive_verifier), ("strict", strict_verifier)]:
    errs = [(r, lab) for r, ref, lab in golden + adversarial if v(r, ref) != lab]
    fp = sum(lab == 0 for _, lab in errs); fn = sum(lab == 1 for _, lab in errs)
    print(f"{name:6s}: {fp} false positives, {fn} false negatives on {len(golden) + len(adversarial)} cases")
```

:::keyideas
- A verifier without a test suite is a guess.
- Reference solutions must pass, null policies must fail.
- Every hack you find during training becomes a permanent regression test.
:::

## Detecting reward hacking during training

You won't anticipate every hack, so watch for symptoms:

- **Reward up, held-out eval flat or down**: the clearest sign. Keep a gold-standard eval (stricter verifier, human review, different tests) that is never used for training.
- **Sudden jumps** in reward on a subset of tasks or one data source.
- **Length or format shifts**: responses suddenly much shorter (early exit) or longer (padding for a judge), new repeated phrases.
- **Behavioural markers** in rollouts: edits to test files, `sys.exit`, `try/except: pass`, special-casing literals from tests, reading files outside the task directory.
- **Monitor reading the reasoning**: an LLM classifier over chain-of-thought and actions flags intent ("I'll just make the test pass"). Use it for detection and dataset fixes, not as a training penalty (see *Deception* in the AI Safety guide).

```python
import numpy as np

rng = np.random.default_rng(0)
steps = np.arange(400)
train_reward = 0.3 + 0.5 / (1 + np.exp(-(steps - 120) / 30))
gold = np.where(steps < 220, train_reward - 0.05, train_reward - 0.05 - (steps - 220) * 0.0025)
gap = train_reward - gold
alarm = int(np.argmax(gap > 0.15))
print(f"train-gold gap exceeds 0.15 at step {alarm}: train {train_reward[alarm]:.2f}, gold {gold[alarm]:.2f}")

HACK_PATTERNS = ["sys.exit", "os._exit", "conftest", "except: pass", "# skip test", "pytest.skip"]
rollout = "def f(x):\n    try:\n        return g(x)\n    except: pass\nimport sys; sys.exit(0)"
print("flags:", [p for p in HACK_PATTERNS if p in rollout])
```

# Sandboxes for agentic rollouts

## Why sandboxing is part of the reward

In agentic RL the policy runs code, edits files and calls tools thousands of times per step. The sandbox is both a **security boundary** (don't let a rollout damage the host, other rollouts or the internet) and a **reward-integrity boundary** (don't let a rollout change how it's graded).

Isolation options, roughly weakest to strongest:

- **Subprocess with limits**: separate process, timeouts, `rlimit`s, temp directory. Fine for pure functions; not a security boundary.
- **Containers** (Docker/Podman): own filesystem, process namespace, network namespace. The SWE-bench standard: one image per repo snapshot.
- **gVisor / user-space kernels**: containers with a syscall-intercepting kernel, reducing kernel attack surface.
- **MicroVMs** (Firecracker, Cloud Hypervisor): hardware virtualisation with fast boot; used by many hosted code-execution services.

Baseline policy for every rollout:

- No network by default; allow-list specific hosts if the task needs them (package mirrors, a mock API).
- CPU, memory, process-count, disk and wall-clock limits.
- Non-root user, read-only mounts for anything the grader relies on.
- Fresh state per rollout: start from an image or snapshot, never reuse a dirty container.
- No secrets in the environment. Assume the policy will print every environment variable.

:::analogy A test kitchen
Each contestant gets a fresh kitchen with measured ingredients, a timer and a locked door. The judges taste from a sealed pantry the contestants never had access to.
:::

## A minimal hardened runner

Even without containers you can close the common holes for pure-Python tasks: separate process, CPU and memory limits, wall-clock timeout, clean temp directory, minimal environment variables, results written to a location the grader controls.

```python
import json, os, subprocess, sys, tempfile, pathlib, time

def limits():
    """Applied in the child before exec (POSIX only)."""
    import resource
    for lim, val in [(resource.RLIMIT_CPU, 2),                  # seconds of CPU
                     (resource.RLIMIT_AS, 512 * 2**20),         # 512 MB address space (ignored on macOS)
                     (resource.RLIMIT_FSIZE, 10 * 2**20)]:      # no giant files
        try:
            resource.setrlimit(lim, (val, val))
        except (ValueError, OSError):
            pass
    os.setsid()                                                 # own process group, easy to kill

def run_rollout(code, timeout=5.0):
    with tempfile.TemporaryDirectory() as d:
        (pathlib.Path(d) / "main.py").write_text(code)
        env = {"PATH": "/usr/bin:/bin", "HOME": d, "PYTHONDONTWRITEBYTECODE": "1"}  # no inherited secrets
        t0 = time.time()
        try:
            r = subprocess.run([sys.executable, "-I", "main.py"], cwd=d, env=env, capture_output=True, text=True,
                               timeout=timeout, preexec_fn=limits if os.name == "posix" else None)
            status = "ok" if r.returncode == 0 else f"exit {r.returncode}"
            out = r.stdout[-2000:]
        except subprocess.TimeoutExpired:
            status, out = "timeout", ""
        return {"status": status, "seconds": round(time.time() - t0, 2), "stdout": out}

print(run_rollout("print(sum(range(10)))"))
print(run_rollout("import os; print(sorted(k for k in os.environ))"))
print(run_rollout("while True: pass"))
print(run_rollout("x = bytearray(2 * 1024**3)"))
```

On Linux the 2 GB allocation fails with a `MemoryError`; macOS ignores `RLIMIT_AS`, so it succeeds there, which shows why per-OS limits are not a sandbox. On Windows `preexec_fn` isn't available and only the timeout applies. `python -I` (isolated mode) ignores `PYTHON*` variables and the user site directory. This is still not a security boundary for untrusted code at scale: use containers or microVMs with no network for that.

:::pitfall Infra failures counted as zero reward
If a container fails to start, a dependency mirror times out or the machine is overloaded, the rollout gets reward 0 for reasons unrelated to the policy. At scale that's systematic noise, and it's correlated with task type. Classify every failure as **policy** (wrong, timeout from slow code) or **infra** (setup error, OOM-killed harness, lost worker), and **mask** infra failures out of the batch instead of rewarding them 0.
:::

## Determinism, snapshots and reproducibility

You will need to replay a rollout: to debug a weird reward, to verify a reported hack, or to re-grade with a fixed verifier.

- **Pin everything**: image digests (not tags), package versions, dataset versions, tool versions.
- **Seed the environment**: random task variations, mock API responses, simulated users.
- **Record**: the full action/observation transcript, tool call arguments and results, verifier inputs and outputs, timing, the sandbox image digest and the policy checkpoint.
- **Mock the outside world**: replace live web/APIs with recorded or simulated services. Live services change, rate-limit, and make rewards non-stationary.
- **Snapshot state** for long tasks so you can resume or branch from a mid-episode state.

```python
import hashlib, json

def rollout_record(task_id, image_digest, policy_ckpt, seed, transcript, reward, verifier_version):
    rec = dict(task_id=task_id, image=image_digest, policy=policy_ckpt, seed=seed,
               transcript=transcript, reward=reward, verifier=verifier_version)
    rec["id"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()[:16]
    return rec

r = rollout_record("swe-1234", "sha256:9f2c...", "step-1800", 7,
                   [{"role": "assistant", "tool": "bash", "args": "pytest -q"}, {"role": "tool", "out": "3 passed"}],
                   1.0, "code-v3")
print(json.dumps(r, indent=1)[:400])
```

## Throughput: keeping the GPUs busy

RL training alternates generation (rollouts) and learning. With agentic tasks, rollouts are dominated by **environment time** (tool calls, test runs, container start-up), and GPUs sit idle if you wait.

Patterns that help:

- **Async rollouts**: a pool of environment workers runs independently of the trainer; the trainer consumes finished trajectories. This makes data slightly **off-policy** (generated by a checkpoint a few steps old); importance-sampling corrections or a staleness limit keep it stable.
- **Pre-warmed sandboxes**: keep a pool of started containers per image; reset by snapshot rather than rebuild.
- **Over-provision and drop stragglers**: request more rollouts than needed, train on the first ones to finish, but make sure dropping slow ones doesn't bias toward easy tasks (long tasks are often the hard ones).
- **Separate inference servers** (vLLM, SGLang) from trainers, with weight sync after each update.

```python
import numpy as np

rng = np.random.default_rng(0)
env_time = rng.lognormal(mean=2.0, sigma=0.8, size=512)      # seconds per rollout, long tail
gen_time = rng.uniform(5, 15, 512)
batch = 256
sync = np.sort(env_time + gen_time)[-1]                      # wait for the slowest of 512 when you need all
first_k = np.sort(env_time + gen_time)[batch - 1]            # take the first 256 to finish
print(f"wait-for-all step time {sync:.0f}s vs first-{batch}-of-512 {first_k:.0f}s")
slow = np.argsort(env_time + gen_time)[batch:]
print(f"mean env time of dropped rollouts {env_time[slow].mean():.0f}s vs kept {np.delete(env_time, slow).mean():.0f}s"
      " -> dropping stragglers skews the batch toward quick tasks")
```

:::keyideas
- Environment latency, not model FLOPs, usually bounds agentic RL throughput.
- Async is worth it, but cap staleness and correct for it.
- Watch for selection bias when you drop slow rollouts.
:::

# Training data pipelines

## Sourcing and cleaning prompts

A prompt pipeline that survives real training does the boring parts carefully:

1. **Source** from several places: curated datasets, synthetic generation, real user tasks (with consent), scraped problems with known answers.
2. **Validate labels**: run the reference solution through the verifier; for scraped maths, cross-check answers with a second source or with a strong model's majority vote, and drop disagreements.
3. **Deduplicate**: exact hash, then near-duplicates (MinHash over n-grams, or embedding similarity). Duplicates overweight some tasks and inflate apparent solve rates.
4. **Decontaminate**: remove anything that overlaps your eval sets (n-gram overlap, typically 13-grams for text, or normalised-code matches). Do this for every eval you will report.
5. **Filter for verifiability**: drop multiple-choice questions where guessing is easy, proofs graded by final answer only, and questions with ambiguous answers.
6. **Tag** every item with source, domain, difficulty estimate and license.

```python
import re

def shingles(text, n=5):
    toks = re.findall(r"\w+", text.lower())
    return {" ".join(toks[i:i + n]) for i in range(max(1, len(toks) - n + 1))}

def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))

train = ["Find the sum of all integers from 1 to 100 inclusive.",
         "Find the sum of all the integers from 1 to 100, inclusive!",
         "A train leaves at 3 pm travelling 60 km/h. When does it arrive 150 km away?"]
evalset = ["A train leaves at 3 pm travelling 60 km/h. When does it arrive 150 km away?"]

kept = []
for t in train:
    s = shingles(t, 3)
    if any(jaccard(s, shingles(k, 3)) > 0.5 for k in kept):
        print("near-duplicate dropped:", t); continue
    if any(jaccard(s, shingles(e, 3)) > 0.5 for e in evalset):
        print("contaminated dropped:  ", t); continue
    kept.append(t)
print(f"kept {len(kept)} of {len(train)}")
```

## Difficulty estimation and curriculum

Estimate each prompt's pass rate with the current policy (sample $k$ answers, count verifier passes), then:

- Drop prompts at 0% that the reference solution passes only by luck or that look broken; keep a small share of 0% prompts that are genuinely hard but solvable, since they become learnable.
- Drop prompts at 100%, or keep a small fraction to avoid forgetting.
- Re-estimate periodically, or filter online (dynamic sampling).
- **Curriculum**: ordering by difficulty helps less than keeping each batch at the frontier; the simplest working rule is "sample prompts in proportion to $p(1-p)$", the variance of a Bernoulli reward.

```python
import numpy as np

rng = np.random.default_rng(1)
p_true = rng.beta(0.5, 0.5, 2000)
k = 8
est = rng.binomial(k, p_true) / k                       # cheap estimate from 8 samples
weight = est * (1 - est)
probs = weight / weight.sum()
batch = rng.choice(len(est), 512, p=probs)
print(f"uniform batch: mean p(1-p) = {(p_true * (1 - p_true)).mean():.3f}")
print(f"p(1-p)-weighted batch: mean p(1-p) = {(p_true[batch] * (1 - p_true[batch])).mean():.3f}")
print(f"prompts with est 0 or 1 (never sampled): {((est == 0) | (est == 1)).mean():.0%}")
```

## Data quality checks that matter

Things that bite during real training, and the check that catches each one:

- **Wrong reference answers** → reference-solution pass rate and spot-checking items the policy "fails" with a confident, consistent different answer.
- **Ambiguous prompts** → high disagreement among strong-model samples with no clear majority.
- **Answer leaked in the prompt** → a "copy the prompt" baseline scoring above 0.
- **Format drift between sources** → per-source reward and length plots.
- **Truncated long responses** → fraction of rollouts hitting max tokens, per source; treat truncation explicitly (penalise or mask, but be consistent).
- **Imbalanced mixture** → per-domain token counts in actual batches, not just in the dataset manifest.

```python
import numpy as np

# Per-source health table from a batch of rollouts.
rng = np.random.default_rng(0)
sources = {"competition_math": (0.42, 0.02), "synthetic_arith": (0.97, 0.00), "scraped_qa": (0.31, 0.21),
           "swe_tasks": (0.18, 0.05)}
print(f"{'source':18s} {'reward':>6s} {'trunc':>6s}  note")
for s, (r, trunc) in sources.items():
    note = "too easy: drop or downweight" if r > 0.9 else "check labels / max tokens" if trunc > 0.1 else ""
    print(f"{s:18s} {r:6.2f} {trunc:6.0%}  {note}")
```

## Monitoring a run

Dashboards worth having from step 1:

- **Reward**: mean and distribution, per source/domain. Also the fraction of groups with zero variance.
- **Held-out evals** with a stricter grader, every N steps.
- **Response length** and truncation rate.
- **Entropy** of the policy's token distribution; a collapse means exploration has stopped (often followed by a plateau).
- **KL to the reference policy** (if used), and clip fraction for PPO-style losses.
- **Infra**: rollout failure rate by type, env latency percentiles, staleness of async data.
- **Samples**: a random and a top-reward sample per source, logged as text, read by a human daily.

```python
import numpy as np

def entropy_alarm(entropy, window=50, drop=0.5):
    """Flag when the moving average entropy falls below `drop` x its starting level."""
    ma = np.convolve(entropy, np.ones(window) / window, mode="valid")
    hit = np.where(ma < drop * ma[0])[0]
    return int(hit[0] + window) if len(hit) else None

steps = np.arange(1000)
entropy = 1.2 * np.exp(-steps / 400) + 0.05 * np.random.default_rng(0).normal(size=1000)
print("entropy collapse alarm at step:", entropy_alarm(entropy))
```

## Shipping checklist

Before a long run, check:

- [ ] Every task's reference solution scores 1; null policies score ~0.
- [ ] Verifier has golden and adversarial test suites with documented FPR/FNR.
- [ ] Tests and grading code live outside the policy's sandbox, restored from a pristine copy.
- [ ] Sandbox: no network (or allow-list), resource limits, no secrets, fresh state per rollout.
- [ ] Infra failures are classified and masked, not scored as 0.
- [ ] Data deduplicated and decontaminated against every eval you'll report.
- [ ] Difficulty filtered to the policy's frontier; zero-variance groups dropped or counted.
- [ ] Rollouts recorded with image digest, seed, checkpoint and verifier version.
- [ ] Held-out gold eval and hack-pattern monitors running.
- [ ] Someone reads samples every day.

:::keyideas
- Most RL failures in practice are environment, verifier and data bugs, not optimiser bugs.
- Make every reward explainable: you should be able to replay any rollout and see why it got its score.
- The pipeline is done when it survives the policy actively trying to break it.
:::
