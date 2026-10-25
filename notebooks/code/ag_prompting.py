# sq-meta: {"title": "ag_prompting", "emoji": "📓", "group": "Agents & Evals", "kind": "code", "desc": "Agents ch. 'Prompting Patterns': prompt builders, answer parsing, and why voting helps or hurts", "topic": ""}
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
    mo.md("# ag_prompting\n\nAgents ch. 'Prompting Patterns': prompt builders, answer parsing, and why voting helps or hurts\n\nEdit anything and re-run. Printed output appears below the cell.")
    return


@app.cell
def _():
    """Agents ch. 'Prompting Patterns': prompt builders, answer parsing, and why voting helps or hurts."""
    import json
    import re

    import numpy as np

    rng = np.random.default_rng(0)
    Q = "Pens cost $3 and notebooks $5. Ana buys 4 pens and 2 notebooks. How much does she pay?"


    def zero_shot(q):
        return [{"role": "user", "content": q}]


    def few_shot(q, examples):
        msgs = []
        for x, y in examples:                                  # demonstrations as real prior turns
            msgs += [{"role": "user", "content": x}, {"role": "assistant", "content": y}]
        return msgs + [{"role": "user", "content": q}]


    def chain_of_thought(q):
        return [{"role": "user", "content": q + "\n\nWork it out step by step. End with a line 'Answer: <number>'."}]


    def structured(q, schema):
        system = "Reply with one JSON object that matches this schema, and nothing else:\n" + json.dumps(schema)
        return [{"role": "system", "content": system}, {"role": "user", "content": q}]


    SCHEMA = {"type": "object", "required": ["total", "currency"],
              "properties": {"total": {"type": "number"}, "currency": {"type": "string"}}}

    for name, msgs in (("zero-shot", zero_shot(Q)),
                       ("few-shot", few_shot(Q, [("Apples cost $2. Tom buys 3. Total?", "Answer: 6")])),
                       ("chain-of-thought", chain_of_thought(Q)), ("structured", structured(Q, SCHEMA))):
        print(f"[prompt] {name:16s} {len(msgs)} messages, roles {[m['role'] for m in msgs]}")


    def parse_answer(text):
        m = re.search(r"Answer:\s*\$?(-?\d+(?:\.\d+)?)\s*$", text.strip(), re.M)
        return float(m[1]) if m else None


    def validate(obj, schema):
        """Tiny JSON-schema check: required keys and primitive types."""
        types = {"number": (int, float), "string": str}
        missing = [k for k in schema["required"] if k not in obj]
        wrong = [k for k, s in schema["properties"].items() if k in obj and not isinstance(obj[k], types[s["type"]])]
        return not missing and not wrong, missing, wrong


    reply = "4 pens cost 4 x 3 = 12.\n2 notebooks cost 2 x 5 = 10.\n12 + 10 = 22.\nAnswer: 22"
    print(f"[parse] CoT reply -> {parse_answer(reply)}   no answer line -> {parse_answer('It is 22 dollars.')}")
    for raw in ('{"total": 22, "currency": "USD"}', '{"total": "22"}'):
        print(f"[schema] {raw:34s} valid/missing/wrong-type: {validate(json.loads(raw), SCHEMA)}")


    # Self-consistency: sample n answers, take the most common. Simulated with a fixed answer distribution.
    def vote_accuracy(probs, n, trials=100_000):
        """probs[0] = P(correct answer); others = distinct wrong answers. Ties broken at random."""
        counts = rng.multinomial(n, probs, size=trials).astype(float)
        counts += rng.random(counts.shape) * 1e-3
        return np.mean(counts.argmax(1) == 0)


    cases = {"correct is the most likely answer (0.40 vs 0.12 x5)": [0.40] + [0.12] * 5,
             "one wrong answer is more likely (0.30 vs 0.45)": [0.30, 0.45, 0.25]}
    for name, probs in cases.items():
        accs = "  ".join(f"n={n}: {vote_accuracy(probs, n):.3f}" for n in (1, 5, 15, 41))
        print(f"[vote] {name}\n[vote]   {accs}")
    return


if __name__ == "__main__":
    app.run()
