# sq-meta: {"title": "Verifier Lab", "emoji": "✅", "group": "RL Environments", "kind": "interactive", "desc": "Write responses that try to fool a maths verifier. Compare a naive and a strict one.", "topic": "verifier"}
import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return (mo, np, plt,)


@app.cell
def _(mo):
    vl_ref = mo.ui.text(value="1/2", label="reference answer")
    vl_resp = mo.ui.text_area(value="Let me think. 0.4? Actually \\boxed{0.5}", label="model response", full_width=True)
    mo.vstack([mo.md("# ✅ Verifier Lab\nTry: several boxed answers, a trailing guess, `\\boxed{\\frac{1}{2}}`, `Answer: 1/2`, an empty box."), vl_ref, vl_resp])
    return (vl_ref, vl_resp,)


@app.cell
def _(mo, vl_ref, vl_resp):
    import re as _re
    from fractions import Fraction as _Fr
    def _num(s):
        s = s.strip().rstrip(".").strip("$").replace(",", "").replace(" ", "")
        s = _re.sub(r"\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", s)
        try: return _Fr(s)
        except (ValueError, ZeroDivisionError): return None
    def _naive(resp, ref):
        nums = _re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", resp)
        return bool(nums) and _num(nums[-1]) == _num(ref), "last number in the text"
    def _strict(resp, ref):
        boxes = _re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", resp)
        if len(boxes) != 1: return False, f"needs exactly one \\boxed{{}}, found {len(boxes)}"
        a, b = _num(boxes[0]), _num(ref)
        if a is None: return False, f"could not parse '{boxes[0]}'"
        return a == b, f"parsed {a} vs {b}"
    _rows = []
    for _name, _v in (("naive", _naive), ("strict", _strict)):
        _ok, _why = _v(vl_resp.value, vl_ref.value)
        _rows.append({"verifier": _name, "reward": int(_ok), "reason": _why})
    mo.ui.table(_rows, selection=None)
    return


if __name__ == "__main__":
    app.run()
