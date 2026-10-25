#!/usr/bin/env python3
"""Turn the three LaTeX books into app content.

Writes, under study_app/public/:
  content/catalog.json        books -> parts -> topics (titles, minutes, search text)
  content/cards.json          flashcards cut from derive / keyideas / algorithm / analogy boxes
  content/<book>/<n>.html     one HTML file per topic (math left as \\(..\\) for KaTeX)
  figs/<book>/<hash>.svg      every TikZ / pgfplots figure, compiled once and cached by hash

The .tex books are only read, never modified.
Usage: ../.venv/bin/python scripts/build_content.py   (needs pandoc and tectonic on PATH)
"""
import hashlib
import html
import json
import pathlib
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[2]          # .../RL
APP = pathlib.Path(__file__).resolve().parents[1]           # .../RL/study_app
OUT = APP / "public"
SD = ROOT.parent / "System-Design"
MY_BOXES = dict(titled={"analogy", "deeper", "walkthrough", "algobox", "derive"}, plain={"keyideas", "startbox"})
BOOKS = [
    dict(id="rl", title="Reinforcement Learning", path=ROOT / "rl_enhanced.tex", icon="🎮", level="section", **MY_BOXES),
    dict(id="llm", title="LLM Fundamentals", path=ROOT / "llm_book/llm_basics.tex", icon="🧠", level="section", **MY_BOXES),
    dict(id="post", title="Pre-Training to Post-Training", path=ROOT / "posttrain_book/pre_to_post.tex", icon="🛠️",
         level="section", **MY_BOXES),
    dict(id="sd", title="System Design: Zero to Expert", path=SD / "main.tex", icon="🏗️", level="chapter",
         titled={"usecase"}, plain={"beginner", "intermediate", "expert", "pitfall", "takeaways", "interview",
                                   "chapterintro", "newin", "csfacts", "lesson"}),
    dict(id="dsa", title="DSA and ML Interviews", path=SD / "dsa-ml-interviews/main.tex", icon="🧩", level="chapter",
         titled={"interview"}, plain={"beginner", "intermediate", "expert", "why", "signals", "mlconn", "pitfall",
                                      "takeaways", "chapterintro", "followups"}),
]
PYZ = {"bs": "\\", "us": "_", "ob": "{", "cb": "}", "ca": "^", "am": "&", "lt": "<", "gt": ">", "sh": "#",
       "pc": "%", "dl": "$", "hy": "-", "sq": "'", "dq": '"', "ti": "~", "at": "@", "lb": "[", "rb": "]"}
ALGO_CMDS = ("Statex|State|Require|Ensure|ForAll|For|EndFor|While|EndWhile|If|ElsIf|Else|EndIf|Repeat|Until|"
             "Loop|EndLoop|Function|EndFunction|Procedure|EndProcedure")


# ----------------------------------------------------------------------------- tex helpers
def braced(s, i):
    """s[i] == '{'. Return (content, index after the closing brace)."""
    depth, j = 0, i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError("unbalanced braces")


def replace_cmd(s, name, fn, nargs=1):
    """Replace \\name{a}{b}... (balanced) by fn(*args)."""
    out, i, pat = [], 0, re.compile(r"\\" + name + r"(?![A-Za-z])\s*")
    while (m := pat.search(s, i)):
        out.append(s[i:m.start()])
        j, args = m.end(), []
        try:
            for _ in range(nargs):
                while s[j] in " \n\t":
                    j += 1
                if s[j] != "{":
                    raise ValueError("missing argument")
                a, j = braced(s, j)
                args.append(a)
        except (ValueError, IndexError):
            out.append(s[m.start():m.end()]); i = m.end(); continue
        out.append(fn(*args)); i = j
    return "".join(out) + s[i:]


def decode_pygments(code):
    code = re.sub(r"\\PY\{[^}]*\}\{((?:[^{}]|\{\})*)\}", lambda m: m[1], code)
    return re.sub(r"\\PYZ(\w\w)\{\}", lambda m: PYZ.get(m[1], ""), code)


def algorithmic_to_tex(body):
    """algpseudocode -> indented lines wrapped in markers that become <div class="algo-line">."""
    body = re.sub(r"^\s*\[[^\]]*\]", "", body)                      # \begin{algorithmic}[1]
    body = body.replace("\\Return", "\\textbf{return}")
    for k, v in (("And", "and"), ("Or", "or"), ("Not", "not"), ("Break", "break"), ("Continue", "continue")):
        body = re.sub(r"\\" + k + r"(?![A-Za-z])", "\\\\textbf{" + v + "} ", body)
    lines, depth = [], 0
    parts = re.split(r"\\(" + ALGO_CMDS + r")(?![A-Za-z])", body)
    for k in range(1, len(parts), 2):
        cmd, rest = parts[k], parts[k + 1]
        args = []
        nargs = {"For": 1, "ForAll": 1, "While": 1, "If": 1, "ElsIf": 1, "Until": 1, "Function": 2, "Procedure": 2}.get(cmd, 0)
        rest = rest.lstrip()
        for _ in range(nargs):
            if rest.startswith("{"):
                a, j = braced(rest, 0); args.append(a); rest = rest[j:]
        rest = replace_cmd(rest.strip(), "Comment", lambda c: f"\\quad\\textit{{▷ {c}}}")
        if cmd in ("EndFor", "EndWhile", "EndIf", "EndLoop", "EndFunction", "EndProcedure", "Until"):
            depth = max(0, depth - 1)
        if cmd in ("ElsIf", "Else"):
            depth = max(0, depth - 1)
        text = {
            "Require": "\\textbf{Input:} " + rest, "Ensure": "\\textbf{Output:} " + rest,
            "For": f"\\textbf{{for}} {args[0] if args else ''} \\textbf{{do}} {rest}",
            "ForAll": f"\\textbf{{for all}} {args[0] if args else ''} \\textbf{{do}} {rest}",
            "While": f"\\textbf{{while}} {args[0] if args else ''} \\textbf{{do}} {rest}",
            "If": f"\\textbf{{if}} {args[0] if args else ''} \\textbf{{then}} {rest}",
            "ElsIf": f"\\textbf{{else if}} {args[0] if args else ''} \\textbf{{then}} {rest}",
            "Else": "\\textbf{else} " + rest, "Repeat": "\\textbf{repeat} " + rest,
            "Until": f"\\textbf{{until}} {args[0] if args else ''} {rest}", "Loop": "\\textbf{loop} " + rest,
            "Function": f"\\textbf{{function}} {args[0] if args else ''}({args[1] if len(args) > 1 else ''}) {rest}",
            "Procedure": f"\\textbf{{procedure}} {args[0] if args else ''}({args[1] if len(args) > 1 else ''}) {rest}",
        }.get(cmd, rest)
        if cmd.startswith("End"):
            text = "\\textbf{end} " + rest if rest else "\\textbf{end}"
        if text.strip():
            lines.append(f"XXIND{depth}XX {text.strip()}")
        if cmd in ("For", "ForAll", "While", "If", "ElsIf", "Else", "Repeat", "Loop", "Function", "Procedure"):
            depth += 1
    return "\n\nXXALGOSTARTXX\n\n" + "\n\n".join(lines) + "\n\nXXALGOENDXX\n\n"


def xltabular_to_tabular(s):
    def fix(m):
        spec = m[1]
        n = max(1, spec.count("X") + len(re.findall(r"p\{", spec)))
        return "\\begin{tabular}{" + "l" * n + "}"
    s = re.sub(r"\\begin\{xltabular\}\{[^}]*\}\{((?:[^{}]|\{[^{}]*\})*)\}", fix, s)
    s = s.replace("\\end{xltabular}", "\\end{tabular}")
    return re.sub(r"\\end(first)?head|\\endlastfoot|\\endfoot", "", s)


# ----------------------------------------------------------------------------- per-topic conversion
class Topic:
    def __init__(self, book, idx, part, title, tex):
        self.book, self.idx, self.part, self.title_tex, self.tex = book, idx, part, title, tex
        self.codes, self.figs = [], []


def colspec_count(spec):
    spec = re.sub(r"[<>]\{(?:[^{}]|\{[^{}]*\})*\}", "", spec)
    spec = re.sub(r"[pmb]\{[^}]*\}", "p", spec)
    spec = re.sub(r"\*\{(\d+)\}\{([^}]*)\}", lambda m: m[2] * int(m[1]), spec)
    return max(1, sum(spec.count(c) for c in "lcrXp"))


def preprocess(t, figs_out, cfg):
    s = t.tex
    s = re.sub(r"\\begin\{lstlisting\}(\[[^\n]*\])?\n(.*?)\\end\{lstlisting\}",
               lambda m: "\\begin{Verbatim}" + ("[commandchars=py]" if "Python" in (m[1] or "") else "[lst]") + "\n"
               + m[2] + "\\end{Verbatim}", s, flags=re.S)

    def code(m):
        opts, body = m[1] or "", m[2]
        kind = "python" if "commandchars" in opts else ("output" if "scriptsize" in opts else "text")
        t.codes.append((kind, decode_pygments(body) if "commandchars=\\\\" in opts else body))
        return f"\n\nXXCODE{len(t.codes) - 1}XX\n\n"
    s = re.sub(r"\\begin\{Verbatim\}(\[[^\n]*\])?\n(.*?)\\end\{Verbatim\}", code, s, flags=re.S)

    def fig(m):
        src = m[0]
        h = hashlib.sha1(src.encode()).hexdigest()[:12]
        figs_out.append((h, src))
        t.figs.append(h)
        return f"\n\nXXFIG{h}XX\n\n"
    s = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", fig, s, flags=re.S)

    s = re.sub(r"\\begin\{algorithmic\}(.*?)\\end\{algorithmic\}", lambda m: algorithmic_to_tex(m[1]), s, flags=re.S)
    s = re.sub(r"\\begin\{algorithm\}(\[[^\]]*\])?", "", s).replace("\\end{algorithm}", "")
    s = re.sub(r"\\begin\{(figure|table)\*?\}(\[[^\]]*\])?", "", s)
    s = re.sub(r"\\end\{(figure|table)\*?\}", "", s)
    s = replace_cmd(s, "caption", lambda c: f"\n\n\\emph{{{c}}}\n\n")
    s = re.sub(r"\\begin\{tabularx\}\{[^}]*\}\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}",
               lambda m: "\\begin{tabular}{" + "l" * colspec_count(m[1]) + "}", s).replace("\\end{tabularx}", "\\end{tabular}")
    s = xltabular_to_tabular(s)
    s = re.sub(r"\\(centering|small|footnotesize|scriptsize|raggedright)\b", "", s)
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    s = re.sub(r"~?\\(?:auto|page)?ref\{(?:\w+:)?([^}]*)\}", lambda m: " (" + m[1].replace("-", " ") + ")", s)
    s = replace_cmd(s, "term", lambda x: "\\textbf{" + x + "}")
    s = replace_cmd(s, "code", lambda x: "\\texttt{" + x + "}")
    s = replace_cmd(s, "bigO", lambda x: "\\(O(" + x + ")\\)")
    s = replace_cmd(s, "cx", lambda a, b: f"\\textbf{{Time:}} \\(O({a})\\) \\quad \\textbf{{Space:}} \\(O({b})\\)", nargs=2)
    s = replace_cmd(s, "case", lambda x: "\\section{Case: " + x + "}")
    s = replace_cmd(s, "csrow", lambda a, b, c, d: f"\\textbf{{Domain:}} {a}\\\\ \\textbf{{Scale:}} {b}\\\\ "
                    f"\\textbf{{Concepts:}} {c}\\\\ \\textbf{{Source:}} {d}", nargs=4)
    s = re.sub(r"\\begin\{(enumerate|itemize)\}\[[^\]]*\]", r"\\begin{\1}", s)
    s = s.replace("\\begin{psmallmatrix}", "\\left(\\begin{smallmatrix}").replace("\\end{psmallmatrix}", "\\end{smallmatrix}\\right)")
    s = replace_cmd(s, "figcap", lambda c: f"\n\n\\emph{{{c}}}\n\n")
    s = replace_cmd(s, "step", lambda n: f"\n\n\\textbf{{Step {n}.}} ")
    s = replace_cmd(s, "path", lambda p: "\\texttt{" + p.replace("_", "\\_") + "}")
    s = s.replace("\\mistakes", "\n\n\\textbf{Common mistakes}\n\n")
    for env in cfg["titled"]:
        s = replace_cmd(s, r"begin\{" + env + r"\}", lambda ttl, e=env: f"\n\nXXBOXSTART {e} XXBOXTITLE {ttl} XXBOXTITLEEND\n\n")
        s = s.replace(f"\\end{{{env}}}", "\n\nXXBOXCLOSEXX\n\n")
    for env in cfg["plain"]:
        s = re.sub(r"\\begin\{" + env + r"\}(\[[^\]]*\])?", lambda m: f"\n\nXXBOXSTART {env} XXBOXTITLE {(m[1] or '')[1:-1]} XXBOXTITLEEND\n\n", s)
        s = s.replace(f"\\begin{{{env}}}", f"\n\nXXBOXSTART {env} XXBOXTITLE  XXBOXTITLEEND\n\n")
        s = s.replace(f"\\end{{{env}}}", "\n\nXXBOXCLOSEXX\n\n")
    s = re.sub(r"\\(clearpage|newpage|noindent|smallskip|medskip|bigskip)\b", "", s)
    t.pre = f"XXTITLE {t.title_tex} XXTITLEEND\n\n" + s


def pandoc(tex):
    r = subprocess.run(["pandoc", "-f", "latex", "-t", "html", "--math-method=mathjax", "--wrap=none"],
                       input=tex, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-500:])
    return r.stdout


BOX_LABEL = {"analogy": "🧩 Analogy", "deeper": "🔍 Deeper", "walkthrough": "🚶 Walkthrough", "algobox": "⚙️ Algorithm",
             "derive": "✏️ Derivation", "keyideas": "🔑 Key ideas", "startbox": "🚀 Starting point",
             "beginner": "🌱 Beginner", "intermediate": "🔧 Intermediate", "expert": "🧠 Expert", "usecase": "📦 Use case",
             "pitfall": "⚠️ Pitfall", "takeaways": "🔑 Takeaways", "interview": "🎤 Interview", "chapterintro": "👋 Intro",
             "newin": "🆕 New in 2026", "csfacts": "📋 Case facts", "lesson": "📚 Lesson", "why": "❓ Why",
             "signals": "📡 Interview signals", "mlconn": "🤖 ML connection", "followups": "➕ Follow-ups"}


def postprocess(t, h):
    m = re.search(r"<p>XXTITLE (.*?) XXTITLEEND</p>", h, re.S)
    t.title_html = m[1].strip() if m else html.escape(t.title_tex)
    h = h[m.end():] if m else h

    def box_open(m):
        kind, ttl = m[1], m[2].strip()
        label = BOX_LABEL.get(kind, kind)
        title = f"{label}: {ttl}" if ttl else label
        return f'<div class="box box-{kind}" data-kind="{kind}"><div class="box-title">{title}</div>'
    h = re.sub(r"<p>XXBOXSTART (\w+) XXBOXTITLE(.*?)XXBOXTITLEEND</p>", box_open, h, flags=re.S)
    h = h.replace("<p>XXBOXCLOSEXX</p>", "</div>")
    h = h.replace("<p>XXALGOSTARTXX</p>", '<div class="algo">').replace("<p>XXALGOENDXX</p>", "</div>")
    h = re.sub(r"<p>XXIND(\d+)XX (.*?)</p>", lambda m: f'<div class="algo-line" style="--d:{m[1]}">{m[2]}</div>', h, flags=re.S)

    def code(m):
        kind, body = t.codes[int(m[1])]
        cls = {"python": "language-python", "output": "output", "text": "language-plaintext"}[kind]
        return f'<pre class="code {kind}"><code class="{cls}">{html.escape(body.rstrip())}</code></pre>'
    h = re.sub(r"<p>XXCODE(\d+)XX</p>", code, h)
    h = re.sub(r"<p>XXFIG(\w+)XX</p>", lambda m: f'<figure class="tikz"><img src="figs/{t.book}/{m[1]}.svg" '
               f'alt="diagram" loading="lazy"></figure>', h)
    h = re.sub(r"XXIND(\d+)XX ?", "", h)
    h = re.sub(r"XX(CODE\d+|FIG\w+|BOXCLOSE|ALGOSTART|ALGOEND)XX", "", h)     # stray markers inside lists/tables
    t.html = h


def boxes(h):
    """Yield (kind, title_html, inner_html) for each top-level box, handling nested divs."""
    for m in re.finditer(r'<div class="box box-(\w+)" data-kind="\w+"><div class="box-title">(.*?)</div>', h, re.S):
        depth, j = 1, m.end()
        for tag in re.finditer(r"<div\b|</div>", h[m.end():]):
            depth += 1 if tag[0] == "<div" else -1
            if depth == 0:
                j = m.end() + tag.start(); break
        yield m[1], m[2], h[m.end():j]


def plain(h):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", h))).strip()


# ----------------------------------------------------------------------------- figures
def compile_figures(book, preamble, figs):
    outdir = OUT / "figs" / book
    outdir.mkdir(parents=True, exist_ok=True)
    todo = list({h: s for h, s in figs if not (outdir / f"{h}.svg").exists()}.items())
    if not todo:
        return 0
    import fitz
    doc = (preamble + "\n\\usepackage[active,tightpage]{preview}\\PreviewEnvironment{tikzpicture}"
           "\\setlength\\PreviewBorder{6pt}\n\\begin{document}\n" + "\n\n".join(s for _, s in todo) + "\n\\end{document}\n")
    with tempfile.TemporaryDirectory() as d:
        (pathlib.Path(d) / "figs.tex").write_text(doc)
        r = subprocess.run(["tectonic", "-X", "compile", "figs.tex"], cwd=d, capture_output=True, text=True)
        pdf = pathlib.Path(d) / "figs.pdf"
        if not pdf.exists():
            print(f"WARN {book}: figure compile failed\n{r.stderr[-1500:]}", file=sys.stderr)
            return 0
        pages = fitz.open(pdf)
        if len(pages) != len(todo):
            print(f"WARN {book}: {len(pages)} pages for {len(todo)} figures; skipping", file=sys.stderr)
            return 0
        for (h, _), page in zip(todo, pages):
            (outdir / f"{h}.svg").write_text(page.get_svg_image(text_as_path=True))
    return len(todo)


# ----------------------------------------------------------------------------- main
def expand_inputs(src, base):
    def inc(m):
        f = base / (m[1] if m[1].endswith(".tex") else m[1] + ".tex")
        return expand_inputs(f.read_text(), base) if f.exists() else ""
    return re.sub(r"(?m)^\s*\\input\{([^}]+)\}", inc, src)


def split_book(book, path, level="section"):
    src = expand_inputs(path.read_text(), path.parent)
    src = re.sub(r"(?m)^%.*$", "", src)
    start = re.search(r"(?m)^\\begin\{document\}", src)
    preamble = src[:start.start()]
    body = src[start.end():src.rindex("\\end{document}")]
    body = body[body.index("\\part{"):]                              # drop title page and TOC
    topics, parts = [], []
    for pchunk in re.split(r"(?m)^\\part(?=\{)", body)[1:]:
        ptitle, j = braced(pchunk, 0)
        parts.append(ptitle)
        secs = re.split(r"(?m)^\\" + level + r"\*?(?=\{)", pchunk[j:])
        for sec in secs[1:]:
            stitle, k = braced(sec, 0)
            topics.append(Topic(book, len(topics), len(parts) - 1, stitle, sec[k:]))
    return preamble, parts, topics


def macros(preamble):
    out = {}
    for m in re.finditer(r"\\(?:re)?newcommand\{?\\(\w+)\}?\{(.*)\}\s*$", preamble, re.M):
        if "#" not in m[2] and m[1] not in ("figcap", "step", "mistakes", "arraystretch", "headrulewidth"):
            out["\\" + m[1]] = m[2]
    for m in re.finditer(r"\\DeclareMathOperator\*?\{\\(\w+)\}\{(.*?)\}", preamble):
        out["\\" + m[1]] = "\\operatorname{" + m[2] + "}"
    return out


def main():
    catalog, cards, all_macros = {"books": []}, [], {}
    for cfg in BOOKS:
        book, name, icon = cfg["id"], cfg["title"], cfg["icon"]
        preamble, parts, topics = split_book(book, cfg["path"], cfg["level"])
        all_macros.update(macros(preamble))
        figs = []
        for t in topics:
            preprocess(t, figs, cfg)
        n_new = compile_figures(book, preamble, figs)
        part_html = pandoc("\n\n".join(f"XXPT {p} XXPTEND" for p in parts))
        part_titles = [plain(x) for x in re.findall(r"XXPT (.*?) XXPTEND", part_html, re.S)]
        with ThreadPoolExecutor(8) as ex:
            htmls = list(ex.map(lambda t: pandoc(t.pre), topics))
        (OUT / "content" / book).mkdir(parents=True, exist_ok=True)
        entry = {"id": book, "title": name, "icon": icon, "parts": [{"title": p, "topics": []} for p in part_titles]}
        for t, h in zip(topics, htmls):
            postprocess(t, h)
            (OUT / "content" / book / f"{t.idx}.html").write_text(t.html)
            text = plain(t.html)
            heads = [plain(x) for x in re.findall(r"<h[23][^>]*>(.*?)</h[23]>", t.html, re.S)]
            entry["parts"][t.part]["topics"].append({
                "id": f"{book}:{t.idx}", "idx": t.idx, "title": plain(t.title_html), "titleHtml": t.title_html,
                "minutes": max(3, round(len(text.split()) / 180)), "heads": heads[:30],
                "search": text[:1500].lower(), "hasCode": "<pre" in t.html, "figs": len(t.figs)})
            for b, (kind, ttl, inner) in enumerate(boxes(t.html)):
                if kind == "walkthrough" or len(plain(inner)) < 40:
                    continue
                if kind in ("beginner", "intermediate", "chapterintro", "usecase", "csfacts", "newin", "lesson", "followups"):
                    continue
                label = {"derive": "Derive", "deeper": "Explain", "algobox": "Write the algorithm", "analogy": "Analogy",
                         "keyideas": "Key ideas", "startbox": "Starting point", "expert": "Expert detail",
                         "pitfall": "Pitfall", "takeaways": "Takeaways", "interview": "Interview", "why": "Why",
                         "signals": "Interview signals", "mlconn": "ML connection"}.get(kind, kind.title())
                sub = re.sub(r"^.*?: ", "", ttl, count=1) if ":" in ttl else plain(t.title_html)
                cards.append({"id": f"{book}:{t.idx}:{b}", "topic": f"{book}:{t.idx}", "kind": kind,
                              "front": f"{label}: {sub}", "topicTitle": plain(t.title_html), "back": inner})
        catalog["books"].append(entry)
        print(f"{book}: {len(parts)} parts, {len(topics)} topics, {len(figs)} figures ({n_new} compiled now)")
    catalog["macros"] = all_macros
    (OUT / "content" / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False))
    (OUT / "content" / "cards.json").write_text(json.dumps(cards, ensure_ascii=False))
    print(f"cards: {len(cards)}; macros: {sorted(all_macros)}")


if __name__ == "__main__":
    main()
