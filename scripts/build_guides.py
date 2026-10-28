#!/usr/bin/env python3
"""Turn the Markdown guides in study_app/guides/ into app content, next to the LaTeX books.

Each guide is one Markdown file:

    ---
    id: safety
    title: AI Safety
    icon: 🛡️
    after: rl            (optional: place the book right after this one in the catalog)
    ---
    # Part title
    ## Topic title
    Body with $math$, ```python blocks``` and boxes:
    :::keyideas            :::pitfall Optional title
    - ...                  ...
    :::                    :::

Writes public/content/<id>/<n>.html and merges the books into public/content/catalog.json, so run it after
build_content.py. `--check` also runs every topic's Python blocks in order (the same way the in-app Run button
does) and fails if any of them errors.

Usage: .venv/bin/python scripts/build_guides.py [--check]
"""
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
APP = HERE.parent
OUT = APP / "public"
sys.path.insert(0, str(HERE))
from build_content import pandoc, plain, postprocess  # noqa: E402

BOXES = {"keyideas", "analogy", "pitfall", "deeper", "walkthrough", "startbox", "takeaways", "expert", "interview"}
FENCE = re.compile(r"^```(\w*)\s*$")


class Topic:
    def __init__(self, book, idx, part, title, md):
        self.book, self.idx, self.part, self.title_tex, self.md = book, idx, part, title, md
        self.codes, self.figs = [], []


def front_matter(src):
    m = re.match(r"---\n(.*?)\n---\n", src, re.S)
    meta = dict(line.split(":", 1) for line in m[1].splitlines() if ":" in line)
    return {k.strip(): v.strip() for k, v in meta.items()}, src[m.end():]


def split_guide(book, body):
    """Split on '# ' (part) and '## ' (topic) headings that are outside code fences."""
    parts, topics, cur, in_code = [], [], None, False
    for line in body.splitlines():
        if FENCE.match(line):
            in_code = not in_code
        if not in_code and line.startswith("# "):
            parts.append(line[2:].strip()); cur = None; continue
        if not in_code and line.startswith("## "):
            cur = Topic(book, len(topics), len(parts) - 1, line[3:].strip(), [])
            topics.append(cur); continue
        if cur is not None:
            cur.md.append(line)
    for t in topics:
        t.md = "\n".join(t.md).strip() + "\n"
    return parts, topics


def preprocess(t):
    """Pull code out (pandoc must not touch it) and turn ::: boxes into the markers build_content understands."""
    out, code, kind = [], None, ""
    for line in t.md.splitlines():
        m = FENCE.match(line)
        if code is None and m:
            code, kind = [], {"python": "python", "output": "output"}.get(m[1], "text")
            continue
        if code is not None:
            if line.strip() == "```":
                t.codes.append((kind, "\n".join(code)))
                out.append(f"\nXXCODE{len(t.codes) - 1}XX\n"); code = None
            else:
                code.append(line)
            continue
        b = re.match(r"^:::(\w+)\s*(.*)$", line)
        if b and b[1] in BOXES:
            out.append(f"\nXXBOXSTART {b[1]} XXBOXTITLE {b[2]} XXBOXTITLEEND\n"); continue
        if line.strip() == ":::":
            out.append("\nXXBOXCLOSEXX\n"); continue
        out.append(line)
    if code is not None:
        raise ValueError(f"{t.book}: unclosed code block in '{t.title_tex}'")
    t.pre = f"XXTITLE {t.title_tex} XXTITLEEND\n\n" + "\n".join(out)


def md_to_html(text):
    r = subprocess.run(["pandoc", "-f", "markdown-fenced_divs", "-t", "html", "--mathjax", "--wrap=none"],
                       input=text, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-500:])
    return r.stdout


def check(topics):
    """Run each topic's Python blocks top to bottom in one process, like pressing Run on the last cell."""
    bad = 0
    for t in topics:
        src = "\n\n".join(c for k, c in t.codes if k == "python")
        if not src:
            continue
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "topic.py"
            f.write_text(src)
            r = subprocess.run([sys.executable, str(f)], cwd=d, capture_output=True, text=True, timeout=120,
                               env={**os.environ, "MPLBACKEND": "Agg"})
        if r.returncode:
            bad += 1
            print(f"FAIL {t.book}:{t.idx} {t.title_tex}\n{r.stderr[-1200:]}", file=sys.stderr)
    return bad


def main():
    cat_path = OUT / "content" / "catalog.json"
    catalog = json.loads(cat_path.read_text())
    failures = 0
    for path in sorted((APP / "guides").glob("*.md")):
        meta, body = front_matter(path.read_text())
        book = meta["id"]
        parts, topics = split_guide(book, body)
        for t in topics:
            preprocess(t)
        with ThreadPoolExecutor(8) as ex:
            htmls = list(ex.map(lambda t: md_to_html(t.pre), topics))
        part_titles = [plain(md_to_html(p)) for p in parts]
        (OUT / "content" / book).mkdir(parents=True, exist_ok=True)
        entry = {"id": book, "title": meta["title"], "icon": meta.get("icon", "📘"),
                 "parts": [{"title": p, "topics": []} for p in part_titles]}
        for t, h in zip(topics, htmls):
            postprocess(t, h)
            (OUT / "content" / book / f"{t.idx}.html").write_text(t.html)
            text = plain(t.html)
            heads = [plain(x) for x in re.findall(r"<h[23][^>]*>(.*?)</h[23]>", t.html, re.S)]
            entry["parts"][t.part]["topics"].append({
                "id": f"{book}:{t.idx}", "idx": t.idx, "title": plain(t.title_html), "titleHtml": t.title_html,
                "minutes": max(3, round(len(text.split()) / 180)), "heads": heads[:30],
                "search": text[:1500].lower(), "hasCode": "<pre" in t.html, "figs": 0})
        books = [b for b in catalog["books"] if b["id"] != book]
        at = next((i + 1 for i, b in enumerate(books) if b["id"] == meta.get("after")), len(books))
        catalog["books"] = books[:at] + [entry] + books[at:]
        print(f"{book}: {len(parts)} parts, {len(topics)} topics, "
              f"{sum(k == 'python' for t in topics for k, _ in t.codes)} python blocks")
        if "--check" in sys.argv:
            failures += check(topics)
    cat_path.write_text(json.dumps(catalog, ensure_ascii=False))
    if failures:
        sys.exit(f"{failures} topic(s) have Python blocks that fail")


if __name__ == "__main__":
    main()
