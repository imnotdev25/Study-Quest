#!/usr/bin/env python3
"""Build a 3D explainer script for every lesson on the RL, LLM and System Design paths.

Reads public/content/catalog.json and each topic's HTML, and writes public/content/explainers.json:
    { "<topic id>": { "motif": "...", "labels": [...], "beats": [ {kind, title, text, tex} ... ] } }
The app turns each entry into a live Three.js scene (src/explainer/), so ~400 lessons cost a few hundred KB
instead of ~400 rendered mp4s.

Usage: python3 scripts/build_explainers.py      (stdlib only; run after build_content.py / build_guides.py)
"""
import html
import json
import pathlib
import re
from html.parser import HTMLParser

APP = pathlib.Path(__file__).resolve().parents[1]
CONTENT = APP / "public/content"
BOOKS = {"rl": "rl", "rlenv": "rl", "llm": "llm", "post": "llm", "sd": "sd", "dsa": "sd"}
SKIP_HEAD = re.compile(r"retrieval|solution|answers?\b|checks? for understanding|exercise|minutes|quick check|"
                       r"further reading|references|self-check|try it|^summary$|open after|hints?\b|common mistakes|starter", re.I)


class Flat(HTMLParser):
    """Flatten a lesson into (kind, box, text) events: head, p, li, math (display), outside <pre>."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.events, self.boxes, self.buf, self.cur, self.pre, self.math = [], [], [], None, 0, None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class", "") or ""
        if tag == "div":
            m = re.search(r"box-(\w+)", cls)
            self.boxes.append(m[1] if m else None)
        elif tag == "pre":
            self.pre += 1
        elif tag in ("h1", "h2", "h3", "h4", "p", "li") and not self.pre:
            self.flush(); self.cur = "head" if tag[0] == "h" else tag
        elif tag == "span" and "math display" in cls:
            self.math = []

    def handle_endtag(self, tag):
        if tag == "div" and self.boxes:
            self.flush(); self.boxes.pop()
        elif tag == "pre":
            self.pre = max(0, self.pre - 1)
        elif tag in ("h1", "h2", "h3", "h4", "p", "li"):
            self.flush()
        elif tag == "span" and self.math is not None:
            tex = "".join(self.math).strip()
            self.events.append(("math", self.box(), re.sub(r"^\\\[|\\\]$", "", tex).strip()))
            self.math = None

    def handle_data(self, d):
        if self.pre:
            return
        if self.math is not None:
            self.math.append(d)
        elif self.cur:
            self.buf.append(d)

    def box(self):
        return next((b for b in reversed(self.boxes) if b), None)

    def flush(self):
        if self.cur:
            text = re.sub(r"\s+", " ", "".join(self.buf)).strip()
            if text:
                self.events.append((self.cur, self.box(), text))
        self.cur, self.buf = None, []


def short(text, n=170):
    """First sentence, or a word-boundary cut, never splitting an inline \\( ... \\) formula."""
    text = re.sub(r"^\d+[.)]\s*", "", text).strip()
    sents = re.findall(r".+?[.!?](?=\s|$)|.+$", text)
    out = ""
    for s_ in sents:                                   # up to two sentences that fit
        if out and len(out) + len(s_) > n:
            break
        out = (out + " " + s_.strip()).strip()
        if len(out) > 60:
            break
    if len(out) > n:
        out = out[:n].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    if out.count("\\(") > out.count("\\)"):
        out = out[:out.rfind("\\(")].rstrip() + "…"
    return out


def usable(t):
    """A paragraph worth showing: long enough, not just a lead-in to a list/equation, not mostly maths."""
    t = t.strip()
    return len(t) > 45 and not t.endswith(":") and t.count("\\(") <= 3


def heading(text):
    return re.sub(r"^(\d+(\.\d+)*[.)]?|Step \d+[.:]?)\s*", "", text).strip(" :")


MOTIFS = [   # (motif, keywords) checked in order against title + headings; first hit wins
    ("tokens", r"attention|transformer|token|embedding|context|kv cache|rope|position|decod|prompt|bpe"),
    ("landscape", r"gradient|backprop|optimi|learning rate|loss\b|adam|momentum|descent|curvature|clipping|lora"),
    ("grid", r"value iteration|value function|bellman|td\(|q-learning|q-table|sarsa|monte carlo|dynamic programming|mdp"),
    ("bars", r"bandit|softmax|policy gradient|reinforce|ppo|grpo|dapo|gspo|dpo|kl\b|preference|sampling|temperature|"
             r"calibration|probabilit|reward model|best-of"),
    ("dist", r"distribution|variance|bootstrap|statistic|estimat|confidence|evaluation|power|e-value|cuped|"
             r"importance sampling|off-policy|risk|uncertainty|gaussian|moment|likelihood"),
    ("stack", r"architecture|mixture of experts|moe|layer|kernel|gpu|inference|serving|quantiz|distributed|"
              r"fsdp|parallel|flash|batch|engine|llama|bert|gpt|deepseek"),
    ("graph", r"graph|tree|bfs|dfs|trie|heap|linked list|dijkstra|topolog|union"),
    ("array", r"array|sort|binary search|two pointer|sliding window|hash|stack|queue|interval|dynamic program|string"),
    ("pipeline", r"pipeline|verifier|sandbox|environment|agent|tool|harness|data|cache|database|load balanc|"
                 r"queue|shard|replica|rate limit|cdn|microservice|api|storage|stream|search|design"),
]
DEFAULT = {"rl": "loop", "llm": "stack", "sd": "pipeline"}
SD_PARTS = [  # pipeline labels picked from what the lesson mentions
    ("client", "Client"), ("cdn", "CDN"), ("load balanc", "Load balancer"), ("gateway", "API gateway"),
    ("rate limit", "Rate limiter"), ("service", "Service"), ("cache", "Cache"), ("queue", "Queue"),
    ("kafka", "Kafka"), ("database", "Database"), ("shard", "Shards"), ("replica", "Replicas"),
    ("index", "Index"), ("search", "Search"), ("storage", "Object store"), ("worker", "Workers"),
]
PIPE_DEFAULTS = {
    "rl": ["Task", "Policy", "Sandbox", "Verifier", "Reward"],
    "llm": ["Crawl", "Filter", "Tokenize", "Train", "Evaluate"],
    "sd": ["Client", "Load balancer", "Service", "Cache", "Database"],
}


def pick_motif(path, book, text):
    t = text.lower()
    for motif, pat in MOTIFS:
        if path == "sd" and motif in ("tokens", "landscape", "grid", "bars", "dist", "stack"):
            continue                              # system design lessons use pipeline / graph / array
        if path != "sd" and motif in ("graph", "array"):
            continue
        if re.search(pat, t):
            return motif
    return DEFAULT[path]


def labels_for(path, book, motif, text):
    t = text.lower()
    if motif == "pipeline":
        if path == "sd":
            found = [lab for key, lab in SD_PARTS if key in t]
            out = (["Client"] if "Client" not in found else []) + found
            return (out + [x for x in PIPE_DEFAULTS["sd"] if x not in out])[:5]
        return PIPE_DEFAULTS["rl" if book in ("rl", "rlenv") else "llm"]
    if motif == "loop":
        if re.search(r"tool|agent", t):
            return ["Agent", "Tool call", "Environment", "Observation"]
        return ["Agent", "Action", "Environment", "Reward"]
    return []


def build(topic, book, path):
    raw = (CONTENT / book / f"{topic['idx']}.html").read_text()
    f = Flat(); f.feed(raw); f.flush()
    ev = f.events
    beats = [{"kind": "title", "title": topic["title"], "text": ""}]
    intro = next(((b, t) for k, b, t in ev if k == "p" and b in ("analogy", "beginner", "chapterintro", "startbox")
                  and usable(t)), None)
    if intro:
        beats.append({"kind": "analogy", "title": "Picture it" if intro[0] == "analogy" else "In short",
                      "text": short(intro[1])})
    sections = []
    for i, (k, b, t) in enumerate(ev):
        if k != "head" or SKIP_HEAD.search(t) or len(t) > 90:
            continue
        body = next((tt for kk, bb, tt in ev[i + 1:i + 8] if kk == "p" and usable(tt)), None)
        if body and heading(t) and heading(t) != topic["title"]:
            sections.append({"kind": "section", "title": heading(t), "text": short(body)})
    if not sections:   # lessons without headings: use their first substantial paragraphs
        sections = [{"kind": "section", "title": "", "text": short(t)} for k, b, t in ev
                    if k == "p" and usable(t) and not t.rstrip().endswith("?")][:4]
    seen = {b["text"] for b in beats}
    sections = [x for x in sections if x["text"] not in seen]
    step = max(1, len(sections) // 4)
    beats += sections[::step][:4]
    tex = next((t for k, b, t in ev if k == "math" and 6 < len(t) < 110 and "\\begin" not in t), None)
    if tex:
        beats.append({"kind": "formula", "title": "The key equation", "text": "", "tex": re.sub(r"\s+", " ", tex)})
    ideas = [short(t, 95) for k, b, t in ev if k == "li" and b in ("keyideas", "takeaways")][:4]
    if ideas:
        beats.append({"kind": "ideas", "title": "Key ideas", "text": "", "items": ideas})
    pit = next((t for k, b, t in ev if k == "p" and b == "pitfall"), None)
    if pit:
        beats.append({"kind": "pitfall", "title": "Watch out", "text": short(pit)})
    beats.append({"kind": "outro", "title": topic["title"], "text": "Now read the lesson and run the code."})
    text = topic["title"] + " " + " ".join(topic.get("heads", []))
    motif = pick_motif(path, book, text)
    return {"motif": motif, "labels": labels_for(path, book, motif, text + " " + topic.get("search", "")), "beats": beats}


def main():
    cat = json.loads((CONTENT / "catalog.json").read_text())
    out, counts = {}, {}
    for b in cat["books"]:
        path = BOOKS.get(b["id"])
        if not path:
            continue
        for p in b["parts"]:
            for t in p["topics"]:
                e = build(t, b["id"], path)
                out[t["id"]] = e
                counts[e["motif"]] = counts.get(e["motif"], 0) + 1
    (CONTENT / "explainers.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    size = (CONTENT / "explainers.json").stat().st_size
    nb = [len(e["beats"]) for e in out.values()]
    print(f"{len(out)} explainers, {size / 1024:.0f} KB, beats per lesson {min(nb)}-{max(nb)} "
          f"(mean {sum(nb) / len(nb):.1f}); motifs {dict(sorted(counts.items(), key=lambda x: -x[1]))}")


if __name__ == "__main__":
    main()
