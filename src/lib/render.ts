// Post-process generated HTML: KaTeX for math, highlight.js for Python, step-by-step reveal for quiz mode.
import katex from "katex";
import hljs from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";

hljs.registerLanguage("python", python);

export function renderMath(root: HTMLElement, macros: Record<string, string>) {
  root.querySelectorAll<HTMLElement>("span.math").forEach((el) => {
    if (el.dataset.done) return;
    const display = el.classList.contains("display");
    const tex = (el.textContent ?? "").trim().replace(/^\\[([]/, "").replace(/\\[)\]]$/, "");
    try {
      katex.render(tex, el, { displayMode: display, throwOnError: false, macros: { ...macros }, strict: "ignore" });
    } catch { /* leave the source visible */ }
    el.dataset.done = "1";
  });
}

export function highlight(root: HTMLElement) {
  root.querySelectorAll<HTMLElement>("pre code.language-python").forEach((el) => {
    if (!el.dataset.highlighted) hljs.highlightElement(el);
  });
}

export const renderAll = (root: HTMLElement, macros: Record<string, string>) => { renderMath(root, macros); highlight(root); };

/** Re-highlight one code element after its text changed. */
export function rehighlight(el: HTMLElement, text: string) {
  delete el.dataset.highlighted;
  el.className = "language-python";
  el.textContent = text;
  hljs.highlightElement(el);
}

/** Quiz mode: derivations and algorithms reveal one step at a time; key-idea boxes hide until you've recalled them. */
export function makeQuiz(root: HTMLElement) {
  root.querySelectorAll<HTMLElement>(".box-derive, .box-algobox, .box-expert, .box-deeper").forEach((box) => {
    const parts = Array.from(box.children).slice(1) as HTMLElement[];
    if (parts.length < 2) return;
    let shown = 1;
    parts.forEach((p, i) => (p.style.display = i < shown ? "" : "none"));
    const btn = document.createElement("button");
    btn.className = "reveal";
    const label = () => (btn.textContent = `Predict the next step, then reveal (${parts.length - shown} left)`);
    label();
    btn.onclick = () => {
      parts[shown].style.display = "";
      shown++;
      if (shown >= parts.length) btn.remove(); else label();
    };
    box.appendChild(btn);
  });
  root.querySelectorAll<HTMLElement>(".box-keyideas, .box-takeaways").forEach((box) => {
    const parts = Array.from(box.children).slice(1) as HTMLElement[];
    parts.forEach((p) => (p.style.display = "none"));
    const btn = document.createElement("button");
    btn.className = "reveal";
    btn.textContent = "Recall the key ideas out loud, then reveal";
    btn.onclick = () => { parts.forEach((p) => (p.style.display = "")); btn.remove(); };
    box.appendChild(btn);
  });
}

/** Tiny markdown for exercise prompts: `code`, **bold**, paragraphs. */
export function miniMd(s: string) {
  const esc = s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return esc.replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .split(/\n{2,}/).map((p) => `<p>${p}</p>`).join("");
}

export function ics(events: { title: string; start: number; minutes: number; desc?: string }[]) {
  const f = (ms: number) => new Date(ms).toISOString().replace(/[-:]/g, "").replace(/\.\d{3}/, "");
  const esc = (s: string) => s.replace(/\\/g, "\\\\").replace(/;/g, "\\;").replace(/,/g, "\\,").replace(/\n/g, "\\n");
  const out = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Study Quest//EN", "CALSCALE:GREGORIAN"];
  events.forEach((e, i) => out.push("BEGIN:VEVENT", `UID:sq-${e.start}-${i}@studyquest`, `DTSTAMP:${f(Date.now())}`,
    `DTSTART:${f(e.start)}`, `DTEND:${f(e.start + e.minutes * 60000)}`, `SUMMARY:${esc(e.title)}`,
    `DESCRIPTION:${esc(e.desc ?? "")}`, "END:VEVENT"));
  out.push("END:VCALENDAR");
  return out.join("\r\n");
}
