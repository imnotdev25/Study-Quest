// Turn every Python block in a lesson into a runnable, editable notebook cell.
// "Run" executes the cells above first (like a notebook), then shows only this cell's output and plots.
import { rehighlight } from "./render";
import { call, type RunOutput } from "./tauri";
import { MOD } from "./platform";

const MARK = "@@SQCELL";
const IMG = "@@SQIMG";
const RESET_FIGS = `__import__("sys").modules.get("matplotlib.pyplot") and __import__("sys").modules["matplotlib.pyplot"].close("all")`;
const DUMP_FIGS = `
import sys as _sq_sys
if "matplotlib.pyplot" in _sq_sys.modules:
    import io as _sq_io, base64 as _sq_b64
    _sq_plt = _sq_sys.modules["matplotlib.pyplot"]
    for _sq_n in _sq_plt.get_fignums():
        _sq_buf = _sq_io.BytesIO()
        _sq_plt.figure(_sq_n).savefig(_sq_buf, format="png", dpi=110, bbox_inches="tight")
        print("${IMG}" + _sq_b64.b64encode(_sq_buf.getvalue()).decode())
`;

function button(label: string, title: string) {
  const b = document.createElement("button");
  b.className = "cell-btn";
  b.textContent = label;
  b.title = title;
  return b;
}

export function attachCells(root: HTMLElement, onFirstRun: () => void) {
  const blocks = Array.from(root.querySelectorAll<HTMLElement>("pre.code.python"));
  const original = blocks.map((b) => b.querySelector("code")?.textContent ?? "");
  const sources = [...original];
  let ran = false;

  blocks.forEach((pre, i) => {
    const code = pre.querySelector("code") as HTMLElement;
    const bar = document.createElement("div");
    bar.className = "cell-bar";
    const label = document.createElement("span");
    label.className = "cell-label";
    label.textContent = `In [${i + 1}]`;
    const run = button("▶ Run", `Run the cells above, then this one (${MOD}↵ while editing)`);
    const alone = button("Run alone", "Run only this cell");
    const edit = button("✏️ Edit", "Edit this cell");
    const reset = button("↺", "Restore the book's code");
    reset.hidden = true;
    bar.append(label, run, alone, edit, reset);
    pre.before(bar);

    const out = document.createElement("div");
    out.className = "cell-out";
    out.hidden = true;
    pre.after(out);

    let editor: HTMLTextAreaElement | null = null;
    const commit = () => {
      if (!editor) return;
      sources[i] = editor.value;
      rehighlight(code, sources[i]);
      editor.remove();
      editor = null;
      pre.hidden = false;
      edit.textContent = "✏️ Edit";
      reset.hidden = sources[i] === original[i];
    };

    const exec = async (cumulative: boolean) => {
      commit();
      const from = cumulative ? 0 : i;
      const script = sources.slice(from, i + 1)
        .map((s, k) => `print(${JSON.stringify(MARK + (from + k))}, flush=True); ${RESET_FIGS}\n${s}`)
        .join("\n") + "\n" + DUMP_FIGS;
      out.hidden = false;
      out.replaceChildren(Object.assign(document.createElement("div"), { className: "muted", textContent: "Running…" }));
      run.disabled = alone.disabled = true;
      try {
        const r = await call<RunOutput>("run_scratch", { code: script });
        const at = r.stdout.lastIndexOf(MARK + i);
        const mine = at >= 0 ? r.stdout.slice(at + (MARK + i).length) : r.stdout;
        const text: string[] = [], imgs: string[] = [];
        for (const line of mine.split("\n")) (line.startsWith(IMG) ? imgs : text).push(line.startsWith(IMG) ? line.slice(IMG.length) : line);
        // Errors from an earlier cell mean this one never ran: say so plainly.
        const failedEarlier = at < 0 && cumulative && i > 0;
        const body = (failedEarlier ? "A cell above failed before this one could run:\n" : "") + text.join("\n").trim()
          + (r.stderr.trim() ? `\n${r.stderr.trim().split("\n").slice(-12).join("\n")}` : "")
          + (r.timed_out ? "\n⏰ stopped after 30 s" : "");
        const pres = document.createElement("pre");
        pres.className = r.stderr.trim() || r.timed_out ? "console err" : "console";
        pres.textContent = body.trim() || "(ran, no output)";
        out.replaceChildren(pres, ...imgs.map((b) => Object.assign(document.createElement("img"), { src: `data:image/png;base64,${b}`, className: "cell-img" })));
        if (!ran) { ran = true; onFirstRun(); }
      } catch (e) {
        out.replaceChildren(Object.assign(document.createElement("pre"), { className: "console err", textContent: String(e) }));
      } finally {
        run.disabled = alone.disabled = false;
      }
    };

    run.onclick = () => exec(true);
    alone.onclick = () => exec(false);
    edit.onclick = () => {
      if (editor) return commit();
      editor = document.createElement("textarea");
      editor.className = "cell-edit";
      editor.value = sources[i];
      editor.spellcheck = false;
      editor.rows = Math.min(30, sources[i].split("\n").length + 1);
      editor.onkeydown = (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); exec(true); }
        if (e.key === "Tab") { e.preventDefault(); editor!.setRangeText("    ", editor!.selectionStart, editor!.selectionEnd, "end"); }
      };
      pre.hidden = true;
      pre.after(editor);
      editor.focus();
      edit.textContent = "✓ Done";
    };
    reset.onclick = () => {
      if (editor) { editor.remove(); editor = null; pre.hidden = false; edit.textContent = "✏️ Edit"; }
      sources[i] = original[i];
      rehighlight(code, sources[i]);
      reset.hidden = true;
    };
  });
  return blocks.length;
}
