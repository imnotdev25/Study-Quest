// Coding playground: 60 tested exercises across tracks, plus a free scratchpad. Runs with your local Python.
import { useMemo, useState } from "react";
import type { Data, Exercise } from "../lib/data";
import { miniMd } from "../lib/render";
import { addXP, setState, toast, useStore } from "../lib/store";
import { call, inTauri, type RunOutput } from "../lib/tauri";
import { Editor } from "../components/Editor";
import { confirmDialog } from "../components/Dialog";
import { MOD } from "../lib/platform";
import { go } from "../App";

type Result = { name: string; ok: boolean; error: string; ms: number };
const TROLLS = [
  "Three red runs in a row. The tests are not the problem. Probably.",
  "Have you tried turning your algorithm off and on again? (Hint button is right there.)",
  "Your loss is going up. That's the wrong direction, friend.",
  "Byte has seen this bug before. Byte will not tell you. Byte is an otter.",
  "Off-by-one? It's always off-by-one.",
];
const SCRATCH = `import numpy as np

# Free scratchpad: anything runs here with your local Python (numpy, torch available in the repo venv).
x = np.linspace(-3, 3, 7)
print("softmax:", np.round(np.exp(x) / np.exp(x).sum(), 3))
`;

export function Playground({ data, id }: { data: Data; id?: string }) {
  const s = useStore();
  const tracks = useMemo(() => [...new Set(data.exercises.map((e) => e.track))], [data]);
  const ex: Exercise | undefined = id === "scratch" ? undefined : data.exercises.find((e) => e.id === id) ?? data.exercises[0];
  const scratch = id === "scratch";
  const saved = ex ? s.ex[ex.id] : undefined;
  const [code, setCode] = useState("");
  const [results, setResults] = useState<Result[] | null>(null);
  const [out, setOut] = useState("");
  const [hints, setHints] = useState(0);
  const [busy, setBusy] = useState(false);
  const key = scratch ? "scratch" : ex!.id;
  const current = code || (scratch ? localStorage.getItem("sq.scratch") ?? SCRATCH : saved?.code ?? ex!.starter);

  const open = (nid: string) => { setCode(""); setResults(null); setOut(""); setHints(0); go({ v: "play", id: nid }); };
  const save = (v: string) => {
    setCode(v);
    if (scratch) localStorage.setItem("sq.scratch", v);
    else setState((x) => ({ ...x, ex: { ...x.ex, [ex!.id]: { ...(x.ex[ex!.id] ?? { passed: false, attempts: 0, fails: 0 }), code: v } } }));
  };

  const run = async () => {
    if (busy) return;
    setBusy(true); setOut("");
    try {
      if (scratch) {
        const r = await call<RunOutput>("run_scratch", { code: current });
        setOut((r.stdout + (r.stderr ? `\n${r.stderr}` : "") + (r.timed_out ? "\n⏰ stopped after 30 s" : "")).trim() || "(no output)");
        return;
      }
      const r = await call<RunOutput>("run_tests", { code: current, tests: ex!.tests });
      const lines = r.stdout.split("\n").filter((l) => l.startsWith("{")).map((l) => JSON.parse(l));
      const tests = lines.filter((l) => l.type === "test") as Result[];
      const summary = lines.find((l) => l.type === "summary");
      const load = lines.find((l) => l.type === "load_error");
      setResults(tests);
      setOut([load?.error, summary?.stdout, r.timed_out ? "⏰ stopped after 20 s (infinite loop?)" : "", !lines.length ? r.stderr : ""]
        .filter(Boolean).join("\n").trim());
      const passed = tests.length > 0 && tests.every((t) => t.ok);
      const prev = s.ex[ex!.id] ?? { code: current, passed: false, attempts: 0, fails: 0 };
      const fails = passed ? 0 : prev.fails + 1;
      setState((x) => ({ ...x, ex: { ...x.ex, [ex!.id]: { ...prev, code: current, attempts: prev.attempts + 1, fails, passed: prev.passed || passed } } }));
      if (passed && !prev.passed) {
        const xp = prev.peeked ? 5 : [0, 15, 25, 40][ex!.level];
        addXP(xp, `Solved ${ex!.title}${prev.peeked ? " (with a peek)" : ""}`, (x) => ({ ...x, today: { ...x.today, exercises: x.today.exercises + 1 } }));
        toast({ kind: "praise", title: "✅ All tests green!", body: prev.attempts === 0 ? "First try. Show-off." : "Persistence pays." });
      } else if (!passed && fails >= 3 && s.settings.trollLevel > 0) {
        toast({ kind: "nudge", title: "🦦 Byte here", body: TROLLS[fails % TROLLS.length] });
      }
    } catch (e) {
      setOut(String(e));
    } finally { setBusy(false); }
  };

  const peek = async () => {
    if (!(await confirmDialog("Show the solution?", "You'll get 5 XP instead of the full reward.", { ok: "Show it" }))) return;
    setState((x) => ({ ...x, ex: { ...x.ex, [ex!.id]: { ...(x.ex[ex!.id] ?? { passed: false, attempts: 0, fails: 0, code: current }), peeked: true, code: ex!.solution } } }));
    setCode(ex!.solution);
  };
  const reset = () => { setState((x) => ({ ...x, ex: { ...x.ex, [ex!.id]: { ...(x.ex[ex!.id] ?? { passed: false, attempts: 0, fails: 0 }), code: ex!.starter } } })); setCode(ex!.starter); setResults(null); };
  const solved = data.exercises.filter((e) => s.ex[e.id]?.passed).length;

  return (
    <div className="play">
      <aside className="play-list">
        <div className="play-head">🧪 {solved}/{data.exercises.length} solved</div>
        <button className={`ex-item ${scratch ? "on" : ""}`} onClick={() => open("scratch")}>📝 Scratchpad</button>
        {tracks.map((tr) => (
          <div key={tr}>
            <div className="ex-track">{tr}</div>
            {data.exercises.filter((e) => e.track === tr).map((e) => (
              <button key={e.id} className={`ex-item ${e.id === ex?.id && !scratch ? "on" : ""}`} onClick={() => open(e.id)}>
                <span>{s.ex[e.id]?.passed ? "✅" : "⬜"}</span><span className="grow">{e.title}</span>
                <small>{"●".repeat(e.level)}</small>
              </button>
            ))}
          </div>
        ))}
      </aside>
      <section className="play-main">
        {!inTauri() && <div className="warn">Running code needs the desktop app (`pnpm tauri dev`).</div>}
        {ex && !scratch && (
          <div className="ex-prompt">
            <h2>{ex.title} <small className="muted">{ex.track} · {"●".repeat(ex.level)}</small></h2>
            <div dangerouslySetInnerHTML={{ __html: miniMd(ex.prompt) }} />
            {ex.hints.slice(0, hints).map((h, i) => <p key={i} className="hint">💡 {h}</p>)}
          </div>
        )}
        {scratch && <div className="ex-prompt"><h2>Scratchpad</h2><p className="muted">Free Python. Output appears below.</p></div>}
        <Editor docKey={key} value={current} onChange={save} onRun={run} />
        <div className="row toolbar">
          <button className="primary" onClick={run} disabled={busy}>{busy ? "Running…" : scratch ? `▶ Run (${MOD}↵)` : `▶ Run tests (${MOD}↵)`}</button>
          {ex && !scratch && <>
            <button onClick={() => setHints(Math.min(ex.hints.length, hints + 1))} disabled={hints >= ex.hints.length}>💡 Hint</button>
            <button onClick={peek}>🙈 Solution</button>
            <button onClick={reset}>↺ Reset</button>
          </>}
        </div>
        {results && (
          <div className="results">
            {results.map((r) => (
              <div key={r.name} className={r.ok ? "res ok" : "res bad"}>
                <span>{r.ok ? "✓" : "✗"}</span><code>{r.name}</code><span className="grow muted">{r.error}</span><small>{r.ms} ms</small>
              </div>
            ))}
          </div>
        )}
        {out && <pre className="console">{out}</pre>}
      </section>
    </div>
  );
}
