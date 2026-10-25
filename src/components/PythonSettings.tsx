// Settings → Python: use the app's uv environment (one click) or point at any interpreter and test it.
import { useEffect, useState } from "react";
import { toast } from "../lib/store";
import { call, inTauri } from "../lib/tauri";

type Probe = { ok: boolean; version?: string; executable?: string; error?: string; packages?: Record<string, boolean> };
type Info = { python: string; probe: Probe; managedVenv: string; usingManaged: boolean };

function Packages({ p }: { p: Probe }) {
  if (!p.ok) return <p className="warn small">❌ {p.error}</p>;
  return (
    <div className="pkgs">
      <span className="pkg ok">Python {p.version}</span>
      {Object.entries(p.packages ?? {}).map(([k, v]) => (
        <span key={k} className={`pkg ${v ? "ok" : k === "torch" ? "opt" : "bad"}`}>{v ? "✓" : k === "torch" ? "○" : "✕"} {k}</span>
      ))}
    </div>
  );
}

export function PythonSettings({ python, setPython }: { python: string; setPython: (v: string) => void }) {
  const [info, setInfo] = useState<Info | null>(null);
  const [draft, setDraft] = useState(python);
  const [test, setTest] = useState<Probe | null>(null);
  const [busy, setBusy] = useState("");
  const [log, setLog] = useState("");
  const refresh = () => { if (inTauri()) call<Info>("app_info").then(setInfo, () => {}); };
  useEffect(refresh, [python]);

  if (!inTauri()) return <p className="muted">Python settings are available in the desktop app.</p>;

  const setup = async (withTorch: boolean) => {
    setBusy(withTorch ? "torch" : "base"); setLog("");
    try {
      setLog(await call<string>("setup_python", { withTorch }));
      if (python) setPython("");                         // switch to the managed env
      toast({ kind: "praise", title: "🐍 Python environment ready", body: "Exercises and notebooks will use it." });
    } catch (e) { setLog(String(e)); toast({ kind: "error", title: "uv setup failed", body: String(e).split("\n")[0] }); }
    finally { setBusy(""); refresh(); }
  };
  const check = async () => { setBusy("test"); setTest(await call<Probe>("check_python", { path: draft })); setBusy(""); };
  const useCustom = () => { setPython(draft.trim()); setTest(null); toast({ kind: "info", title: "Using your Python", body: draft.trim() }); };

  return (
    <div className="pyset">
      {info && (
        <div className="py-now">
          <small className="muted">Currently using {python ? "your custom Python" : info.usingManaged ? "the app's uv environment" : "an auto-detected Python"}</small>
          <code>{info.python}</code>
          <Packages p={info.probe} />
        </div>
      )}

      <div className="py-opt">
        <b>⚡ Set up with uv</b>
        <p className="muted small">Creates <code>{info?.managedVenv ?? "~/.study-quest/.venv"}</code> with pinned marimo, numpy and matplotlib.
          uv downloads Python 3.12 if needed.</p>
        <div className="row">
          <button className="primary" disabled={!!busy} onClick={() => setup(false)}>{busy === "base" ? "Installing…" : "Set up with uv"}</button>
          <button disabled={!!busy} onClick={() => setup(true)}>{busy === "torch" ? "Installing PyTorch… (a few minutes)" : "+ PyTorch"}</button>
        </div>
        {log && <pre className="console small-console">{log}</pre>}
      </div>

      <div className="py-opt">
        <b>📍 Use my own Python</b>
        <p className="muted small">Any interpreter: a conda env, a project <code>.venv</code>, pyenv… Leave empty to use the uv environment.</p>
        <div className="row">
          <input className="grow" placeholder="/path/to/python" value={draft} onChange={(e) => { setDraft(e.target.value); setTest(null); }} />
          <button disabled={!draft.trim() || !!busy} onClick={check}>{busy === "test" ? "Testing…" : "Test"}</button>
          <button className="primary" disabled={!draft.trim() || (test !== null && !test.ok)} onClick={useCustom}>Use it</button>
          {python && <button onClick={() => { setPython(""); setDraft(""); }}>Clear</button>}
        </div>
        {test && <Packages p={test} />}
        {test?.ok && test.packages && !test.packages.marimo && (
          <p className="muted small">Missing packages: run <code>{draft.trim()} -m pip install marimo numpy matplotlib</code></p>
        )}
      </div>
    </div>
  );
}
