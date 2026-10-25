// A marimo notebook embedded right inside a lesson. Starts the local server on demand.
import { useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import type { Notebook } from "../lib/data";
import { addXP } from "../lib/store";
import { call, inTauri } from "../lib/tauri";

export function InlineNotebook({ nb, autoOpen = false }: { nb: Notebook; autoOpen?: boolean }) {
  const [url, setUrl] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState(autoOpen);
  const [tall, setTall] = useState(false);

  const starting = useRef(false);
  const start = async () => {
    setOpen(true);
    if (url || starting.current) return;
    if (!inTauri()) { setErr(`Notebooks run in the desktop app. In a terminal: marimo edit ~/.study-quest/notebooks/${nb.file}`); return; }
    starting.current = true;
    setBusy(true); setErr("");
    try {
      setUrl(await call<string>("start_marimo", { file: nb.file }));
      addXP(5, `Opened notebook: ${nb.title}`);
    } catch (e) { setErr(String(e)); }
    finally { setBusy(false); starting.current = false; }
  };
  // marimo was stopped (window closed or app quitting): drop the dead page and show "Run" again.
  useEffect(() => {
    if (!inTauri()) return;
    const un = listen("marimo-stopped", () => { setUrl(""); setOpen(false); });
    return () => { un.then((f) => f()); };
  }, []);
  // Opened from the picker or just created: start marimo right away instead of showing an empty card.
  useEffect(() => { if (autoOpen) start(); }, [autoOpen, nb.file]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <section className="card inline-nb" id={`nb-${nb.file}`}>
      <div className="inline-nb-head">
        <span className="tile-emoji">{nb.emoji}</span>
        <div className="grow"><b>{nb.title}</b><div className="muted small">{nb.desc}</div></div>
        {(!open || (!url && !busy)) && <button className="primary" onClick={start}>{err ? "↻ Try again" : "▶ Run notebook here"}</button>}
        {open && url && <>
          <button onClick={() => setTall(!tall)}>{tall ? "Shrink" : "Taller"}</button>
          {url && <button onClick={() => call("open_local", { url })}>Browser ↗</button>}
          <button onClick={() => setOpen(false)}>Hide</button>
        </>}
      </div>
      {open && busy && <div className="loading">Starting marimo… (first start takes a few seconds)</div>}
      {open && err && <div className="warn">{err}</div>}
      {open && url && <iframe title={nb.title} src={url} className="inline-nb-frame" style={{ height: tall ? "85vh" : 560 }} />}
    </section>
  );
}
