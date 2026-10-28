// "Watch the 3D explainer" card for a lesson. The script comes from public/content/explainers.json; the Three.js
// player is a separate lazy chunk, loaded only when someone presses play.
import { lazy, Suspense, useEffect, useState } from "react";
import type { Script } from "./engine";

const Player = lazy(() => import("./Player"));

let cache: Promise<Record<string, Script>> | null = null;
export function loadExplainers() {
  cache ??= fetch("content/explainers.json").then((r) => (r.ok ? r.json() : {}), () => ({}));
  return cache;
}
export function useExplainer(id: string) {
  const [s, setS] = useState<Script | null>(null);
  useEffect(() => { let live = true; loadExplainers().then((all) => live && setS(all[id] ?? null)); return () => { live = false; }; }, [id]);
  return s;
}

const seedOf = (id: string) => [...id].reduce((h, c) => (h * 31 + c.charCodeAt(0)) >>> 0, 7) % 1000;

export function LessonExplainer({ id, macros, open: initial = false }: { id: string; macros: Record<string, string>; open?: boolean }) {
  const script = useExplainer(id);
  const [open, setOpen] = useState(initial);
  if (!script) return null;
  const secs = Math.round(script.beats.reduce((s, b) => s + ({ title: 4, analogy: 6.5, section: 6.5, formula: 6, ideas: 7.5, pitfall: 5.5, outro: 4 })[b.kind], 0));
  return (
    <section className="card ex-card">
      <div className="row">
        <h3 className="grow">🎬 3D explainer</h3>
        <span className="muted small">{secs}s · {script.beats.length} scenes</span>
        <button className={open ? "" : "primary"} onClick={() => setOpen(!open)}>{open ? "Close" : "▶ Watch"}</button>
      </div>
      {!open && <p className="muted small">A short animated walk through this lesson: the idea, the main steps
        {script.beats.some((b) => b.kind === "formula") ? ", the key equation" : ""} and what to remember.</p>}
      {open && (
        <Suspense fallback={<div className="loading">Loading 3D…</div>}>
          <Player script={script} seed={seedOf(id)} macros={macros} />
        </Suspense>
      )}
    </section>
  );
}
