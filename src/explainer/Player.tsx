// Plays one lesson's 3D explainer: WebGL scene underneath, captions/formulas as accessible DOM on top.
// Loaded lazily (Three.js is a large chunk) from LessonExplainer.tsx.
import { useEffect, useMemo, useRef, useState } from "react";
import katex from "katex";
import { createExplainer, timeline, type Beat, type Script } from "./engine";

/** Render text with inline \( ... \) maths via KaTeX. */
function rich(text: string, macros: Record<string, string>) {
  return text.split(/(\\\(.*?\\\))/g).map((part) => part.startsWith("\\(")
    ? katex.renderToString(part.slice(2, -2), { throwOnError: false, macros: { ...macros } })
    : part.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]!)).join("");
}

function Overlay({ beat, frac, macros }: { beat: Beat; frac: number; macros: Record<string, string> }) {
  const op = Math.min(1, frac * 5, (1 - frac) * 8);
  if (beat.kind === "title" || beat.kind === "outro") {
    return (
      <div className="ex-title" style={{ opacity: op }}>
        <div className="ex-kicker">{beat.kind === "title" ? "3D explainer" : "That's the idea"}</div>
        <h2>{beat.title}</h2>
        {beat.text && <p>{beat.text}</p>}
      </div>
    );
  }
  if (beat.kind === "formula" && beat.tex) {
    const html = katex.renderToString(beat.tex, { displayMode: true, throwOnError: false, macros: { ...macros } });
    return (
      <div className="ex-panel ex-formula" style={{ opacity: op }}>
        <div className="ex-kicker">{beat.title}</div>
        <div dangerouslySetInnerHTML={{ __html: html }} />
      </div>
    );
  }
  if (beat.kind === "ideas" && beat.items) {
    return (
      <div className="ex-panel ex-ideas" style={{ opacity: op }}>
        <div className="ex-kicker">{beat.title}</div>
        <ul>{beat.items.map((s, k) => (
          <li key={k} style={{ opacity: Math.min(1, Math.max(0, frac * 6 - k * 0.8)) }} dangerouslySetInnerHTML={{ __html: rich(s, macros) }} />
        ))}</ul>
      </div>
    );
  }
  return (
    <div className={`ex-caption ${beat.kind === "pitfall" ? "warn" : ""}`} style={{ opacity: op }}>
      {beat.title && <div className="ex-kicker">{beat.kind === "pitfall" ? "⚠️ " : ""}{beat.title}</div>}
      <p dangerouslySetInnerHTML={{ __html: rich(beat.text, macros) }} />
    </div>
  );
}

export default function Player({ script, seed, macros }: { script: Script; seed: number; macros: Record<string, string> }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const ex = useRef<ReturnType<typeof createExplainer> | null>(null);
  const tl = useMemo(() => timeline(script.beats), [script]);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(true);
  const tRef = useRef(0), playRef = useRef(true);
  playRef.current = playing;

  useEffect(() => {
    const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const accent = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#7c3aed";
    const e = createExplainer(canvas.current!, script, { seed, accent, still });
    ex.current = e;
    let raf = 0, last = performance.now(), shown = -1;
    const loop = (now: number) => {
      if (playRef.current) {
        tRef.current = Math.min(e.duration, tRef.current + (now - last) / 1000);
        if (tRef.current >= e.duration) setPlaying(false);
      }
      last = now;
      e.renderAt(tRef.current);
      const tenth = Math.round(tRef.current * 10);
      if (tenth !== shown) { shown = tenth; setT(tRef.current); }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    const ro = new ResizeObserver(() => e.resize());
    ro.observe(canvas.current!);
    return () => { cancelAnimationFrame(raf); ro.disconnect(); e.dispose(); ex.current = null; };
  }, [script, seed]);

  const { i, frac } = tl.at(t);
  const seek = (v: number) => { tRef.current = v; setT(v); };
  const toggle = () => {
    if (!playing && tRef.current >= tl.duration - 0.05) seek(0);
    setPlaying(!playing);
  };

  return (
    <div className="ex-player" tabIndex={0} aria-label="3D explainer"
      onKeyDown={(e) => {
        if (e.key === " " || e.key === "k") { e.preventDefault(); toggle(); }
        if (e.key === "ArrowRight") seek(Math.min(tl.duration, tRef.current + 5));
        if (e.key === "ArrowLeft") seek(Math.max(0, tRef.current - 5));
      }}>
      <div className="ex-stage">
        <canvas ref={canvas} aria-hidden="true" />
        <div className="ex-overlay" aria-live="polite"><Overlay beat={script.beats[i]} frac={frac} macros={macros} /></div>
      </div>
      <div className="ex-controls">
        <button onClick={toggle} aria-label={playing ? "Pause" : "Play"}>{playing ? "❚❚" : "▶"}</button>
        <input type="range" min={0} max={tl.duration} step={0.1} value={t} aria-label="Seek"
          onChange={(e) => seek(Number(e.target.value))} />
        <span className="muted small">{Math.floor(t)}s / {Math.round(tl.duration)}s</span>
        <div className="ex-chapters">
          {script.beats.map((b, k) => (
            <button key={k} className={k === i ? "on" : ""} title={b.title || b.kind} aria-label={`Jump to ${b.title || b.kind}`}
              onClick={() => seek(tl.starts[k] + 0.01)} />
          ))}
        </div>
      </div>
    </div>
  );
}
