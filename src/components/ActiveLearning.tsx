// Active-learning tools around a lesson. Before reading: rate confidence, guess first.
// After: explain it back (with a key-term check), connect it to an earlier topic, flag what's confusing,
// re-rate confidence. Every text box takes voice typing.
import { useMemo, useState } from "react";
import type { Data, Topic } from "../lib/data";
import { awardOnce, setState, useStore } from "../lib/store";
import { VoiceTextarea } from "./VoiceTextarea";

const FACES = ["😶", "😕", "🙂", "😊", "😎"];

function Confidence({ id, which }: { id: string; which: "before" | "after" }) {
  const s = useStore();
  const v = s.conf[id]?.[which];
  const set = (n: number) => {
    setState((x) => ({ ...x, conf: { ...x.conf, [id]: { ...x.conf[id], [which]: n } } }));
    awardOnce(`conf-${which}:${id}`, 2, which === "before" ? "Rated what you already know" : "Re-rated your confidence");
  };
  return (
    <div className="conf">
      {FACES.map((f, i) => (
        <button key={i} className={v === i + 1 ? "on" : ""} onClick={() => set(i + 1)} title={`${i + 1}/5`}>{f}</button>
      ))}
    </div>
  );
}

/** Shown above the lesson. */
export function WarmUp({ t }: { t: Topic }) {
  const s = useStore();
  const [open, setOpen] = useState(!s.guesses[t.id] && !s.done[t.id]);
  const guess = s.guesses[t.id] ?? "";
  const prompts = t.heads.filter((h) => !/retrieval|answers|exercise|checks?$/i.test(h)).slice(0, 3);
  if (!open) {
    return (
      <button className="warmup-closed" onClick={() => setOpen(true)}>
        🔮 Warm-up {guess ? "· your guess saved" : "· predict before you read"}
      </button>
    );
  }
  return (
    <section className="card warmup">
      <div className="row"><h3 className="grow">🔮 Warm-up: predict before you read</h3><button onClick={() => setOpen(false)}>Hide</button></div>
      <p className="muted small">Guessing first, even wrongly, makes the real answer stick better. You'll compare at the end.</p>
      <div className="row"><span className="small">How well do you know this already?</span><Confidence id={t.id} which="before" /></div>
      {prompts.length > 0 && (
        <ul className="small muted">{prompts.map((p) => <li key={p}>{p}?</li>)}</ul>
      )}
      <VoiceTextarea rows={3} value={guess} placeholder={`What do you think “${t.title.slice(0, 60)}” is about? How might it work?`}
        onChange={(v) => {
          setState((x) => ({ ...x, guesses: { ...x.guesses, [t.id]: v } }));
          if (v.length >= 40) awardOnce(`guess:${t.id}`, 5, "Made a prediction before reading");
        }} />
    </section>
  );
}

/** Bold terms in the lesson: what a good explanation should mention. */
function keyTerms(html: string) {
  const out = new Set<string>();
  for (const m of html.matchAll(/<strong>([^<]{3,40})<\/strong>/g)) {
    const w = m[1].trim().replace(/[.:,]$/, "");
    if (!/^(step|input|output|note|example|answer|why|common mistakes)/i.test(w) && w.split(" ").length <= 4) out.add(w);
  }
  return [...out].slice(0, 10);
}

type Tab = "explain" | "connect" | "muddy" | "confidence";

/** Shown below the lesson. */
export function AfterRead({ t, data, html }: { t: Topic; data: Data; html: string }) {
  const s = useStore();
  const [tab, setTab] = useState<Tab>("explain");
  const id = t.id;
  const terms = useMemo(() => keyTerms(html), [html]);
  const note = s.notes[id] ?? "";
  const covered = terms.filter((w) => note.toLowerCase().includes(w.toLowerCase()));
  const pos = data.order.indexOf(id);
  const earlier = data.order.slice(0, pos).filter((x) => x.split(":")[0] === id.split(":")[0] && (s.done[x] || s.visited[x])).slice(-6).reverse();
  const [linkTo, setLinkTo] = useState(earlier[0] ?? "");
  const tabs: [Tab, string][] = [["explain", "✍️ Explain it back"], ["connect", "🔗 Connect it"], ["muddy", "🌫️ What's unclear?"],
    ["confidence", "📈 Confidence"]];

  return (
    <section className="card after">
      <h3>🧠 Make it stick</h3>
      <div className="seg tabs">{tabs.map(([k, l]) => <button key={k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>{l}</button>)}</div>

      {tab === "explain" && <>
        <p className="muted small">Explain it to a friend in your own words, without scrolling up. Then check which key terms you used.</p>
        <VoiceTextarea rows={5} value={note} placeholder="In plain words, this is about… (🎙 to say it out loud)" onChange={(v) => {
          setState((x) => ({ ...x, notes: { ...x.notes, [id]: v } }));
          if (v.length >= 80) awardOnce(`explain:${id}`, 10, "Explained it in your own words");
        }} />
        {terms.length > 0 && note.length > 20 && (
          <div className="terms">
            <small className="muted">Key terms {covered.length}/{terms.length}:</small>
            {terms.map((w) => <span key={w} className={covered.includes(w) ? "term hit" : "term"}>{covered.includes(w) ? "✓ " : ""}{w}</span>)}
          </div>
        )}
        {s.guesses[id] && <details className="compare"><summary>🔮 Compare with your guess before reading</summary><p>{s.guesses[id]}</p>
          <p className="muted small">What did you get right? What surprised you? That surprise is where the learning happened.</p></details>}
      </>}

      {tab === "connect" && <>
        <p className="muted small">New ideas stick when they hang on old ones. How does this relate to something you studied earlier?</p>
        {earlier.length ? <>
          <select value={linkTo} onChange={(e) => setLinkTo(e.target.value)}>
            {earlier.map((x) => <option key={x} value={x}>{data.topics.get(x)!.title}</option>)}
          </select>
          <VoiceTextarea rows={3} value={s.links[id] ?? ""} placeholder="This is like … because … but differs in …" onChange={(v) => {
            setState((x) => ({ ...x, links: { ...x.links, [id]: v } }));
            if (v.length >= 50) awardOnce(`link:${id}`, 8, "Connected two topics");
          }} />
        </> : <p className="muted">Open a few earlier topics in this book first, then come back to connect them.</p>}
      </>}

      {tab === "muddy" && <>
        <p className="muted small">The muddiest point: the one thing that's least clear. It goes on your dashboard so you come back to it.</p>
        <VoiceTextarea rows={3} value={s.muddy[id]?.text ?? ""} placeholder="I still don't get why…" onChange={(v) => {
          setState((x) => ({ ...x, muddy: { ...x.muddy, [id]: { text: v, at: x.muddy[id]?.at ?? Date.now() } } }));
          if (v.length >= 20) awardOnce(`muddy:${id}`, 4, "Named what's unclear (that's the hard part)");
        }} />
        {s.muddy[id]?.text && !s.muddy[id].resolved && (
          <button onClick={() => { setState((x) => ({ ...x, muddy: { ...x.muddy, [id]: { ...x.muddy[id], resolved: true } } }));
            awardOnce(`resolved:${id}`, 10, "Cleared up a muddy point"); }}>✅ I get it now</button>
        )}
      </>}

      {tab === "confidence" && <>
        <p className="muted small">How well could you explain this now, without notes?</p>
        <Confidence id={id} which="after" />
        {s.conf[id]?.before && s.conf[id]?.after && (
          <p className="small">Before {FACES[s.conf[id].before! - 1]} → after {FACES[s.conf[id].after! - 1]}
            {s.conf[id].after! - s.conf[id].before! > 0 ? " · nice jump!" : s.conf[id].after! <= 2 ? " · flag what's unclear and come back tomorrow." : ""}</p>
        )}
      </>}
    </section>
  );
}
