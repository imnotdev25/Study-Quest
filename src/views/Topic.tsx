// One topic: warm-up, read or quiz mode, runnable code, notebooks inline, active-learning tools, prev/next.
import { useEffect, useRef, useState } from "react";
import { pathOf, relatedExercises, relatedNotebooks, topicHtml, useNotebooks, type Data } from "../lib/data";
import { makeQuiz, renderAll } from "../lib/render";
import { addXP, getState, setState, toast, useStore } from "../lib/store";
import { inTauri } from "../lib/tauri";
import { widgetsFor } from "../components/Widgets";
import { InlineNotebook } from "../components/InlineNotebook";
import { AfterRead, WarmUp } from "../components/ActiveLearning";
import { attachCells } from "../lib/cells";
import { createNotebook } from "./Notebooks";
import { promptDialog } from "../components/Dialog";
import { hasBoard, topicBoardId } from "../lib/boards";
import { BoardFrame, exportBoard } from "./Diagrams";
import { go } from "../App";

export function TopicView({ data, id }: { data: Data; id: string }) {
  const t = data.topics.get(id);
  const s = useStore();
  const notebooks = useNotebooks(data);
  const [html, setHtml] = useState("");
  const [quiz, setQuiz] = useState(false);
  const [drawing, setDrawing] = useState(false);
  useEffect(() => setDrawing(false), [id]);
  const [extra, setExtra] = useState<string[]>([]);       // notebooks opened here from the picker
  useEffect(() => setExtra([]), [id]);
  const body = useRef<HTMLDivElement>(null);
  const title = useRef<HTMLHeadingElement>(null);

  useEffect(() => {
    setHtml("");
    topicHtml(id).then(setHtml);
    setState((x) => ({ ...x, lastTopic: id, lastByPath: { ...x.lastByPath, [pathOf(id)]: id }, visited: { ...x.visited, [id]: x.visited[id] ?? Date.now() } }));
    document.querySelector(".main")?.scrollTo(0, 0);
  }, [id]);

  useEffect(() => {
    if (!body.current || !html) return;
    body.current.innerHTML = html;
    renderAll(body.current, data.macros);
    if (quiz) makeQuiz(body.current);
    attachCells(body.current, () => addXP(3, "Ran lesson code"));
  }, [html, quiz, data.macros]);

  useEffect(() => { if (title.current && t) { title.current.innerHTML = t.titleHtml; renderAll(title.current, data.macros); } }, [t, data.macros]);

  if (!t) return <div className="page"><p>Topic not found.</p></div>;
  const pos = data.order.indexOf(id);
  const prev = data.order[pos - 1], next = data.order[pos + 1];
  const book = data.books.find((b) => b.id === t.book)!;
  const done = !!s.done[id];
  const widgets = widgetsFor(`${t.title} ${t.heads.join(" ")}`);
  const exs = relatedExercises(t, data);
  const related = relatedNotebooks(t, notebooks);
  const nbs = [...related, ...extra.map((f) => notebooks.find((n) => n.file === f)!).filter((n) => n && !related.includes(n))];
  const scrollTo = (file: string) => document.getElementById(`nb-${file}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
  const groups = [...new Set(notebooks.map((n) => n.group || "Other"))].sort();

  const finish = () => {
    if (done) return;
    addXP(20, `Finished “${t.title.slice(0, 40)}”`, (x) => ({ ...x, done: { ...x.done, [id]: Date.now() },
      today: { ...x.today, topics: x.today.topics + 1 } }));
    if (getState().today.topics === 1) setTimeout(() => toast({ kind: "praise", title: "First topic today!", body: "The streak gods are pleased." }), 600);
  };
  const newHere = async () => {
    const name = await promptDialog("New notebook for this topic", `Notes: ${t.title.slice(0, 50)}`);
    if (!name) return;
    try {
      const file = await createNotebook(data, name, "My notebooks", t.title.split(/[:(]/)[0].trim().slice(0, 60), "blank");
      setExtra((x) => [...x, file]);
      setTimeout(() => scrollTo(file), 150);
    } catch (e) { toast({ kind: "error", title: "Couldn't create notebook", body: String(e) }); }
  };

  return (
    <div className="page topic">
      <div className="crumbs">{book.icon} {book.title} · {t.part}</div>
      <h1 ref={title} />
      <div className="topic-bar">
        <span className="muted">⏱ ~{t.minutes} min{t.figs ? ` · ${t.figs} diagrams` : ""}{t.hasCode ? " · code" : ""}</span>
        <div className="seg">
          <button className={!quiz ? "on" : ""} onClick={() => setQuiz(false)}>📖 Read</button>
          <button className={quiz ? "on" : ""} onClick={() => setQuiz(true)} title="Derivations reveal one step at a time">🧠 Quiz me</button>
        </div>
        <button className={done ? "done-btn" : "primary"} onClick={finish}>{done ? "✅ Done" : "Mark done +20 XP"}</button>
      </div>

      <div className="related">
        {nbs.map((n) => <button key={n.file} className="chip" onClick={() => scrollTo(n.file)}>{n.emoji} Notebook below: {n.title}</button>)}
        {exs.map((e) => <button key={e.id} className="chip" onClick={() => go({ v: "play", id: e.id })}>🧪 Exercise: {e.title}</button>)}
        {t.hasCode && <span className="chip muted">▶ Code in this lesson runs in place</span>}
      </div>

      <WarmUp t={t} />

      {!html && <div className="loading">Loading…</div>}
      <article className="prose" ref={body} />

      {nbs.map((n) => <InlineNotebook key={n.file} nb={n} autoOpen={extra.includes(n.file)} />)}
      <div className="nb-picker">
        <span className="muted">📓 Notebook in this lesson:</span>
        <select value="" onChange={(e) => {
          const f = e.target.value;
          if (!f) return;
          setExtra((x) => (x.includes(f) ? x : [...x, f]));
          setTimeout(() => scrollTo(f), 50);
        }}>
          <option value="">Open one…</option>
          {groups.map((g) => (
            <optgroup key={g} label={g}>
              {notebooks.filter((n) => (n.group || "Other") === g).map((n) => <option key={n.file} value={n.file}>{n.emoji} {n.title}</option>)}
            </optgroup>
          ))}
        </select>
        {inTauri() && <button onClick={newHere}>+ New notebook for this topic</button>}
      </div>

      {widgets.map((w) => (
        <section key={w.id} className="card widget"><h3>{w.emoji} Try it: {w.title}</h3>{w.el()}</section>
      ))}

      {hasBoard(id) && (
        <section className="card design-it">
          <div className="row">
            <h3 className="grow">🏗️ Design it yourself</h3>
            {drawing && <button onClick={() => exportBoard(topicBoardId(id), t.title)}>Export</button>}
            {drawing && <button onClick={() => go({ v: "draw", id: topicBoardId(id) })}>Full screen</button>}
            <button className={drawing ? "" : "primary"} onClick={() => setDrawing(!drawing)}>{drawing ? "Hide board" : "✏️ Open drawing board"}</button>
          </div>
          {!drawing && <p className="muted small">Sketch the architecture before (or after) reading: clients, load balancers, services, caches,
            queues, databases. Your board for this lesson is saved and also listed under Diagrams.</p>}
          {drawing && <BoardFrame id={topicBoardId(id)} title={t.title} height={620} />}
        </section>
      )}

      {html && <AfterRead t={t} data={data} html={html} />}

      <nav className="prevnext">
        {prev ? <button onClick={() => go({ v: "topic", id: prev })}>← {data.topics.get(prev)!.title.slice(0, 50)}</button> : <span />}
        {next && <button onClick={() => { finish(); go({ v: "topic", id: next }); }}>{data.topics.get(next)!.title.slice(0, 50)} →</button>}
      </nav>
    </div>
  );
}
