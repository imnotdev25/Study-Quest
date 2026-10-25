// Diagrams: your Excalidraw boards (one per System Design lesson you sketched in, plus free boards).
import { lazy, Suspense, useState } from "react";
import type { Data } from "../lib/data";
import { boardKey, deleteBoard, listBoards, newBoardId, touchBoard } from "../lib/boards";
import { confirmDialog, promptDialog } from "../components/Dialog";
import { toast } from "../lib/store";
import { call, inTauri } from "../lib/tauri";
import { go } from "../App";

const Board = lazy(() => import("../components/Board"));

export async function exportBoard(id: string, title: string) {
  const json = localStorage.getItem(boardKey(id));
  if (!json) { toast({ kind: "info", title: "Nothing to export yet", body: "Draw something first." }); return; }
  const name = `${title.replace(/[^a-z0-9]+/gi, "-").replace(/^-|-$/g, "").toLowerCase() || "diagram"}.excalidraw`;
  if (!inTauri()) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([json], { type: "application/json" })); a.download = name; a.click();
    return;
  }
  try { toast({ kind: "info", title: "💾 Diagram exported", body: await call<string>("save_download", { name, contents: json }) }); }
  catch (e) { toast({ kind: "error", title: "Export failed", body: String(e) }); }
}

export function BoardFrame({ id, title, height }: { id: string; title: string; height?: number | string }) {
  return (
    <Suspense fallback={<div className="loading">Loading the drawing board…</div>}>
      <Board id={id} title={title} height={height} />
    </Suspense>
  );
}

export function Diagrams({ data, id }: { data: Data; id?: string }) {
  const [list, setList] = useState(listBoards);
  const open = list.find((b) => b.id === id);
  const titleFor = (bid: string) => listBoards().find((b) => b.id === bid)?.title ?? open?.title ?? "Diagram";

  if (id) {
    return (
      <div className="nb-view">
        <div className="nb-bar">
          <button onClick={() => { setList(listBoards()); go({ v: "draw" }); }}>← Diagrams</button>
          <b className="grow">✏️ {titleFor(id)}</b>
          <button onClick={() => exportBoard(id, titleFor(id))}>Export .excalidraw</button>
        </div>
        <BoardFrame id={id} title={titleFor(id)} height="100%" />
      </div>
    );
  }

  const create = async () => {
    const title = await promptDialog("New diagram", "My system design");
    if (!title) return;
    const bid = newBoardId();
    touchBoard(bid, title, 0);
    setList(listBoards());
    go({ v: "draw", id: bid });
  };
  const remove = async (bid: string, title: string) => {
    if (!(await confirmDialog(`Delete “${title}”?`, "This removes the drawing from this app. Export it first to keep a copy.",
                              { ok: "Delete", danger: true }))) return;
    deleteBoard(bid); setList(listBoards());
  };
  const sdTopics = data.order.filter((t) => t.startsWith("sd:")).slice(0, 6).map((t) => data.topics.get(t)!);

  return (
    <div className="page">
      <h1>✏️ Diagrams</h1>
      <p className="muted">Sketch architectures with Excalidraw. Every System Design lesson also has its own board at the bottom
        ("Design it"); those show up here once you draw in them. Boards are saved on this computer.</p>
      <div className="row"><button className="primary" onClick={create}>+ New diagram</button></div>
      {list.length > 0 && (
        <div className="tiles">
          {list.map((b) => (
            <div key={b.id} className="tile-wrap">
              <button className="tile" onClick={() => go({ v: "draw", id: b.id })}>
                <span className="tile-emoji">✏️</span><b>{b.title}</b>
                <small className="muted">{b.shapes} shapes · {new Date(b.updated).toLocaleDateString()}</small>
              </button>
              <button className="tile-del" title="Delete" onClick={() => remove(b.id, b.title)}>🗑</button>
            </div>
          ))}
        </div>
      )}
      {!list.length && (
        <section className="card">
          <h2>Try one of these</h2>
          <p className="muted">Open a lesson and design it yourself before reading the answer:</p>
          {sdTopics.map((t) => (
            <button key={t.id} className="quest" onClick={() => go({ v: "topic", id: t.id })}><span>🏗️</span>{t.title}</button>
          ))}
        </section>
      )}
    </div>
  );
}
