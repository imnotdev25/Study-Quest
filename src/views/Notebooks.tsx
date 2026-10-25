// Notebook gallery grouped by topic, a "New notebook" form, and the selected notebook embedded.
// Notebooks live in ~/.study-quest/notebooks (examples + mine/ for yours).
import { useEffect, useMemo, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { GROUP_ORDER, refreshNotebooks, useNotebooks, type Data, type Notebook } from "../lib/data";
import { addXP, toast } from "../lib/store";
import { call, inTauri } from "../lib/tauri";
import { confirmDialog } from "../components/Dialog";
import { FILE_MANAGER, TRASH } from "../lib/platform";
import { go } from "../App";

export async function createNotebook(d: Data, title: string, group: string, topic: string, templateKind: string) {
  const file = await call<string>("create_notebook", { title, group, topic, templateKind });
  await refreshNotebooks(d.notebooks);
  addXP(5, "Created a notebook");
  return file;
}

function NewNotebook({ data, groups, onDone }: { data: Data; groups: string[]; onDone: () => void }) {
  const [title, setTitle] = useState("");
  const [group, setGroup] = useState("My notebooks");
  const [topic, setTopic] = useState("");
  const [tpl, setTpl] = useState("blank");
  const [busy, setBusy] = useState(false);
  const make = async () => {
    setBusy(true);
    try { const f = await createNotebook(data, title, group, topic, tpl); onDone(); go({ v: "notebooks", file: f }); }
    catch (e) { toast({ kind: "error", title: "Couldn't create notebook", body: String(e) }); }
    finally { setBusy(false); }
  };
  return (
    <section className="card">
      <h2>✏️ New notebook</h2>
      <div className="form nb-form">
        <input placeholder="Title, e.g. My PPO experiments" value={title} onChange={(e) => setTitle(e.target.value)} autoFocus />
        <input list="nb-groups" placeholder="Group" value={group} onChange={(e) => setGroup(e.target.value)} />
        <datalist id="nb-groups">{groups.map((g) => <option key={g} value={g} />)}</datalist>
        <input placeholder="Topic keyword (links it to lessons), e.g. PPO" value={topic} onChange={(e) => setTopic(e.target.value)} />
        <select value={tpl} onChange={(e) => setTpl(e.target.value)}>
          <option value="blank">Blank (numpy)</option><option value="slider">Slider + plot</option><option value="torch">PyTorch</option>
        </select>
        <button className="primary" disabled={busy} onClick={make}>Create</button>
        <button onClick={onDone}>Cancel</button>
      </div>
      <p className="muted small">Saved as a plain .py file in <code>~/.study-quest/notebooks/mine/</code>. A topic keyword makes it
        show up inside matching lessons.</p>
    </section>
  );
}

export function Notebooks({ data, file }: { data: Data; file?: string }) {
  const list = useNotebooks(data);
  const [url, setUrl] = useState("");
  const [err, setErr] = useState("");
  const [creating, setCreating] = useState(false);
  const [q, setQ] = useState("");
  const nb = list.find((n) => n.file === file);

  const [restart, setRestart] = useState(0);
  const stopped = useRef(false);
  useEffect(() => {
    setUrl(""); setErr("");
    if (!nb || !inTauri()) return;
    call<string>("start_marimo", { file: nb.file }).then((u) => { setUrl(u); addXP(5, `Opened ${nb.title}`); }, (e) => setErr(String(e)));
  }, [nb, restart]);
  // marimo stops when the window closes; restart it when the window comes back to this notebook.
  useEffect(() => {
    if (!inTauri()) return;
    const un = listen("marimo-stopped", () => { setUrl(""); stopped.current = true; });
    // Only restart if marimo was actually stopped, so switching apps doesn't reload a running notebook.
    const back = () => {
      if (document.visibilityState === "visible" && stopped.current) { stopped.current = false; setRestart((n) => n + 1); }
    };
    document.addEventListener("visibilitychange", back);
    return () => { un.then((f) => f()); document.removeEventListener("visibilitychange", back); };
  }, []);

  const groups = useMemo(() => {
    const t = q.toLowerCase();
    const shown = list.filter((n) => !t || `${n.title} ${n.desc} ${n.topic} ${n.group}`.toLowerCase().includes(t));
    const by = new Map<string, Notebook[]>();
    for (const n of shown) by.set(n.group || "Other", [...(by.get(n.group || "Other") ?? []), n]);
    const names = [...by.keys()].sort((a, b) => (GROUP_ORDER.indexOf(a) + 99) % 99 - (GROUP_ORDER.indexOf(b) + 99) % 99 || a.localeCompare(b));
    return names.map((g) => [g, by.get(g)!.sort((a, b) => (a.kind === "code" ? 1 : 0) - (b.kind === "code" ? 1 : 0) || a.title.localeCompare(b.title))] as const);
  }, [list, q]);

  if (nb) {
    return (
      <div className="nb-view">
        <div className="nb-bar">
          <button onClick={() => go({ v: "notebooks" })}>← Notebooks</button>
          <b>{nb.emoji} {nb.title}</b><span className="muted grow">{nb.group}{nb.desc ? ` · ${nb.desc}` : ""}</span>
          {url && <button onClick={() => call("open_local", { url })}>Open in browser ↗</button>}
        </div>
        {!inTauri() && <div className="warn">Notebooks run through the desktop app. In a terminal: <code>marimo edit ~/.study-quest/notebooks/{nb.file}</code></div>}
        {err && <div className="warn">{err}</div>}
        {!url && !err && inTauri() && <div className="loading">Starting marimo… (first start takes a few seconds)</div>}
        {url && <iframe title={nb.title} src={url} className="nb-frame" />}
      </div>
    );
  }

  const del = async (n: Notebook) => {
    if (!(await confirmDialog(`Move “${n.title}” to the ${TRASH}?`, `You can get it back from the ${TRASH}.`, { ok: `Move to ${TRASH}`, danger: true }))) return;
    try {
      const where = await call<string>("delete_notebook", { file: n.file });
      await refreshNotebooks(data.notebooks);
      toast({ kind: "info", title: `🗑 Deleted “${n.title}”`, body: where === TRASH ? `It's in the ${TRASH} if you change your mind.` : `Moved to ${where}` });
    } catch (e) { toast({ kind: "error", title: "Delete failed", body: String(e) }); }
  };
  const restore = async () => {
    if (!(await confirmDialog("Restore the example notebooks?", "Overwrites the examples with the original versions. Your notebooks in mine/ are not touched.", { ok: "Restore" }))) return;
    try { const n = await call<number>("restore_examples"); await refreshNotebooks(data.notebooks); toast({ kind: "info", title: `Restored ${n} examples` }); }
    catch (e) { toast({ kind: "error", title: "Restore failed", body: String(e) }); }
  };

  return (
    <div className="page">
      <h1>📓 Notebooks</h1>
      <p className="muted">Reactive marimo notebooks: change a slider and every dependent cell re-runs. Stored in
        <code> ~/.study-quest/notebooks</code>; they also open inside lessons.</p>
      <div className="row">
        <button className="primary" onClick={() => setCreating(true)} disabled={!inTauri()}>+ New notebook</button>
        <button onClick={() => call("reveal_notebooks").catch(() => {})} disabled={!inTauri()}>Show in {FILE_MANAGER}</button>
        <button onClick={restore} disabled={!inTauri()}>Restore examples</button>
        <input className="grow" placeholder="Filter notebooks…" value={q} onChange={(e) => setQ(e.target.value)} />
      </div>
      {creating && <NewNotebook data={data} groups={GROUP_ORDER} onDone={() => setCreating(false)} />}
      {groups.map(([g, items]) => (
        <section key={g}>
          <h2 className="group-title">{g} <small className="muted">{items.length}</small></h2>
          <div className="tiles">
            {items.map((n) => (
              <div key={n.file} className="tile-wrap">
                <button className="tile" onClick={() => go({ v: "notebooks", file: n.file })}>
                  <span className="tile-emoji">{n.emoji}</span><b>{n.title}</b>
                  <small className="muted">{n.desc || (n.mine ? "Your notebook" : "")}</small>
                  {n.kind === "code" && <small className="tag">book code</small>}
                </button>
                {n.mine && <button className="tile-del" title="Move to Trash" onClick={() => del(n)}>🗑</button>}
              </div>
            ))}
          </div>
        </section>
      ))}
      {!groups.length && <p className="muted">No notebooks match. Make one!</p>}
    </div>
  );
}
