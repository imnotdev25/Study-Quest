// Sidebar: one learning path at a time.
//   brand + streak · search · nav · path switcher · path card (progress ring + continue) · book tabs · parts → topic timeline
import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { PATHS, pathOf, pathTopicIds, type Data } from "../lib/data";
import { liveStreak, setPath, useStore } from "../lib/store";
import { Icon, Logo, Ring, StatusDot, type IconName } from "./Icon";
import { MOD } from "../lib/platform";
import { go, type Route } from "../App";

const NAV: [Route, IconName, string][] = [
  [{ v: "home" }, "home", "Home"], [{ v: "play" }, "flask", "Playground"], [{ v: "notebooks" }, "notebook", "Notebooks"], [{ v: "draw" }, "pen", "Diagrams"],
  [{ v: "plan" }, "calendar", "Plan"], [{ v: "settings" }, "sliders", "Settings"],
];

export function Sidebar({ data, route, onSearch }: { data: Data; route: Route; onSearch: () => void }) {
  const s = useStore();
  const current = route.v === "topic" ? route.id : "";
  const pathId = s.settings.path || "rl";
  const path = PATHS.find((p) => p.id === pathId) ?? PATHS[0];
  const books = data.books.filter((b) => path.books.includes(b.id));
  const [bookId, setBookId] = useState(books[0]?.id);
  const book = books.find((b) => b.id === bookId) ?? books[0];
  const [openPart, setOpenPart] = useState("");

  // Follow the topic you're reading: right path, right book, its part open.
  useEffect(() => {
    if (!current) return;
    if (pathOf(current) !== pathId) setPath(pathOf(current));
    const bid = current.split(":")[0];
    setBookId(bid);
    const p = data.books.find((x) => x.id === bid)?.parts.find((x) => x.topics.some((t) => t.id === current));
    if (p) setOpenPart(`${bid}/${p.title}`);
  }, [current, data, pathId]);
  useEffect(() => { if (!books.some((b) => b.id === bookId)) setBookId(books[0]?.id); }, [pathId]); // eslint-disable-line

  const ids = useMemo(() => pathTopicIds(data, pathId), [data, pathId]);
  const doneN = ids.filter((id) => s.done[id]).length;
  const last = s.lastByPath[pathId];
  const nextId = last && !s.done[last] ? last : ids.find((id) => !s.done[id]) ?? ids[0];
  const next = nextId ? data.topics.get(nextId) : undefined;
  const streak = liveStreak(s);
  const accent = { "--path": path.color } as CSSProperties;

  return (
    <aside className="sidebar" style={accent}>
      <div className="drag" data-tauri-drag-region />
      <div className="brand" data-tauri-drag-region>
        <Logo size={26} />
        <span className="grow">Study Quest</span>
        <span className={`streak-chip ${streak ? "hot" : ""}`} title={`${streak}-day streak`}>
          <Icon name="flame" size={13} stroke={2.2} />{streak}
        </span>
      </div>

      <button className="searchbtn" onClick={onSearch}>
        <Icon name="search" size={14} /><span className="grow">Search</span><kbd>{MOD}K</kbd>
      </button>

      <nav className="nav-list" aria-label="Main">
        {NAV.map(([r, ic, label]) => (
          <button key={label} className={`nav ${route.v === r.v ? "on" : ""}`} onClick={() => go(r)}>
            <Icon name={ic} size={16} /><span className="grow">{label}</span>
          </button>
        ))}
      </nav>

      <div className="section-label">Learning path</div>
      <div className="path-switch" role="tablist" aria-label="Learning path">
        {PATHS.map((p) => (
          <button key={p.id} role="tab" aria-selected={p.id === pathId} title={p.long}
            className={p.id === pathId ? "on" : ""} style={{ "--path": p.color } as CSSProperties} onClick={() => setPath(p.id)}>
            <Icon name={p.ic} size={15} /><span>{p.title}</span>
          </button>
        ))}
      </div>

      <div className="path-card">
        <Ring value={doneN / Math.max(1, ids.length)} size={44} stroke={5} color={path.color}
          label={<b>{Math.round((doneN / Math.max(1, ids.length)) * 100)}%</b>} />
        <div className="path-card-text">
          <b>{path.long}</b>
          <small className="muted">{doneN} of {ids.length} topics done</small>
        </div>
      </div>
      {next && (
        <button className="continue" onClick={() => go({ v: "topic", id: next.id })} title={next.title}>
          <span className="continue-play"><Icon name="play" size={11} stroke={0} style={{ fill: "currentColor" }} /></span>
          <span className="continue-text"><small>{last && !s.done[last] ? "Continue" : "Up next"}</small><span>{next.title}</span></span>
          <Icon name="arrow" size={14} />
        </button>
      )}

      {books.length > 1 && (
        <div className="book-tabs" role="tablist">
          {books.map((b) => {
            const ts = b.parts.flatMap((p) => p.topics), n = ts.filter((t) => s.done[t.id]).length;
            return (
              <button key={b.id} role="tab" aria-selected={b.id === book?.id} className={b.id === book?.id ? "on" : ""}
                onClick={() => setBookId(b.id)} title={b.title}>
                <Icon name="book" size={13} /><span className="grow">{b.title.split(":")[0]}</span><small>{n}/{ts.length}</small>
              </button>
            );
          })}
        </div>
      )}

      <div className="tree">
        {book?.parts.map((p) => {
          const key = `${book.id}/${p.title}`, open = openPart === key;
          const n = p.topics.filter((t) => s.done[t.id]).length;
          const hasCurrent = p.topics.some((t) => t.id === current);
          return (
            <div key={key} className={`part ${open ? "open" : ""} ${hasCurrent ? "has-current" : ""}`}>
              <button className="part-btn" aria-expanded={open} onClick={() => setOpenPart(open ? "" : key)}>
                <Icon name="chevron" size={13} className="caret" />
                <span className="part-title">{p.title}</span>
                <Ring value={n / p.topics.length} size={18} stroke={2.5} color={path.color} />
              </button>
              {open && (
                <ol className="timeline">
                  {p.topics.map((t) => {
                    const state = s.done[t.id] ? "done" : s.visited[t.id] ? "started" : "todo";
                    return (
                      <li key={t.id}>
                        <button className={`topic-row ${t.id === current ? "on" : ""} ${state}`} onClick={() => go({ v: "topic", id: t.id })}>
                          <StatusDot state={state} />
                          <span className="topic-title">{t.title}</span>
                          <small className="topic-min">{t.minutes}m</small>
                        </button>
                      </li>
                    );
                  })}
                </ol>
              )}
            </div>
          );
        })}
      </div>
    </aside>
  );
}
