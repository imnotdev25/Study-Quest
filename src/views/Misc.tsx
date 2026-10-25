// Plan (Calendar sync), Settings, and the Cmd+K search palette.
import { useEffect, useMemo, useRef, useState } from "react";
import { PATHS, pathTopicIds, type Data } from "../lib/data";
import { ics } from "../lib/render";
import { dayKey, getState, replaceState, resetState, setState, toast, useStore, type Plan } from "../lib/store";
import { confirmDialog } from "../components/Dialog";
import { addCalendarEvent, call, deleteCalendarEvent, inTauri } from "../lib/tauri";
import { isMac, TRAY } from "../lib/platform";
import { ACCENTS } from "../lib/theme";
import { PythonSettings } from "../components/PythonSettings";
import { go, type Route } from "../App";

const toLocalInput = (ms: number) => {
  const d = new Date(ms - new Date().getTimezoneOffset() * 60000);
  return d.toISOString().slice(0, 16);
};

export function PlanView({ data }: { data: Data }) {
  const s = useStore();
  const [title, setTitle] = useState("");
  const [when, setWhen] = useState(() => toLocalInput(Math.ceil(Date.now() / 36e5) * 36e5));
  const [mins, setMins] = useState(30);
  const [busy, setBusy] = useState(false);

  const [planPath, setPlanPath] = useState(s.settings.path || "rl");
  const planned = (st: typeof s) => new Set(st.plans.filter((p) => p.start > Date.now() && p.topic).map((p) => p.topic!));
  /** Next unfinished, not-yet-planned topics. "mix" rotates RL → LLMs → System Design. */
  const nextTopics = (n: number, path = planPath) => {
    const st = getState(), skip = planned(st);
    const queue = (pid: string) => pathTopicIds(data, pid).filter((id) => !st.done[id] && !skip.has(id));
    if (path !== "mix") return queue(path).slice(0, n).map((id) => data.topics.get(id)!);
    const qs = PATHS.map((p) => queue(p.id)), out: string[] = [];
    for (let i = 0; out.length < n && qs.some((q) => q.length); i++) {
      const q = qs[i % qs.length];
      if (q.length) out.push(q.shift()!);
    }
    return out.map((id) => data.topics.get(id)!);
  };
  const add = async (p: Omit<Plan, "id" | "status">) => {
    let status = "saved", uid = "", calendar = s.settings.calendarName;
    if (inTauri() && isMac) {
      try {
        const r = await addCalendarEvent(p.title, "Planned in Study Quest. Open the app and press Start.", p.start, p.minutes, calendar, true);
        status = `in ${r.calendar}`; uid = r.uid; calendar = r.calendar;
      } catch (e) { status = "not synced"; toast({ kind: "error", title: "Calendar sync failed", body: String(e) }); }
    }
    setState((x) => ({ ...x, plans: [...x.plans, { ...p, id: `${p.start}-${Math.random().toString(36).slice(2, 7)}`, status, uid, calendar }]
      .sort((a, b) => a.start - b.start) }));
    return status;
  };
  const addOne = async () => {
    setBusy(true);
    const t = title.trim() || `Study: ${nextTopics(1)[0]?.title ?? "anything"}`;
    const st = await add({ title: t, start: new Date(when).getTime(), minutes: mins });
    toast({ kind: "info", title: "📅 Study block planned", body: st });
    setTitle(""); setBusy(false);
  };
  const planWeek = async () => {
    if (busy) return;
    setBusy(true);
    try {
      const topics = nextTopics(5);
      if (!topics.length) { toast({ kind: "info", title: "Nothing left to plan", body: "Every topic in this path is done or already planned." }); return; }
      // Evenings that already have a block are skipped, so pressing this twice never double-books a day.
      const busyDays = new Set(getState().plans.filter((p) => p.start > Date.now()).map((p) => dayKey(new Date(p.start))));
      const day = new Date(); day.setHours(19, 0, 0, 0);
      let added = 0;
      for (let d = 0; added < topics.length && d < 28; d++) {
        day.setDate(day.getDate() + 1);                       // calendar days, safe across DST changes
        if (day.getDay() === 0 || day.getDay() === 6 || busyDays.has(dayKey(day))) continue;
        const t = topics[added];
        await add({ title: `Study: ${t.title}`, start: day.getTime(), minutes: s.settings.goalMinutes + 10, topic: t.id });
        busyDays.add(dayKey(day));
        added++;
      }
      toast({ kind: "praise", title: `📅 ${added} weekday sessions planned`,
              body: `7 pm, one topic each from ${planPath === "mix" ? "all three paths" : PATHS.find((p) => p.id === planPath)?.title}. Byte will remind you.` });
    } finally { setBusy(false); }
  };
  /** ✕ removes the block from the list and from Calendar (plus any duplicate with the same title and time). */
  const removePlan = async (p: Plan) => {
    const drop = () => setState((x) => ({ ...x, plans: x.plans.filter((q) => q.id !== p.id) }));
    if (!inTauri() || !isMac || p.status === "not synced" || p.status === "saved") { drop(); return; }
    try {
      const n = await deleteCalendarEvent(p.uid ?? "", p.title, p.start, p.calendar ?? s.settings.calendarName);
      drop();
      toast({ kind: "info", title: n ? `🗑 Removed from Calendar` : "Removed from the list",
              body: n > 1 ? `${n} matching events deleted (duplicates too).` : n ? p.title : "It was already gone from Calendar." });
    } catch (e) {
      toast({ kind: "error", title: "Couldn't delete it in Calendar", body: `${String(e)} It's still in the list; try again.` });
    }
  };
  const exportIcs = async () => {
    const ev = [...s.plans.filter((p) => p.start > Date.now()).map((p) => ({ title: `🦦 ${p.title}`, start: p.start, minutes: p.minutes })),
                ...s.sessions.map((x) => ({ title: `🦦 Studied ${x.minutes} min`, start: x.start, minutes: x.minutes, desc: x.topics.join("\n") }))];
    const text = ics(ev);
    if (inTauri()) {
      try { const p = await call<string>("save_ics", { contents: text, name: "study-quest" }); toast({ kind: "info", title: "Saved .ics", body: p }); }
      catch (e) { toast({ kind: "error", title: "Export failed", body: String(e) }); }
    } else {
      const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type: "text/calendar" })); a.download = "study-quest.ics"; a.click();
    }
  };

  const upcoming = s.plans.filter((p) => p.start > Date.now() - 36e5);
  return (
    <div className="page">
      <h1>📅 Plan</h1>
      {isMac
        ? <p className="muted">Study blocks and finished sessions go into a <b>{s.settings.calendarName}</b> calendar in macOS Calendar,
            with a 10-minute alert. The first time, macOS asks you to allow Study Quest to control Calendar.</p>
        : <p className="muted">Blocks are saved here. Use <b>Export .ics</b> to open them in your calendar app (Outlook, GNOME Calendar,
            Thunderbird…). Removing a block here doesn't remove it from that app.</p>}
      <section className="card">
        <h2>Plan a study block</h2>
        <div className="form">
          <input placeholder={`Study: ${nextTopics(1)[0]?.title ?? ""}`} value={title} onChange={(e) => setTitle(e.target.value)} />
          <input type="datetime-local" value={when} onChange={(e) => setWhen(e.target.value)} />
          <select value={mins} onChange={(e) => setMins(Number(e.target.value))}>
            {[15, 20, 30, 45, 60, 90].map((m) => <option key={m} value={m}>{m} min</option>)}
          </select>
          <button className="primary" disabled={busy} onClick={addOne}>{isMac ? "Add to Calendar" : "Add block"}</button>
        </div>
        <div className="row" style={{ marginTop: 10 }}>
          <select value={planPath} onChange={(e) => setPlanPath(e.target.value)} aria-label="Path to plan">
            {PATHS.map((p) => <option key={p.id} value={p.id}>{p.icon} {p.title}</option>)}
            <option value="mix">🔀 Mix all paths</option>
          </select>
          <button disabled={busy} onClick={planWeek}>{busy ? "Planning…" : "✨ Plan my week (5 weekday evenings)"}</button>
          <button onClick={exportIcs}>Export .ics</button>
        </div>
        {isMac && <label className="check"><input type="checkbox" checked={s.settings.calendarAuto}
          onChange={(e) => setState((x) => ({ ...x, settings: { ...x.settings, calendarAuto: e.target.checked } }))} />
          Log every finished study session to Calendar automatically (with the topics you covered)</label>}
      </section>
      <section className="card">
        <h2>Upcoming</h2>
        {!upcoming.length && <p className="muted">Nothing planned. Byte is mildly concerned.</p>}
        {upcoming.map((p) => (
          <div key={p.id} className="plan-row">
            <span>{new Date(p.start).toLocaleString([], { weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })}</span>
            <span className="grow">{p.title}</span><small className="muted">{p.minutes} min · {p.status}</small>
            <button className="link" title="Remove from the list and from Calendar" onClick={() => removePlan(p)}>✕</button>
          </div>
        ))}
      </section>
      <section className="card">
        <h2>Recent sessions</h2>
        {!s.sessions.length && <p className="muted">Sessions of 5+ minutes appear here when you take a break.</p>}
        {s.sessions.slice(-8).reverse().map((x) => (
          <div key={x.start} className="plan-row"><span>{new Date(x.start).toLocaleString()}</span>
            <span className="grow">{x.topics.slice(0, 3).join(", ") || "exercises / notebooks"}</span><small>{x.minutes} min</small></div>
        ))}
      </section>
    </div>
  );
}

export function SettingsView() {
  const s = useStore();
  const set = <K extends keyof typeof s.settings>(k: K, v: (typeof s.settings)[K]) =>
    setState((x) => ({ ...x, settings: { ...x.settings, [k]: v } }));
  const exportJson = async () => {
    const text = JSON.stringify(getState(), null, 1);
    if (!inTauri()) {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([text], { type: "application/json" }));
      a.download = "study-quest-progress.json"; a.click();
      return;
    }
    try {
      const p = await call<string>("save_download", { name: `study-quest-progress-${dayKey()}.json`, contents: text });
      toast({ kind: "info", title: "💾 Progress exported", body: p });
    } catch (e) { toast({ kind: "error", title: "Export failed", body: String(e) }); }
  };
  const importJson = async (f: File) => {
    try {
      const data = JSON.parse(await f.text());
      if (typeof data !== "object" || data === null || typeof data.xp !== "number") throw new Error("not a Study Quest progress file");
      if (!(await confirmDialog("Import this progress file?", "It replaces your current XP, streak and notes.", { ok: "Import" }))) return;
      replaceState(data);
      toast({ kind: "info", title: "Progress imported", body: `${data.xp} XP, ${Object.keys(data.done ?? {}).length} topics done` });
    } catch (e) { toast({ kind: "error", title: "Import failed", body: String(e) }); }
  };
  const reset = async () => {
    if (!(await confirmDialog("Reset everything?", "This deletes all XP, streaks, notes and exercise progress. Export first if you might want it back.",
                              { ok: "Delete my progress", danger: true }))) return;
    resetState();
    toast({ kind: "info", title: "Fresh start", body: "Progress cleared. Byte is ready when you are." });
  };
  const hours = Array.from({ length: 24 }, (_, h) => h);
  return (
    <div className="page settings">
      <h1>⚙️ Settings</h1>
      <section className="card">
        <h2>🎨 Appearance</h2>
        <label>Mode
          <div className="seg">
            {(["system", "light", "dark"] as const).map((m) => (
              <button key={m} className={s.settings.themeMode === m ? "on" : ""} onClick={() => set("themeMode", m)}>
                {m === "system" ? "🖥 System" : m === "light" ? "☀️ Light" : "🌙 Dark"}
              </button>
            ))}
          </div></label>
        <label>Accent colour
          <div className="swatches">
            {ACCENTS.map((a) => (
              <button key={a.id} title={a.name} aria-label={a.name} aria-pressed={s.settings.accent === a.id}
                className={`swatch ${s.settings.accent === a.id ? "on" : ""}`} style={{ background: a.color }}
                onClick={() => set("accent", a.id)} />
            ))}
            <input type="color" aria-label="Custom accent" value={ACCENTS.find((a) => a.id === s.settings.accent)?.color ?? s.settings.accent}
              onChange={(e) => set("accent", e.target.value)} />
          </div></label>
      </section>
      <section className="card">
        <h2>You</h2>
        <label>Name (for nudges) <input value={s.settings.name} onChange={(e) => set("name", e.target.value)} /></label>
        <label>Daily goal <select value={s.settings.goalMinutes} onChange={(e) => set("goalMinutes", Number(e.target.value))}>
          {[10, 15, 20, 30, 45, 60, 90].map((m) => <option key={m} value={m}>{m} minutes</option>)}</select></label>
      </section>
      <section className="card">
        <h2>🦦 Nudges</h2>
        <label className="check"><input type="checkbox" checked={s.settings.nudges} onChange={(e) => set("nudges", e.target.checked)} />
          Desktop notifications when I haven't studied (they keep working with the window closed; the app lives in the {TRAY})</label>
        <label>Remind me every <select value={s.settings.intervalMin} onChange={(e) => set("intervalMin", Number(e.target.value))}>
          {[30, 60, 90, 120, 180, 240].map((m) => <option key={m} value={m}>{m} minutes</option>)}</select></label>
        <label>Quiet hours <select value={s.settings.quietStart} onChange={(e) => set("quietStart", Number(e.target.value))}>
          {hours.map((h) => <option key={h} value={h}>{h}:00</option>)}</select> to <select value={s.settings.quietEnd}
          onChange={(e) => set("quietEnd", Number(e.target.value))}>{hours.map((h) => <option key={h} value={h}>{h}:00</option>)}</select></label>
        <label>Troll level
          <div className="seg">
            {["😇 Nice", "😏 Cheeky", "😈 Savage", "🔥 Unhinged"].map((l, i) => (
              <button key={l} className={s.settings.trollLevel === i ? "on" : ""} onClick={() => set("trollLevel", i)}>{l}</button>
            ))}
          </div></label>
        <label className="check"><input type="checkbox" checked={s.settings.voice} onChange={(e) => set("voice", e.target.checked)} />
          Byte reads nudges out loud (system voice)</label>
        <div className="row">
          <button onClick={() => call("nudge_now").catch((e) => toast({ kind: "error", title: "Nudge", body: String(e) }))}>Send a test nudge</button>
          <button onClick={() => call("say", { text: "Hi! I'm Byte. Go study, please." }).catch(() => {})}>Test voice</button>
        </div>
      </section>
      {isMac && (
      <section className="card">
        <h2>📅 Calendar</h2>
        <label>Calendar name <input value={s.settings.calendarName} onChange={(e) => set("calendarName", e.target.value || "Study Quest")} /></label>
        <label className="check"><input type="checkbox" checked={s.settings.calendarAuto} onChange={(e) => set("calendarAuto", e.target.checked)} />
          Log finished study sessions to Calendar</label>
      </section>
      )}
      <section className="card">
        <h2>🐍 Python</h2>
        <PythonSettings python={s.settings.python} setPython={(v) => set("python", v)} />
      </section>
      <section className="card">
        <h2>Progress</h2>
        <div className="row">
          <button onClick={exportJson}>Export progress</button>
          <label className="button">Import<input type="file" accept="application/json,.json" hidden onChange={(e) => {
            const f = e.target.files?.[0];
            e.target.value = "";
            if (f) importJson(f);
          }} /></label>
          <button className="danger" onClick={reset}>Reset everything</button>
        </div>
      </section>
    </div>
  );
}

type Hit = { label: string; sub: string; route: Route };
export function Search({ data, onClose }: { data: Data; onClose: () => void }) {
  const [q, setQ] = useState("");
  const [sel, setSel] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => input.current?.focus(), []);
  const hits = useMemo<Hit[]>(() => {
    const t = q.trim().toLowerCase();
    if (!t) return [];
    const words = t.split(/\s+/);
    const score = (text: string, title: string) => words.every((w) => text.includes(w)) ? (words.every((w) => title.includes(w)) ? 2 : 1) : 0;
    const out: (Hit & { sc: number })[] = [];
    for (const id of data.order) {
      const tp = data.topics.get(id)!, tl = tp.title.toLowerCase();
      const sc = score(`${tl} ${tp.heads.join(" ").toLowerCase()} ${tp.search}`, tl);
      if (sc) out.push({ sc: sc + 0.5, label: tp.title, sub: `${data.books.find((b) => b.id === tp.book)?.icon} ${tp.part}`, route: { v: "topic", id } });
    }
    for (const e of data.exercises) {
      const sc = score(`${e.title} ${e.track} ${e.prompt}`.toLowerCase(), e.title.toLowerCase());
      if (sc) out.push({ sc, label: `🧪 ${e.title}`, sub: `Exercise · ${e.track}`, route: { v: "play", id: e.id } });
    }
    for (const n of data.notebooks) {
      const sc = score(`${n.title} ${n.desc}`.toLowerCase(), n.title.toLowerCase());
      if (sc) out.push({ sc, label: `${n.emoji} ${n.title}`, sub: "Notebook", route: { v: "notebooks", file: n.file } });
    }
    return out.sort((a, b) => b.sc - a.sc).slice(0, 12);
  }, [q, data]);
  const pick = (h: Hit) => { go(h.route); onClose(); };
  return (
    <div className="overlay" onClick={onClose}>
      <div className="palette" onClick={(e) => e.stopPropagation()}>
        <input ref={input} placeholder="Search topics, exercises, notebooks…" value={q}
          onChange={(e) => { setQ(e.target.value); setSel(0); }}
          onKeyDown={(e) => {
            if (e.key === "Escape") onClose();
            if (e.key === "ArrowDown") { e.preventDefault(); setSel(Math.min(hits.length - 1, sel + 1)); }
            if (e.key === "ArrowUp") { e.preventDefault(); setSel(Math.max(0, sel - 1)); }
            if (e.key === "Enter" && hits[sel]) pick(hits[sel]);
          }} />
        <div className="hits">
          {hits.map((h, i) => (
            <button key={i} className={i === sel ? "hit on" : "hit"} onMouseEnter={() => setSel(i)} onClick={() => pick(h)}>
              <span>{h.label}</span><small className="muted">{h.sub}</small>
            </button>
          ))}
          {q && !hits.length && <p className="muted pad">No matches. Byte looked everywhere. Even under the couch.</p>}
        </div>
      </div>
    </div>
  );
}
