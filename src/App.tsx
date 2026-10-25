// Shell: hash routing, sidebar, study timer, sync with the Rust side (nudges, tray, Calendar).
import { useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { useData, type Data } from "./lib/data";
import { addMinute, dayKey, getState, liveStreak, setState, toast, useStore, type State } from "./lib/store";
import { addCalendarEvent, call, inTauri } from "./lib/tauri";
import { Toasts } from "./components/Toasts";
import { Sidebar } from "./components/Sidebar";
import { DialogHost } from "./components/Dialog";
import { isMac } from "./lib/platform";
import { useTheme } from "./lib/theme";
import { Dashboard } from "./views/Dashboard";
import { TopicView } from "./views/Topic";
import { Playground } from "./views/Playground";
import { Notebooks } from "./views/Notebooks";
import { PlanView, Search, SettingsView } from "./views/Misc";
import { Diagrams } from "./views/Diagrams";

export type Route =
  | { v: "home" } | { v: "topic"; id: string } | { v: "play"; id?: string }
  | { v: "notebooks"; file?: string } | { v: "draw"; id?: string } | { v: "plan" } | { v: "settings" };

function parse(h: string): Route {
  const [v, ...rest] = h.replace(/^#\/?/, "").split("/");
  const arg = rest.length ? decodeURIComponent(rest.join("/")) : undefined;
  switch (v) {
    case "topic": return arg ? { v, id: arg } : { v: "home" };
    case "play": return { v, id: arg };
    case "draw": return { v, id: arg };
    case "notebooks": return { v, file: arg };
    case "plan": case "settings": return { v };
    default: return { v: "home" };
  }
}
export function go(r: Route) {
  const arg = "id" in r ? r.id : "file" in r ? r.file : undefined;
  location.hash = `#/${r.v}${arg ? "/" + encodeURIComponent(arg) : ""}`;
}

const STUDY_VIEWS = new Set(["topic", "play", "notebooks", "draw"]);

/** Counts a minute when the window is visible, you're on a study view, and you touched something in the last 3 min. */
function useStudyTimer(route: Route, data: Data | null) {
  useEffect(() => {
    let lastInput = Date.now();
    let session: { start: number; minutes: number; topics: Set<string>; last: number } | null = null;
    const touch = () => (lastInput = Date.now());
    const evs = ["mousemove", "keydown", "wheel", "click"] as const;
    evs.forEach((e) => window.addEventListener(e, touch, { passive: true }));
    const closeSession = () => {
      if (!session || session.minutes < 5) { session = null; return; }
      const done = { start: session.start, minutes: session.minutes, topics: [...session.topics] };
      session = null;
      setState((s) => ({ ...s, sessions: [...s.sessions.slice(-199), done] }));
      const st = getState().settings;
      if (st.calendarAuto && isMac && inTauri())
        addCalendarEvent(`Studied ${done.minutes} min`, done.topics.join("\n") || "Exercises and notebooks", done.start, done.minutes, st.calendarName, false)
          .then(() => toast({ kind: "info", title: "📅 Session logged to Calendar", body: `${done.minutes} minutes` }),
                (e) => toast({ kind: "error", title: "Calendar", body: String(e) }));
    };
    const tick = setInterval(() => {
      const r = parse(location.hash);
      const activeNow = document.visibilityState === "visible" && Date.now() - lastInput < 3 * 60000 && STUDY_VIEWS.has(r.v);
      if (activeNow) {
        addMinute();
        session ??= { start: Date.now() - 60000, minutes: 0, topics: new Set(), last: Date.now() };
        session.minutes++; session.last = Date.now();
        if (r.v === "topic" && data) session.topics.add(data.topics.get(r.id)?.title ?? r.id);
      } else if (session && Date.now() - session.last > 10 * 60000) closeSession();
    }, 60000);
    const hide = () => document.visibilityState === "hidden" && closeSession();
    document.addEventListener("visibilitychange", hide);
    return () => { clearInterval(tick); evs.forEach((e) => window.removeEventListener(e, touch)); document.removeEventListener("visibilitychange", hide); };
  }, [data]);
  void route;
}

/** Push a snapshot to Rust: it runs the nudge scheduler and owns the menu-bar title. */
function useSync(s: State, data: Data | null) {
  useEffect(() => {
    if (!inTauri() || !data) return;
    const t = setTimeout(() => {
      const st = s.settings, streak = liveStreak(s);
      call("set_config", { config: { python: st.python || null } }).catch(() => {});
      call("sync_state", { snapshot: {
        enabled: st.nudges, lastStudyMs: s.lastStudyMs, streak, todayMinutes: s.days[dayKey()] ?? 0,
        goalMinutes: st.goalMinutes, intervalMin: st.intervalMin, quietStart: st.quietStart,
        quietEnd: st.quietEnd, trollLevel: st.trollLevel, voice: st.voice, name: st.name,
        tzOffsetMin: new Date().getTimezoneOffset(), firstRun: s.firstRun,
        trayTitle: streak ? `🔥${streak}` : "🦦",
      } }).catch(() => {});
    }, 400);
    return () => clearTimeout(t);
  }, [s, data]);
}

export default function App() {
  const { data, error } = useData();
  const [route, setRoute] = useState<Route>(() => parse(location.hash));
  const [search, setSearch] = useState(false);
  const s = useStore();
  useStudyTimer(route, data);
  useSync(s, data);
  useTheme(s.settings.themeMode, s.settings.accent);

  useEffect(() => {
    const f = () => setRoute(parse(location.hash));
    window.addEventListener("hashchange", f);
    const k = (e: KeyboardEvent) => { if ((e.metaKey || e.ctrlKey) && e.key === "k") { e.preventDefault(); setSearch((x) => !x); } };
    window.addEventListener("keydown", k);
    return () => { window.removeEventListener("hashchange", f); window.removeEventListener("keydown", k); };
  }, []);

  useEffect(() => {
    if (!inTauri()) return;
    const un = listen<{ t: string; b: string }>("nudge", (e) => toast({ kind: "nudge", title: e.payload.t, body: e.payload.b }));
    import("@tauri-apps/plugin-notification").then(async (n) => {
      if (getState().settings.nudges && !(await n.isPermissionGranted())) await n.requestPermission();
    }).catch(() => {});
    return () => { un.then((f) => f()); };
  }, []);

  if (error) return <div className="page"><h1>Couldn't load content</h1><pre>{error}</pre><p>Run <code>pnpm content</code>.</p></div>;
  if (!data) return <div className="boot"><div className="mascot big">🦦</div><p>Waking Byte up…</p></div>;

  return (
    <div className="app">
      <Sidebar data={data} route={route} onSearch={() => setSearch(true)} />
      <main className={`main ${route.v === "play" || (route.v === "notebooks" && route.file) || (route.v === "draw" && route.id) ? "full" : ""}`}>
        <div className="topdrag" data-tauri-drag-region />
        {route.v === "home" && <Dashboard data={data} />}
        {route.v === "topic" && <TopicView key={route.id} data={data} id={route.id} />}
        {route.v === "play" && <Playground key={route.id ?? "first"} data={data} id={route.id} />}
        {route.v === "notebooks" && <Notebooks data={data} file={route.file} />}
        {route.v === "draw" && <Diagrams key={route.id ?? "list"} data={data} id={route.id} />}
        {route.v === "plan" && <PlanView data={data} />}
        {route.v === "settings" && <SettingsView />}
      </main>
      {search && <Search data={data} onClose={() => setSearch(false)} />}
      <Toasts />
      <DialogHost />
    </div>
  );
}
