// Home: greeting, streak, today's quest, progress per book, a 12-week heatmap and badges.
import type { Data } from "../lib/data";
import { PATHS, pathTopicIds } from "../lib/data";
import { BADGES, dayKey, level, liveStreak, setPath, useStore } from "../lib/store";
import { Icon, Ring } from "../components/Icon";
import { Activity } from "../components/Activity";
import { go } from "../App";
import type { CSSProperties } from "react";

const greet = () => {
  const h = new Date().getHours();
  return h < 5 ? "Burning the midnight oil" : h < 12 ? "Good morning" : h < 18 ? "Good afternoon" : "Good evening";
};
const QUIPS = [
  "Ten minutes beats zero minutes. Every time.",
  "Your future self is a very demanding reward model.",
  "Today's discount factor: high. Invest in later-you.",
  "Small gradient steps, every day. That's the whole trick.",
  "The otter believes in you. The otter has seen things.",
];

export function Dashboard({ data }: { data: Data }) {
  const s = useStore();
  const streak = liveStreak(s), lv = level(s.xp);
  const today = s.days[dayKey()] ?? 0, goal = s.settings.goalMinutes;
  const solved = data.exercises.filter((e) => s.ex[e.id]?.passed).length;
  const startOfDay = new Date().setHours(0, 0, 0, 0);
  const explainedToday = Object.entries(s.awarded ?? {}).some(([k, t]) => k.startsWith("explain:") && t >= startOfDay);
  const pathIds = pathTopicIds(data, s.settings.path || "rl");
  const last = s.lastByPath[s.settings.path || "rl"];
  const next = data.topics.get(last && !s.done[last] ? last : pathIds.find((id) => !s.done[id]) ?? pathIds[0])!;
  const quest = [
    { done: s.today.topics >= 1, label: "Finish one topic", go: () => go({ v: "topic", id: next.id }) },
    { done: explainedToday, label: "Explain one topic back in your own words", go: () => go({ v: "topic", id: next.id }) },
    { done: s.today.exercises >= 1, label: "Pass one coding exercise", go: () => go({ v: "play" }) },
  ];
  const randomTopic = () => {
    const open = pathIds.filter((id) => !s.done[id]);
    go({ v: "topic", id: open[Math.floor(Math.random() * open.length)] ?? data.order[0] });
  };
  const name = s.settings.name ? `, ${s.settings.name}` : "";

  return (
    <div className="page dash">
      <header className="hero">
        <div className="mascot" aria-hidden>🦦</div>
        <div>
          <h1>{greet()}{name}</h1>
          <p className="muted">{QUIPS[new Date().getDate() % QUIPS.length]}</p>
        </div>
      </header>

      <section className="stats">
        <div className="stat"><span className="stat-emoji">{streak ? "🔥" : "🪵"}</span><b>{streak}</b><small>day streak</small></div>
        <div className="stat">
          <span className="stat-emoji">⭐</span><b>Lv {lv.lv}</b><small>{lv.title}</small>
          <div className="meter"><i style={{ width: `${lv.frac * 100}%` }} /></div><small>{lv.toNext} XP to next</small>
        </div>
        <div className="stat">
          <span className="stat-emoji">⏱️</span><b>{today}/{goal}</b><small>minutes today</small>
          <div className="meter"><i style={{ width: `${Math.min(100, (today / goal) * 100)}%` }} /></div>
        </div>
        <button className="stat clickable" onClick={() => go({ v: "play" })}>
          <span className="stat-emoji">🧪</span><b>{solved}/{data.exercises.length}</b><small>exercises solved</small>
        </button>
      </section>

      <section className="grid2">
        <div className="card">
          <h2>Today's quest</h2>
          {quest.map((q) => (
            <button key={q.label} className={`quest ${q.done ? "done" : ""}`} onClick={q.go}>
              <span>{q.done ? "✅" : "⬜"}</span>{q.label}
            </button>
          ))}
          <div className="row" style={{ marginTop: 12 }}>
            <button className="primary" onClick={() => go({ v: "topic", id: next.id })}>
              {last ? "Continue" : "Start"}: {next.title.slice(0, 42)}
            </button>
            <button onClick={randomTopic} title="Pick a random unfinished topic">🎲 I'm bored</button>
          </div>
        </div>
        <div className="card">
          <h2>Learning paths</h2>
          <p className="muted small">Pick one to focus on. The sidebar shows only that path.</p>
          <div className="path-tiles">
            {PATHS.map((p) => {
              const ids = pathTopicIds(data, p.id), n = ids.filter((id) => s.done[id]).length;
              const on = p.id === (s.settings.path || "rl");
              return (
                <button key={p.id} className={`path-tile ${on ? "on" : ""}`} style={{ "--path": "var(--accent)" } as CSSProperties}
                  onClick={() => { setPath(p.id); go({ v: "topic", id: s.lastByPath[p.id] ?? ids.find((id) => !s.done[id]) ?? ids[0] }); }}>
                  <span className="path-tile-icon"><Icon name={p.ic} size={18} /></span>
                  <span className="path-tile-text"><b>{p.long}</b><small className="muted">{n} of {ids.length} topics{on ? " · current" : ""}</small></span>
                  <Ring value={n / Math.max(1, ids.length)} size={34} stroke={4} color="var(--accent)"
                    label={<small>{Math.round((n / Math.max(1, ids.length)) * 100)}</small>} />
                </button>
              );
            })}
          </div>
        </div>
      </section>

      {Object.entries(s.muddy).some(([, m]) => m.text && !m.resolved) && (
        <section className="card">
          <h2>🌫️ Still unclear <small className="muted">your muddiest points</small></h2>
          {Object.entries(s.muddy).filter(([, m]) => m.text && !m.resolved).slice(0, 5).map(([tid, m]) => (
            <button key={tid} className="quest" onClick={() => go({ v: "topic", id: tid })}>
              <span>❓</span><span className="grow">{m.text}</span><small className="muted">{data.topics.get(tid)?.title.slice(0, 30)}</small>
            </button>
          ))}
        </section>
      )}

      <Activity s={s} streak={streak} />

      <section className="card">
        <h2>Badges <small className="muted">{s.badges.length}/{BADGES.length}</small></h2>
        <div className="badges">
          {BADGES.map((b) => (
            <div key={b.id} className={`badge ${s.badges.includes(b.id) ? "got" : ""}`} title={b.desc}>
              <span>{s.badges.includes(b.id) ? b.emoji : "🔒"}</span><small>{b.name}</small>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
