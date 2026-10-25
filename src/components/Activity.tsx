// Dashboard activity card: this week vs last, a 7-day bar chart against the daily goal, and a calendar heatmap
// with weekday and month labels (weeks run Monday to Sunday, like Calendar).
import { useState, type CSSProperties } from "react";
import { dayKey, type State } from "../lib/store";
import { Icon } from "./Icon";

const DAY = 864e5;
const WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const startOfDay = (d: Date) => { const x = new Date(d); x.setHours(0, 0, 0, 0); return x; };
const addDays = (d: Date, n: number) => { const x = new Date(d); x.setDate(x.getDate() + n); return x; };
const mondayOf = (d: Date) => addDays(startOfDay(d), -((d.getDay() + 6) % 7));
const fmt = (m: number) => (m >= 60 ? `${Math.floor(m / 60)}h ${m % 60 ? `${m % 60}m` : ""}`.trim() : `${m}m`);

function bestStreak(days: Record<string, number>) {
  const on = Object.keys(days).filter((k) => days[k] > 0).sort();
  let best = 0, run = 0, prev = 0;
  for (const k of on) {
    const t = new Date(`${k}T12:00:00`).getTime();
    run = prev && Math.round((t - prev) / DAY) === 1 ? run + 1 : 1;
    best = Math.max(best, run); prev = t;
  }
  return best;
}

export function Activity({ s, streak }: { s: State; streak: number }) {
  const [weeksShown, setWeeksShown] = useState(16);
  const goal = s.settings.goalMinutes;
  const today = startOfDay(new Date());
  const min = (d: Date) => s.days[dayKey(d)] ?? 0;

  const thisMon = mondayOf(today);
  const week = Array.from({ length: 7 }, (_, i) => addDays(thisMon, i));
  const lastWeek = week.map((d) => addDays(d, -7));
  const thisTotal = week.reduce((a, d) => a + min(d), 0), lastTotal = lastWeek.reduce((a, d) => a + min(d), 0);
  const delta = lastTotal ? Math.round(((thisTotal - lastTotal) / lastTotal) * 100) : thisTotal ? 100 : 0;
  const activeDays = week.filter((d) => min(d) > 0).length;
  const goalDays = week.filter((d) => min(d) >= goal).length;
  const peak = Math.max(goal * 1.25, ...week.map(min), 1);

  // Heatmap: columns are weeks (oldest left), rows Mon..Sun.
  const firstMon = addDays(thisMon, -7 * (weeksShown - 1));
  const cols = Array.from({ length: weeksShown }, (_, w) => Array.from({ length: 7 }, (_, d) => addDays(firstMon, w * 7 + d)));
  const level = (m: number) => (m === 0 ? 0 : m < goal * 0.5 ? 1 : m < goal ? 2 : m < goal * 2 ? 3 : 4);
  const total = cols.flat().reduce((a, d) => a + min(d), 0);

  return (
    <section className="card activity">
      <div className="act-head">
        <h2>Your activity</h2>
        <div className="seg small-seg">
          {[8, 16, 26].map((n) => <button key={n} className={weeksShown === n ? "on" : ""} onClick={() => setWeeksShown(n)}>{n === 26 ? "6 mo" : `${n} wk`}</button>)}
        </div>
      </div>

      <div className="act-stats">
        <div className="act-stat">
          <small>This week</small><b>{fmt(thisTotal)}</b>
          <span className={`delta ${delta > 0 ? "up" : delta < 0 ? "down" : ""}`}>
            {lastTotal || thisTotal ? `${delta > 0 ? "▲" : delta < 0 ? "▼" : "•"} ${Math.abs(delta)}% vs last week` : "no data yet"}
          </span>
        </div>
        <div className="act-stat"><small>Active days</small><b>{activeDays}<span className="of">/7</span></b><span className="muted">{goalDays} hit the goal</span></div>
        <div className="act-stat"><small>Current streak</small><b className="flame"><Icon name="flame" size={16} />{streak}</b><span className="muted">best {Math.max(bestStreak(s.days), streak)}</span></div>
        <div className="act-stat"><small>Daily goal</small><b>{fmt(goal)}</b><span className="muted">change in Settings</span></div>
      </div>

      <div className="bars7" style={{ "--goal": `${(goal / peak) * 100}%` } as CSSProperties}>
        {week.map((d, i) => {
          const m = min(d), isToday = d.getTime() === today.getTime(), future = d > today;
          return (
            <div key={i} className={`bar7 ${isToday ? "today" : ""} ${future ? "future" : ""}`} title={`${d.toDateString()}: ${fmt(m)}`}>
              <div className="bar7-track">
                <div className={`bar7-fill ${m >= goal ? "met" : ""}`} style={{ height: `${(m / peak) * 100}%` }}>
                  {m > 0 && <span>{fmt(m)}</span>}
                </div>
              </div>
              <small>{isToday ? "Today" : WD[i]}</small>
            </div>
          );
        })}
      </div>

      <div className="heat2">
        <div className="heat2-months">
          {cols.map((c, w) => (
            // Label a column when its week starts a new month.
            <span key={w}>{w === 0 || c[0].getMonth() !== cols[w - 1][0].getMonth() ? c[0].toLocaleString([], { month: "short" }) : ""}</span>
          ))}
        </div>
        <div className="heat2-body">
          <div className="heat2-wd">{WD.map((d, i) => <span key={d}>{i % 2 === 0 ? d : ""}</span>)}</div>
          <div className="heat2-grid">
            {cols.map((c, w) => (
              <div key={w} className="heat2-col">
                {c.map((d) => {
                  const m = min(d), future = d > today;
                  return <i key={d.getTime()} className={`l${future ? "x" : level(m)} ${d.getTime() === today.getTime() ? "today" : ""}`}
                            title={future ? "" : `${d.toLocaleDateString([], { weekday: "short", month: "short", day: "numeric" })}: ${fmt(m)}`} />;
                })}
              </div>
            ))}
          </div>
        </div>
        <div className="heat2-foot">
          <span className="muted">{fmt(total)} in the last {weeksShown} weeks</span>
          <span className="legend">Less {[0, 1, 2, 3, 4].map((l) => <i key={l} className={`l${l}`} />)} More</span>
        </div>
      </div>
    </section>
  );
}
