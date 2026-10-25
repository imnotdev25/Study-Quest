// App state: progress, streaks, spaced repetition and settings, persisted in localStorage.
import { useSyncExternalStore } from "react";

export type Settings = {
  name: string; goalMinutes: number; nudges: boolean; intervalMin: number; quietStart: number; quietEnd: number;
  trollLevel: number; voice: boolean; calendarAuto: boolean; calendarName: string; python: string; path: string;
  themeMode: "system" | "light" | "dark"; accent: string;
};
export type ExState = { code: string; passed: boolean; attempts: number; fails: number; peeked?: boolean };
export type Session = { start: number; minutes: number; topics: string[] };
export type Plan = { id: string; title: string; start: number; minutes: number; status: string; topic?: string; uid?: string;
  calendar?: string };
export type Today = { day: string; topics: number; exercises: number; goalToasted: boolean };
export type State = {
  v: 1; xp: number; streak: number; lastDay: string; days: Record<string, number>; done: Record<string, number>;
  visited: Record<string, number>; lastByPath: Record<string, string>; notes: Record<string, string>;
  ex: Record<string, ExState>; badges: string[]; lastStudyMs: number; lastTopic: string; firstRun: boolean;
  sessions: Session[]; plans: Plan[]; today: Today; settings: Settings;
  // active learning, all keyed by topic id
  guesses: Record<string, string>; muddy: Record<string, { text: string; at: number; resolved?: boolean }>;
  links: Record<string, string>; conf: Record<string, { before?: number; after?: number }>;
  awarded: Record<string, number>;
};

/** Give XP once per key (e.g. "guess:rl:3"), so editing a note doesn't farm points. */
export function awardOnce(key: string, n: number, why: string) {
  if (getState().awarded?.[key]) return;
  addXP(n, why, (x) => ({ ...x, awarded: { ...x.awarded, [key]: Date.now() } }));
}

const KEY = "studyquest.v1";
export const dayKey = (d = new Date()) =>
  `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;

const fresh = (): State => ({
  v: 1, xp: 0, streak: 0, lastDay: "", days: {}, done: {}, visited: {}, lastByPath: {}, notes: {}, ex: {}, badges: [],
  lastStudyMs: 0, lastTopic: "", firstRun: true, sessions: [], plans: [],
  guesses: {}, muddy: {}, links: {}, conf: {}, awarded: {},
  today: { day: dayKey(), topics: 0, exercises: 0, goalToasted: false },
  settings: { name: "", goalMinutes: 20, nudges: true, intervalMin: 90, quietStart: 23, quietEnd: 8, trollLevel: 1,
              voice: false, calendarAuto: false, calendarName: "Study Quest", python: "", path: "rl",
              themeMode: "system", accent: "violet" },
});

export const setPath = (path: string) => setState((x) => ({ ...x, settings: { ...x.settings, path } }));

function load(): State {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) {
      const s = JSON.parse(raw);
      const f = fresh();
      return { ...f, ...s, settings: { ...f.settings, ...s.settings }, today: { ...f.today, ...s.today } };
    }
  } catch { /* corrupted storage: start fresh */ }
  return fresh();
}

let state = load();
const subs = new Set<() => void>();
export const getState = () => state;
export function setState(fn: (s: State) => State) {
  state = rollDay(fn(state));
  localStorage.setItem(KEY, JSON.stringify(state));
  subs.forEach((f) => f());
}
export const useStore = () => useSyncExternalStore((f) => (subs.add(f), () => subs.delete(f)), () => state);
export const replaceState = (s: State) => setState(() => ({ ...fresh(), ...s }));
export const resetState = () => setState(() => fresh());

function rollDay(s: State): State {
  const d = dayKey();
  return s.today.day === d ? s : { ...s, today: { day: d, topics: 0, exercises: 0, goalToasted: false } };
}

/** Effective streak: still alive if the last active day was today or yesterday. */
export function liveStreak(s: State) {
  const y = dayKey(new Date(Date.now() - 864e5));
  return s.lastDay === dayKey() || s.lastDay === y ? s.streak : 0;
}

function active(s: State): State {
  const today = dayKey();
  if (s.lastDay === today) return { ...s, lastStudyMs: Date.now() };
  const y = dayKey(new Date(Date.now() - 864e5));
  return { ...s, streak: s.lastDay === y ? s.streak + 1 : 1, lastDay: today, lastStudyMs: Date.now() };
}

// ---------------------------------------------------------------- toasts
export type Toast = { id: number; title: string; body?: string; kind: "xp" | "nudge" | "badge" | "praise" | "info" | "error" };
const toastSubs = new Set<(t: Toast) => void>();
let tid = 0;
export const onToast = (f: (t: Toast) => void) => (toastSubs.add(f), () => { toastSubs.delete(f); });
export const toast = (t: Omit<Toast, "id">) => toastSubs.forEach((f) => f({ ...t, id: ++tid }));

// ---------------------------------------------------------------- XP, levels, badges
export function addXP(n: number, why: string, extra?: (s: State) => State) {
  setState((s) => checkBadges((extra ?? ((x) => x))(active({ ...s, xp: s.xp + n }))));
  toast({ kind: "xp", title: `+${n} XP`, body: why });
}

export function addMinute() {
  setState((s) => {
    const d = dayKey();
    let next = active({ ...s, firstRun: false, xp: s.xp + 1, days: { ...s.days, [d]: (s.days[d] ?? 0) + 1 } });
    if (!next.today.goalToasted && next.days[d] >= next.settings.goalMinutes) {
      next = { ...next, xp: next.xp + 25, today: { ...next.today, goalToasted: true } };
      setTimeout(() => toast({ kind: "praise", title: "🎉 Daily goal crushed! +25 XP", body: "Byte is doing a happy otter spin." }));
    }
    return checkBadges(next);
  });
}

export function level(xp: number) {
  const lv = Math.floor(Math.sqrt(xp / 50)) + 1;
  const lo = 50 * (lv - 1) ** 2, hi = 50 * lv ** 2;
  const titles = ["Random Policy", "Curious Agent", "Bandit Tamer", "Q-Learner", "Policy Gradienter", "Actor-Critic",
    "PPO Pilot", "Transformer Whisperer", "RLHF Wrangler", "GRPO Grandmaster", "Inference Engineer", "Frontier Researcher"];
  return { lv, frac: (xp - lo) / (hi - lo), toNext: hi - xp, title: titles[Math.min(lv - 1, titles.length - 1)] };
}

export const BADGES: { id: string; emoji: string; name: string; desc: string; test: (s: State) => boolean }[] = [
  { id: "first", emoji: "🐣", name: "Hello, World", desc: "Finish your first topic", test: (s) => Object.keys(s.done).length >= 1 },
  { id: "ten", emoji: "📚", name: "Bookworm", desc: "Finish 10 topics", test: (s) => Object.keys(s.done).length >= 10 },
  { id: "fifty", emoji: "🦉", name: "Night Owl Scholar", desc: "Finish 50 topics", test: (s) => Object.keys(s.done).length >= 50 },
  { id: "streak3", emoji: "🔥", name: "On Fire", desc: "3-day streak", test: (s) => s.streak >= 3 },
  { id: "streak7", emoji: "🌋", name: "Volcano", desc: "7-day streak", test: (s) => s.streak >= 7 },
  { id: "streak30", emoji: "☄️", name: "Comet", desc: "30-day streak", test: (s) => s.streak >= 30 },
  { id: "connect5", emoji: "🔗", name: "Connector", desc: "Connect 5 topics to earlier ones", test: (s) => Object.values(s.links ?? {}).filter((l) => l.length >= 50).length >= 5 },
  { id: "ex1", emoji: "🧪", name: "Green Tests", desc: "Pass your first exercise", test: (s) => Object.values(s.ex).some((e) => e.passed) },
  { id: "ex10", emoji: "⚙️", name: "Code Monkey", desc: "Pass 10 exercises", test: (s) => Object.values(s.ex).filter((e) => e.passed).length >= 10 },
  { id: "noPeek", emoji: "🙈", name: "No Peeking", desc: "Pass 5 exercises without viewing the solution", test: (s) => Object.values(s.ex).filter((e) => e.passed && !e.peeked).length >= 5 },
  { id: "notes", emoji: "✍️", name: "Feynman Mode", desc: "Explain 5 topics in your own words", test: (s) => Object.values(s.notes).filter((n) => n.length >= 80).length >= 5 },
  { id: "lv5", emoji: "🏅", name: "Level 5", desc: "Reach level 5", test: (s) => level(s.xp).lv >= 5 },
];

function checkBadges(s: State): State {
  const got = BADGES.filter((b) => !s.badges.includes(b.id) && b.test(s));
  if (!got.length) return s;
  got.forEach((b) => setTimeout(() => toast({ kind: "badge", title: `${b.emoji} Badge: ${b.name}`, body: b.desc })));
  return { ...s, badges: [...s.badges, ...got.map((b) => b.id)] };
}
