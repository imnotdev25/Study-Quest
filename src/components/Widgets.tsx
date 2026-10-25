// Small in-page toys shown inside matching lessons ("Try it"). No backend needed.
import { useMemo, useState, type ReactNode } from "react";
import { addXP } from "../lib/store";

function Plot({ series, xs, yMax, yMin = 0, band }: { series: { ys: number[]; color: string; dash?: boolean }[];
  xs: number[]; yMax: number; yMin?: number; band?: [number, number] }) {
  const W = 360, H = 160, x0 = xs[0], x1 = xs[xs.length - 1];
  const sx = (x: number) => ((x - x0) / (x1 - x0)) * W;
  const sy = (y: number) => H - ((y - yMin) / (yMax - yMin || 1)) * H;
  return (
    <svg viewBox={`-4 -4 ${W + 8} ${H + 8}`} className="plot" role="img">
      {band && <rect x={sx(band[0])} width={sx(band[1]) - sx(band[0])} y={0} height={H} className="plot-band" />}
      <line x1={0} x2={W} y1={sy(0)} y2={sy(0)} className="plot-axis" />
      {series.map((s, i) => (
        <polyline key={i} fill="none" style={{ stroke: s.color }} strokeWidth={2.4} strokeDasharray={s.dash ? "5 4" : undefined}
          points={s.ys.map((y, j) => `${sx(xs[j])},${sy(Math.max(yMin, Math.min(yMax, y)))}`).join(" ")} />
      ))}
    </svg>
  );
}

const Slider = ({ label, min, max, step, value, set }: { label: string; min: number; max: number; step: number;
  value: number; set: (v: number) => void }) => (
  <label className="slider">
    <span>{label} <b>{Number.isInteger(step) ? value : value.toFixed(2)}</b></span>
    <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => set(Number(e.target.value))} />
  </label>
);

function Softmax() {
  const [z, setZ] = useState([2, 1, 0.1, -1]);
  const [T, setT] = useState(1);
  const e = z.map((v) => Math.exp((v - Math.max(...z)) / T));
  const p = e.map((v) => v / e.reduce((a, b) => a + b, 0));
  const H = -p.reduce((a, v) => a + (v > 0 ? v * Math.log(v) : 0), 0);
  return (
    <div>
      <div className="bars">
        {p.map((v, i) => (
          <div key={i} className="bar-col">
            <div className="bar" style={{ height: `${v * 100}%` }}><span>{(v * 100).toFixed(0)}%</span></div>
            <input type="range" min={-4} max={4} step={0.1} value={z[i]} aria-label={`logit ${i}`}
              onChange={(ev) => setZ(z.map((x, j) => (j === i ? Number(ev.target.value) : x)))} />
            <small>z={z[i].toFixed(1)}</small>
          </div>
        ))}
      </div>
      <Slider label="temperature" min={0.05} max={5} step={0.05} value={T} set={setT} />
      <p className="muted">Entropy {H.toFixed(2)} nats (max {Math.log(4).toFixed(2)}). T→0 is argmax, T→∞ is uniform.</p>
    </div>
  );
}

function Bandit() {
  const means = useMemo(() => { const m = [0.2 + Math.random() * 0.3, 0.3 + Math.random() * 0.4, 0.25 + Math.random() * 0.3];
    return m; }, []);
  const [pulls, setPulls] = useState<number[][]>([[], [], []]);
  const n = pulls.flat().length, best = Math.max(...means);
  const regret = pulls.reduce((a, r, i) => a + r.length * (best - means[i]), 0);
  const pull = (i: number) => {
    const r = Math.random() < means[i] ? 1 : 0;
    setPulls(pulls.map((x, j) => (j === i ? [...x, r] : x)));
    if (n + 1 === 20) addXP(3, "Bandit Casino: 20 pulls");
  };
  return (
    <div>
      <div className="row">
        {["🍒", "🍋", "🔔"].map((em, i) => (
          <button key={i} className="slot" onClick={() => pull(i)}>
            <span>{em}</span>
            <small>{pulls[i].length} pulls · {pulls[i].length ? (pulls[i].reduce((a, b) => a + b, 0) / pulls[i].length).toFixed(2) : "?"}</small>
          </button>
        ))}
      </div>
      <p className="muted">{n} pulls, regret {regret.toFixed(2)}.
        {n >= 20 ? ` Truth: ${means.map((m) => m.toFixed(2)).join(" / ")}. Best was ${["🍒", "🍋", "🔔"][means.indexOf(best)]}.`
                 : " Find the best machine. After 20 pulls I reveal the truth."}</p>
    </div>
  );
}

function PPOClip() {
  const [eps, setEps] = useState(0.2);
  const [A, setA] = useState(1);
  const xs = Array.from({ length: 120 }, (_, i) => 0.3 + (i / 119) * 1.7);
  const obj = xs.map((r) => Math.min(r * A, Math.min(Math.max(r, 1 - eps), 1 + eps) * A));
  return (
    <div>
      <Plot xs={xs} yMin={-2.2} yMax={2.2} band={[1 - eps, 1 + eps]}
        series={[{ ys: xs.map((r) => r * A), color: "#9ca3af", dash: true }, { ys: obj, color: "var(--accent)" }]} />
      <Slider label="clip ε" min={0.05} max={0.5} step={0.01} value={eps} set={setEps} />
      <div className="row"><button className={A > 0 ? "chip on" : "chip"} onClick={() => setA(1)}>A &gt; 0</button>
        <button className={A < 0 ? "chip on" : "chip"} onClick={() => setA(-1)}>A &lt; 0</button></div>
      <p className="muted">Purple: PPO's objective vs ratio r. Where it goes flat, the gradient is zero.</p>
    </div>
  );
}

function KVCalc() {
  const [L, setL] = useState(32), [kv, setKv] = useState(8), [hd, setHd] = useState(128), [ctx, setCtx] = useState(8192);
  const [users, setUsers] = useState(16);
  const per = 2 * L * kv * hd * 2;
  const total = (per * ctx * users) / 2 ** 30;
  return (
    <div>
      <Slider label="layers" min={8} max={128} step={1} value={L} set={setL} />
      <Slider label="KV heads" min={1} max={64} step={1} value={kv} set={setKv} />
      <Slider label="head dim" min={32} max={256} step={32} value={hd} set={setHd} />
      <Slider label="context" min={1024} max={131072} step={1024} value={ctx} set={setCtx} />
      <Slider label="concurrent users" min={1} max={256} step={1} value={users} set={setUsers} />
      <p className="big">{(per / 1024).toFixed(0)} KiB/token · <b>{total.toFixed(1)} GiB</b> of KV cache</p>
      <p className="muted">{total > 60 ? "That won't fit next to the weights on one 80 GB GPU. Try fewer KV heads (GQA) or FP8." : "Fits on one 80 GB GPU next to an 8B model."}</p>
    </div>
  );
}

function SpecDecode() {
  const [a, setA] = useState(0.8), [c, setC] = useState(0.05);
  const ks = Array.from({ length: 20 }, (_, i) => i + 1);
  const tok = ks.map((k) => (1 - a ** (k + 1)) / (1 - a));
  const sp = tok.map((t, i) => t / (1 + c * ks[i]));
  const best = sp.indexOf(Math.max(...sp));
  return (
    <div>
      <Plot xs={ks} yMax={Math.max(...tok) * 1.05} series={[{ ys: tok, color: "#9ca3af", dash: true }, { ys: sp, color: "var(--accent)" }]} />
      <Slider label="acceptance a" min={0.3} max={0.97} step={0.01} value={a} set={setA} />
      <Slider label="draft cost c" min={0} max={0.3} step={0.01} value={c} set={setC} />
      <p className="big">Best k = {ks[best]} → <b>{sp[best].toFixed(2)}× faster</b></p>
    </div>
  );
}

function lgamma(x: number): number {
  const g = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let y = x, t = x + 5.5, s = 1.000000000190015;
  t -= (x + 0.5) * Math.log(t);
  for (const c of g) s += c / ++y;
  return -t + Math.log((2.5066282746310005 * s) / x);
}

function Bayes() {
  const [h, setH] = useState(7), [t, setT] = useState(3), [a, setA] = useState(2);
  const xs = Array.from({ length: 150 }, (_, i) => 0.003 + (i / 149) * 0.994);
  const beta = (p: number, q: number) => xs.map((x) => Math.exp((p - 1) * Math.log(x) + (q - 1) * Math.log(1 - x) - (lgamma(p) + lgamma(q) - lgamma(p + q))));
  const post = beta(a + h, a + t), prior = beta(a, a);
  return (
    <div>
      <Plot xs={xs} yMax={Math.max(...post, ...prior) * 1.05} series={[{ ys: prior, color: "#9ca3af", dash: true }, { ys: post, color: "var(--accent)" }]} />
      <Slider label="heads" min={0} max={60} step={1} value={h} set={setH} />
      <Slider label="tails" min={0} max={60} step={1} value={t} set={setT} />
      <Slider label="prior strength" min={1} max={20} step={1} value={a} set={setA} />
      <p className="muted">Posterior mean {((a + h) / (2 * a + h + t)).toFixed(3)} vs MLE {(h / Math.max(1, h + t)).toFixed(3)}.</p>
    </div>
  );
}

export const WIDGETS: { id: string; emoji: string; title: string; keys: string[]; el: () => ReactNode }[] = [
  { id: "softmax", emoji: "🌡️", title: "Softmax & temperature", keys: ["softmax", "temperature", "decoding", "sampling"], el: () => <Softmax /> },
  { id: "bandit", emoji: "🎰", title: "Bandit casino", keys: ["bandit", "exploration"], el: () => <Bandit /> },
  { id: "ppo", emoji: "✂️", title: "PPO clipping", keys: ["ppo", "trust region", "clipped", "grpo"], el: () => <PPOClip /> },
  { id: "kv", emoji: "🧮", title: "KV cache calculator", keys: ["kv cache", "paged", "inference", "attention variants", "gqa"], el: () => <KVCalc /> },
  { id: "spec", emoji: "🏎️", title: "Speculative decoding", keys: ["speculative"], el: () => <SpecDecode /> },
  { id: "bayes", emoji: "🪙", title: "Bayesian coin", keys: ["bayes", "probability rules", "likelihood", "beta"], el: () => <Bayes /> },
];

export const widgetsFor = (title: string) => {
  const t = title.toLowerCase();
  return WIDGETS.filter((w) => w.keys.some((k) => t.includes(k))).slice(0, 2);
};
