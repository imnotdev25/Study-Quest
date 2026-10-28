// Multi-armed bandits: three slot machines with hidden payout rates. Estimates from pulls, greedy getting stuck
// on bad luck, epsilon-greedy exploring, estimates converging, and regret over time.
import { C, THREE, cameraPath, clamp, dynLabel, ease, fade, label, lerp, mat, smooth, story } from "../lib.js";

export const meta = {
  title: "Multi-armed bandits", path: "rl", duration: 52,
  desc: "Three slot machines, hidden payout rates: why pure greed gets stuck, and how a little exploration fixes it.",
  keywords: ["Multi-Armed Bandits", "Bandit", "Exploration", "Upper Confidence"],
  poster: 26,
};

const TRUE = [0.35, 0.7, 0.45];
const NAMES = ["A", "B", "C"];
const COLORS = [0x2563eb, 0x7c3aed, 0x0d9488];
const CH = [
  { t: 0, title: "", text: "" },
  { t: 4.5, title: "The problem", text: "Three machines. Each pays a coin with a hidden probability. Which one is best? We only learn by pulling." },
  { t: 11, title: "Try each once", text: "One pull each. The bar is our estimate: wins ÷ pulls. A won, B and C lost." },
  { t: 18, title: "Greedy gets stuck", text: "Always pulling the best-looking machine means A forever. B lost once by bad luck and is never tried again." },
  { t: 26, title: "Explore a little (ε-greedy)", text: "1 pull in 10 picks a random machine. B gets another chance, wins, and its estimate climbs." },
  { t: 35, title: "Estimates converge", text: "After hundreds of pulls the estimates sit on the true payout rates (outlines). B really is the best." },
  { t: 43, title: "The cost of greed: regret", text: "Regret = coins lost versus always pulling the best machine. Greedy's regret grows forever; ε-greedy's levels off." },
];
// [time, machine, win, explore?]
const PULLS = [
  [12.2, 0, 1], [13.8, 1, 0], [15.4, 2, 0],
  [19, 0, 0], [20.2, 0, 1], [21.4, 0, 0], [22.6, 0, 1], [23.8, 0, 0],
  [27, 0, 1], [28.2, 1, 1, 1], [29.4, 1, 1], [30.6, 1, 0], [31.8, 1, 1], [33, 2, 1, 1],
];
const FINAL = { pulls: [92, 410, 58], est: TRUE };

function stateAt(t) {
  const n = [0, 0, 0], w = [0, 0, 0];
  for (const [pt, m, win] of PULLS) if (t >= pt + 0.5) { n[m]++; w[m] += win; }
  const est = n.map((k, m) => (k ? w[m] / k : null));
  const conv = smooth((t - 35.2) / 6);
  if (conv > 0) {
    return { n: n.map((k, m) => Math.round(lerp(k, FINAL.pulls[m], conv))),
             est: est.map((e, m) => lerp(e ?? 0, FINAL.est[m], conv)), conv };
  }
  return { n, est, conv: 0 };
}

export function setup({ scene, camera, render, hud }) {
  const X = (m) => (m - 1) * 3.4;
  const machines = NAMES.map((name, m) => {
    const g = new THREE.Group();
    const body = new THREE.Mesh(new THREE.BoxGeometry(2.0, 2.6, 1.3), mat(COLORS[m]));
    body.position.y = 1.3; body.castShadow = true; g.add(body);
    const top = new THREE.Mesh(new THREE.CylinderGeometry(0.65, 0.65, 2.0, 24, 1, false, 0, Math.PI), mat(COLORS[m]));
    top.rotation.z = Math.PI / 2; top.position.y = 2.6; g.add(top);
    const win = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.7, 0.05), mat(0x0f172a, { emissive: 0x1e293b, emissiveIntensity: 0.4 }));
    win.position.set(0, 1.75, 0.66); g.add(win);
    const reels = [-0.42, 0, 0.42].map((x) => {
      const r = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.5, 0.04), mat(0xffffff));
      r.position.set(x, 1.75, 0.7); g.add(r); return r;
    });
    const tray = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.18, 0.5), mat(0x334155));
    tray.position.set(0, 0.55, 0.8); g.add(tray);
    const arm = new THREE.Group();
    const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 1.1), mat(0x94a3b8, { metalness: 0.8 }));
    rod.position.y = 0.55; arm.add(rod);
    const knob = new THREE.Mesh(new THREE.SphereGeometry(0.17, 16, 12), mat(0xdc2626));
    knob.position.y = 1.1; arm.add(knob);
    arm.position.set(1.12, 1.4, 0); g.add(arm);
    const name3 = label(name, { size: 46, color: "#ffffff", bg: null });
    name3.position.set(0, 2.85, 0.75); g.add(name3);
    g.position.set(X(m), 0, 0); scene.add(g);
    // estimate bar on a pedestal in front
    const ped = new THREE.Mesh(new THREE.BoxGeometry(1.1, 0.12, 1.1), mat(0xe2e8f0));
    ped.position.set(X(m), 0.06, 2.6); scene.add(ped);
    const bar = new THREE.Mesh(new THREE.BoxGeometry(0.8, 1, 0.8), mat(COLORS[m], { transparent: true, opacity: 0.92 }));
    bar.castShadow = true; scene.add(bar);
    const ghost = new THREE.Mesh(new THREE.BoxGeometry(0.95, 1, 0.95), new THREE.MeshBasicMaterial({ color: 0x0f172a, wireframe: true, transparent: true, opacity: 0.35 }));
    ghost.scale.y = TRUE[m] * 4; ghost.position.set(X(m), 0.12 + TRUE[m] * 2, 2.6); scene.add(ghost);
    const est = dynLabel({ size: 30 }); est.position.set(X(m), 0.2, 3.5); scene.add(est);
    const pop = dynLabel({ size: 40 }); scene.add(pop);
    const c = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.06, 28), mat(0xfbbf24, { metalness: 0.6, roughness: 0.3 }));
    c.rotation.x = Math.PI / 2; scene.add(c);
    return { g, arm, reels, bar, ghost, est, pop, coin: c };
  });
  const warn = label("never retried", { size: 28, color: C.red }); scene.add(warn);
  const dice = label("ε: random pick", { size: 28, color: C.amber }); scene.add(dice);
  const best = label("best ★", { size: 32, color: C.green }); scene.add(best);

  const KEYS = [[0, [0, 6, 16], [0, 1.6, 0.6]], [4.5, [0, 4.6, 12.5], [0, 1.6, 0.8]], [11, [0, 4.2, 11.5], [0, 1.4, 1.2]],
                [18, [-2.4, 4.4, 10.5], [-1.4, 1.4, 1.2]], [26, [1.4, 4.4, 10.5], [0.8, 1.4, 1.2]],
                [35, [0, 5.5, 12], [0, 1.4, 1.2]], [43, [0, 8, 13], [0, 1, 0.5]], [52, [0, 8.5, 14], [0, 1, 0.5]]];

  return function renderAt(t) {
    const st = stateAt(t);
    machines.forEach((M, m) => {
      // most recent pull of this machine
      const last = [...PULLS].reverse().find(([pt, mm]) => mm === m && t >= pt && t < pt + 1.6);
      const lk = last ? (t - last[0]) / 1.6 : 1;
      M.arm.rotation.x = last ? Math.sin(clamp(lk * 2.2) * Math.PI) * 0.9 : 0;
      M.reels.forEach((r, j) => { r.position.y = 1.75 + (last && lk < 0.45 ? Math.sin(t * 40 + j) * 0.05 : 0); });
      const won = last && last[2];
      M.coin.visible = !!last && lk > 0.3;
      M.coin.material.color.set(won ? 0xfbbf24 : 0x94a3b8);
      M.coin.position.set(X(m), 0.7 + ease.out((lk - 0.3) / 0.5) * 1.6, 1.1);
      M.coin.rotation.z = t * 6;
      M.pop.visible = !!last && lk > 0.35;
      M.pop.userData.set(won ? "+1 coin" : "nothing", won ? C.green : C.muted);
      M.pop.position.set(X(m), 3.2 + lk * 0.4, 1.0);
      M.pop.material.opacity = fade(lk, 0.35, 1, 0.15);
      const e = st.est[m], h = e === null ? 0.05 : Math.max(0.05, e * 4);
      M.bar.scale.y = h; M.bar.position.set(X(m), 0.12 + h / 2, 2.6);
      M.ghost.visible = st.conv > 0.05;
      M.ghost.material.opacity = 0.35 * st.conv;
      M.est.userData.set(e === null ? "? (no pulls)" : `${st.n[m] ? Math.round(e * st.n[m]) : 0}/${st.n[m]} = ${e.toFixed(2)}`);
      M.est.position.set(X(m), 0.12 + h + 0.45, 2.6);
      M.est.visible = t > 11;
    });
    warn.position.set(X(1), 1.1, 3.4);
    warn.material.opacity = fade(t, 19, 26, 0.5);
    const lastExplore = [...PULLS].reverse().find(([pt, , , ex]) => ex && t >= pt && t < pt + 1.8);
    dice.visible = !!lastExplore;
    if (lastExplore) dice.position.set(X(lastExplore[1]), 4.1, 0.6);
    best.position.set(X(1), 4.0, 0); best.material.opacity = fade(t, 39, 60, 0.6);
    cameraPath(camera, KEYS, t);
    render();

    hud.clear();
    hud.title("Multi-armed bandits", "explore or exploit?", fade(t, -1, 4.3, 0.7));
    const c = story(hud, CH, t);
    if (c.i === 6) {
      const k = clamp(c.local / 3);
      const T = 80, greedy = [], eps = [];
      for (let s = 0; s < T; s++) { greedy.push(s * (0.7 - 0.35)); eps.push(Math.min(s, 12) * 0.25 + Math.max(0, s - 12) * 0.035 * 2); }
      hud.chart(330, 70, 620, 300, [{ data: greedy, color: C.red, upto: k * T }, { data: eps, color: C.green, upto: k * T }],
        { title: "Total regret over pulls (coins lost)", n: T, range: [0, 28], digits: 0 });
      hud.text("greedy (stuck on A)", 930, 120, 15, C.red, "right", 700);
      hud.text("ε-greedy", 930, 290, 15, C.green, "right", 700);
    }
    if (c.i >= 2 && c.i <= 4) {
      hud.panel(30, 44, 270, 64);
      const mode = c.i === 2 ? "try each once" : c.i === 3 ? "greedy: pick the max" : "ε-greedy: 90% max, 10% random";
      hud.text("Rule", 48, 70, 14, C.muted, "left", 700);
      hud.text(mode, 48, 94, 17, C.ink, "left", 650);
    }
  };
}
