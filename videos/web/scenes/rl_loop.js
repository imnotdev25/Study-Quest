// RL in one loop: an agent on a gridworld acts, the environment returns a new state and a reward, and over
// episodes the agent finds a better path. Ends with returns per episode.
import { C, THREE, cameraPath, clamp, fade, hex, label, lerp, mat, smooth } from "../lib.js";

export const meta = {
  title: "The RL loop", path: "rl", duration: 32,
  desc: "An agent acts, the environment answers with a new state and a reward, and returns improve episode by episode.",
  keywords: ["RL is choosing actions", "MDPs and Discounted Return", "Value Functions", "Monte Carlo Prediction", "TD(0)"],
  poster: 13,
};

const N = 5, GOAL = [4, 4], PIT = [2, 3], WALLS = [[1, 1], [2, 1], [3, 3]];
const MOVES = { R: [1, 0], L: [-1, 0], U: [0, 1], D: [0, -1] };
// Three episodes: falls in the pit, wanders to the goal, then takes a short path.
const EPISODES = [
  { start: 4.5, path: "UURRU", end: "pit" },
  { start: 11.5, path: "RRRUULRRUU", end: "goal" },
  { start: 19.5, path: "UUUURRRR", end: "goal" },
];
const STEP = 0.55, GAMMA = 0.9;

function cells(path) {
  const out = [[0, 0]];
  for (const m of path) { const [x, y] = out.at(-1); out.push([x + MOVES[m][0], y + MOVES[m][1]]); }
  return out;
}
const W = (x, y) => new THREE.Vector3((x - 2) * 1.6, 0, (2 - y) * 1.6);
const isAt = (c, p) => c[0] === p[0] && c[1] === p[1];

export function setup({ scene, camera, render, hud }) {
  const tiles = [];
  for (let x = 0; x < N; x++) for (let y = 0; y < N; y++) {
    const wall = WALLS.some((w) => isAt(w, [x, y]));
    const col = isAt([x, y], GOAL) ? 0x86efac : isAt([x, y], PIT) ? 0xfca5a5 : wall ? 0x94a3b8 : 0xeef2ff;
    const m = new THREE.Mesh(new THREE.BoxGeometry(1.5, wall ? 1.2 : 0.2, 1.5), mat(col));
    m.position.copy(W(x, y)).setY(wall ? 0.6 : 0.1);
    m.castShadow = m.receiveShadow = true;
    scene.add(m); tiles.push(m);
  }
  const flag = new THREE.Group();
  flag.add(new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 1.4), mat(0x334155)));
  const cloth = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.35, 0.03), mat(0x16a34a));
  cloth.position.set(0.3, 0.5, 0); flag.add(cloth);
  flag.position.copy(W(...GOAL)).setY(0.9); scene.add(flag);
  const lg = label("+10 goal", { color: C.green }); lg.position.copy(W(...GOAL)).setY(2.0); scene.add(lg);
  const lp = label("−10 pit", { color: C.red }); lp.position.copy(W(...PIT)).setY(1.1); scene.add(lp);

  // agent: a small robot
  const agent = new THREE.Group();
  const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.35, 0.4, 8, 16), mat(0x7c3aed));
  body.position.y = 0.6; body.castShadow = true; agent.add(body);
  for (const s of [-1, 1]) {
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.07, 12, 8), mat(0xffffff, { emissive: 0xffffff, emissiveIntensity: 0.5 }));
    eye.position.set(s * 0.13, 0.78, 0.3); agent.add(eye);
  }
  scene.add(agent);
  // trail of visited cells
  const trail = Array.from({ length: 12 }, () => {
    const d = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 0.04, 20), mat(0xc4b5fd));
    d.visible = false; scene.add(d); return d;
  });

  const KEYS = [[0, [9, 9, 11], [0, 0, 0]], [4.5, [0, 10, 9.5], [0, 0, 0.3]], [26, [0, 10, 9.5], [0, 0, 0.3]], [32, [-6, 12, 12], [0, 0, 0]]];

  function episodeAt(t) {
    let ep = null;
    for (const e of EPISODES) if (t >= e.start) ep = e;
    return ep;
  }

  return function renderAt(t) {
    const ep = episodeAt(t);
    let pos = W(0, 0), hop = 0, sink = 0, k = 0, cs = [[0, 0]], rewards = [];
    if (ep) {
      cs = cells(ep.path);
      const f = (t - ep.start) / STEP;
      k = Math.min(cs.length - 1, Math.floor(f));
      const frac = clamp(f - k);
      const a = W(...cs[k]), b = W(...cs[Math.min(cs.length - 1, k + 1)]);
      pos = a.clone().lerp(b, smooth(frac));
      hop = k < cs.length - 1 ? Math.sin(frac * Math.PI) * 0.35 : 0;
      if (k >= cs.length - 1 && ep.end === "pit") sink = clamp((f - (cs.length - 1)) / 1.5);
      rewards = cs.slice(1, k + 1).map((c) => (isAt(c, GOAL) ? 10 : isAt(c, PIT) ? -10 : -1));
    }
    agent.position.copy(pos).setY(hop - sink * 1.2);
    agent.scale.setScalar(1 - sink * 0.6);
    agent.rotation.y = t * 0.6;
    trail.forEach((d, i) => {
      d.visible = !!ep && i < k && i < cs.length;
      if (d.visible) d.position.copy(W(...cs[i])).setY(0.23);
    });
    cloth.rotation.y = Math.sin(t * 3) * 0.2;
    cameraPath(camera, KEYS, t);
    render();

    hud.clear();
    hud.title("The RL loop", "act → observe → get rewarded → do better next time", fade(t, -1, 4.2, 0.8));
    // loop diagram
    const a = fade(t, 4.5, 26.5);
    if (a > 0) {
      hud.alpha(a);
      hud.panel(30, 30, 300, 210);
      hud.text("Every step", 50, 62, 19, C.ink, "left", 700);
      const phase = ep ? Math.floor(((t - ep.start) / STEP) * 3) % 3 : -1;
      [["Agent picks an action a", C.accent], ["Environment moves to state s′", C.blue], ["…and pays a reward r", C.green]].forEach(([s, col], i) => {
        hud.bar(50, 82 + i * 46, 10, 30, phase === i ? col : "#e2e8f0");
        hud.text(s, 72, 104 + i * 46, 17, phase === i ? C.ink : C.muted, "left", phase === i ? 650 : 500);
      });
      if (ep) {
        const G = rewards.reduce((s, r, i) => s + r * GAMMA ** i, 0);
        const n = EPISODES.indexOf(ep) + 1;
        hud.panel(950, 30, 300, 130);
        hud.text(`Episode ${n}`, 970, 62, 19, C.ink, "left", 700);
        hud.text(`steps ${rewards.length}   last reward ${rewards.length ? rewards.at(-1) : "–"}`, 970, 94, 16, C.muted);
        hud.text(`return G = Σ γᵗ rₜ = ${G.toFixed(1)}`, 970, 126, 17, G >= 0 ? C.green : C.red, "left", 650);
      }
      hud.alpha(1);
    }
    hud.caption("Episode 1: no idea yet. It walks straight into the pit.", fade(t, 5, 11.2, 0.4));
    hud.caption("Episode 2: it reaches the goal, but the long way: every step costs −1.", fade(t, 11.8, 19.2, 0.4));
    hud.caption("Episode 3: experience says go up, then right. Fewer steps, bigger return.", fade(t, 19.8, 26.2, 0.4));
    // returns summary
    const s = fade(t, 26.5, 40, 0.6);
    if (s > 0) {
      hud.alpha(s);
      hud.panel(340, 150, 600, 330);
      hud.text("Return per episode (γ = 0.9)", 370, 192, 22, C.ink, "left", 700);
      EPISODES.forEach((e, i) => {
        const cs2 = cells(e.path).slice(1);
        const G = cs2.reduce((acc, c, j) => acc + (isAt(c, GOAL) ? 10 : isAt(c, PIT) ? -10 : -1) * GAMMA ** j, 0);
        const y = 230 + i * 70, x0 = 640, w = (G / 10) * 260 * smooth((t - 27) / 1.5);
        hud.text(`Episode ${i + 1}`, 370, y + 26, 18, C.ink, "left", 600);
        hud.bar(Math.min(x0, x0 + w), y + 8, Math.abs(w), 28, G >= 0 ? C.green : C.red);
        hud.text(G.toFixed(1), G >= 0 ? x0 + w + 10 : x0 + w - 10, y + 28, 16, C.ink, G >= 0 ? "left" : "right", 650);
      });
      hud.text("The goal of RL: a policy that maximises expected return.", 640, 455, 17, C.muted, "center");
      hud.alpha(1);
    }
  };
}
