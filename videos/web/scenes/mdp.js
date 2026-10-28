// MDPs, returns and value: an agent on a grid, one episode, rewards collected, the return, discounting with
// shrinking coins, the value of every cell, and the greedy policy that follows from it.
import { C, THREE, cameraPath, clamp, coin, dynLabel, ease, fade, fmt, floorArrow, label, lerp, mat, robot, smooth, story } from "../lib.js";

export const meta = {
  title: "MDPs, returns and value", path: "rl", duration: 50,
  desc: "States, actions and rewards; one episode; why we add up future rewards, why we discount them, and what V(s) means.",
  keywords: ["RL is choosing actions", "MDPs and Discounted Return", "Value Functions", "Autoregressive LLMs as RL Policies"],
  poster: 25,
};

const N = 5, GOAL = [4, 4], GAMMA = 0.9, STEP = -0.5, WIN = 10;
const PATH = [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [4, 4]];
const REW = PATH.slice(1).map((c, k) => (k === PATH.length - 2 ? WIN : STEP));
const dist = (x, y) => Math.abs(GOAL[0] - x) + Math.abs(GOAL[1] - y);
const V = (x, y) => { const d = dist(x, y); let v = 0; for (let k = 0; k < d - 1; k++) v += STEP * GAMMA ** k; return v + WIN * GAMMA ** (d - 1); };
const P = (x, y) => new THREE.Vector3((x - 2) * 1.5, 0, (2 - y) * 1.5);
const CH = [
  { t: 0, title: "", text: "" },
  { t: 4.5, title: "The setup", text: "An agent (the robot) in an environment (the grid). It sees its state, picks an action, and gets a reward." },
  { t: 11, title: "One episode", text: "Each move costs −0.5. Reaching the green goal pays +10 and ends the episode." },
  { t: 19, title: "The return G", text: "Add up every reward collected from the start: 7 × (−0.5) + 10 = 6.5. RL maximises this." },
  { t: 26, title: "Discounting (γ = 0.9)", text: "A reward k steps away counts γᵏ as much. Far coins shrink: +10 seven steps away is worth only 4.8 now." },
  { t: 33, title: "Value of a state", text: "V(s) = expected discounted return starting from s. Cells near the goal are worth more." },
  { t: 41, title: "The policy follows", text: "Acting well = moving toward higher value. From every cell, the arrows point the way to the goal." },
];

export function setup({ scene, camera, render, hud }) {
  for (let x = 0; x < N; x++) for (let y = 0; y < N; y++) {
    const goal = x === GOAL[0] && y === GOAL[1], start = x === 0 && y === 0;
    const m = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.16, 1.4), mat(goal ? 0x86efac : start ? 0xdbeafe : 0xf1f5f9));
    m.position.copy(P(x, y)).setY(0.08); m.receiveShadow = true; scene.add(m);
  }
  const flag = new THREE.Group();
  flag.add(new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 1.4), mat(0x334155)));
  const cloth = new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.32, 0.03), mat(0x16a34a));
  cloth.position.set(0.28, 0.5, 0); flag.add(cloth);
  flag.position.copy(P(...GOAL)).setY(0.85); scene.add(flag);
  const gl = label("goal +10", { size: 30, color: C.green }); gl.position.copy(P(...GOAL)).setY(2.0); scene.add(gl);
  const bot = robot(); scene.add(bot);
  const pop = dynLabel({ size: 36 }); scene.add(pop);
  const coins = PATH.slice(1).map((c, k) => {
    const m = coin(k === PATH.length - 2 ? 0.42 : 0.24);
    if (k !== PATH.length - 2) m.material = mat(0x94a3b8, { metalness: 0.5 });
    scene.add(m);
    const l = dynLabel({ size: 26 }); scene.add(l);
    return { m, l, c, k };
  });
  const bars = [], arrows = [];
  for (let x = 0; x < N; x++) for (let y = 0; y < N; y++) {
    if (x === GOAL[0] && y === GOAL[1]) continue;
    const b = new THREE.Mesh(new THREE.BoxGeometry(0.9, 1, 0.9), mat(0x8b5cf6, { transparent: true, opacity: 0.85 }));
    b.castShadow = true; b.userData = { x, y, v: V(x, y) }; scene.add(b); bars.push(b);
    const a = floorArrow(0x0f172a, 0.75);
    const [dx, dy] = x < GOAL[0] ? [1, 0] : [0, 1];
    a.userData.point(dx, -dy); a.userData.xy = [x, y]; scene.add(a); arrows.push(a);
  }
  const vl = [[0, 0], [3, 4], [2, 2]].map(([x, y]) => { const l = label(`V = ${fmt(V(x, y))}`, { size: 28, color: C.accent }); l.userData = { x, y }; scene.add(l); return l; });
  const confetti = Array.from({ length: 40 }, (_, k) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.1, 0.02), mat([0xf59e0b, 0x16a34a, 0x7c3aed, 0xdb2777][k % 4]));
    scene.add(m); return m;
  });

  const MOVE0 = 11.4, DT = 0.85;
  const KEYS = [[0, [8, 10, 11], [0, 0, 0]], [4.5, [0, 8, 10], [0, 0.5, 0.5]], [11, [-1, 7.5, 9.5], [0, 0.5, 0.5]],
                [19, [0, 9, 8.5], [0, 0, 0]], [26, [5, 7, 9], [0, 0.5, 0]], [33, [-7, 7, 8], [0, 1, 0]],
                [41, [0, 11, 6], [0, 0, 0.3]], [50, [2, 11.5, 7], [0, 0, 0.3]]];

  return function renderAt(t) {
    // robot: walks the episode (ch 2), then replays the greedy path (ch 6)
    let pos = P(0, 0), hop = 0, step = 0;
    const walk = (t0, dt) => {
      const f = clamp((t - t0) / dt, 0, PATH.length - 1);
      step = Math.floor(f);
      const fr = f - step, a = P(...PATH[step]), b = P(...PATH[Math.min(PATH.length - 1, step + 1)]);
      pos = a.clone().lerp(b, smooth(fr)); hop = step < PATH.length - 1 ? Math.sin(fr * Math.PI) * 0.3 : 0;
      return f;
    };
    const f = t < 41 ? walk(MOVE0, DT) : walk(42.5, 0.6);
    bot.position.copy(pos).setY(0.16 + hop);
    bot.rotation.y = Math.sin(t * 2) * 0.15;
    // reward popup per move during the episode
    const mv = Math.floor(f - 1e-6);
    pop.visible = t > MOVE0 && t < 19 && mv >= 0 && mv < REW.length && f - mv > 0.6;
    if (pop.visible) {
      pop.userData.set(REW[mv] > 0 ? "+10" : "−0.5", REW[mv] > 0 ? C.green : C.red);
      pop.position.copy(P(...PATH[mv + 1])).setY(1.6 + (f - mv) * 0.3);
    }
    // confetti at the goal
    const ck = (t - (MOVE0 + DT * (PATH.length - 1))) / 2.2;
    confetti.forEach((m, k) => {
      m.visible = ck > 0 && ck < 1;
      const a = (k / confetti.length) * Math.PI * 2;
      m.position.copy(P(...GOAL)).add(new THREE.Vector3(Math.cos(a) * ck * 1.5, 1.5 + ck * 2 - ck * ck * 3, Math.sin(a) * ck * 1.5));
      m.rotation.set(t * 5 + k, t * 3, 0);
    });
    // coins: appear (ch 3), shrink by gamma^k (ch 4), fade (ch 5)
    coins.forEach(({ m, l, c, k }) => {
      const appear = ease.back((t - 19.4 - k * 0.45) / 0.5);
      const shrink = smooth((t - 26.6 - k * 0.6) / 0.6);
      const gone = smooth((t - 33) / 1.2);
      const s = Math.max(0.001, appear * lerp(1, GAMMA ** k, shrink) * (1 - gone));
      m.visible = s > 0.002;
      m.scale.setScalar(s);
      m.position.copy(P(...c)).setY(0.55 + Math.sin(t * 2 + k) * 0.05);
      m.rotation.z = t * 1.5;
      const val = REW[k] * lerp(1, GAMMA ** k, shrink);
      l.userData.set(fmt(val, Math.abs(val) < 1 ? 2 : 1), REW[k] > 0 ? C.green : C.muted);
      l.visible = m.visible; l.position.copy(P(...c)).setY(1.2);
      l.material.opacity = Math.min(appear, 1 - gone);
    });
    bars.forEach((b) => {
      const { x, y, v } = b.userData;
      const k = smooth((t - 33.5 - (8 - dist(x, y)) * 0.18) / 0.8);
      const h = Math.max(0.02, Math.abs(v) * 0.32 * k);
      b.visible = k > 0.01; b.scale.y = h; b.position.copy(P(x, y)).setY(0.16 + h / 2);
      b.material.color.set(v >= 0 ? 0x8b5cf6 : 0xef4444);
    });
    arrows.forEach((a) => {
      const [x, y] = a.userData.xy, k = ease.back((t - 41.3 - (x + y) * 0.06) / 0.5);
      a.visible = k > 0.01; a.scale.setScalar(Math.max(0.01, k));
      a.position.copy(P(x, y)).setY(0.2 + Math.abs(V(x, y)) * 0.32 + 0.05);
    });
    vl.forEach((l) => { const { x, y } = l.userData; l.position.copy(P(x, y)).setY(0.6 + Math.abs(V(x, y)) * 0.32); l.material.opacity = fade(t, 35, 41.2, 0.5); });
    cloth.rotation.y = Math.sin(t * 3) * 0.25;
    cameraPath(camera, KEYS, t);
    render();

    hud.clear();
    hud.title("MDPs, returns and value", "what an RL agent is actually trying to maximise", fade(t, -1, 4.3, 0.7));
    const c = story(hud, CH, t);
    if (c.i === 1) {   // agent-environment loop
      const k = (t * 0.6) % 1;
      hud.panel(40, 50, 320, 150);
      hud.panel(70, 80, 100, 44, k < 0.5 ? "#ede9fe" : "#fff"); hud.text("Agent", 120, 108, 17, C.accent, "center", 700);
      hud.panel(230, 80, 110, 44, k >= 0.5 ? "#dbeafe" : "#fff"); hud.text("Grid", 285, 108, 17, C.blue, "center", 700);
      hud.text("action a →", 200, 150, 15, k < 0.5 ? C.accent : C.muted, "center", 650);
      hud.text("← state s′, reward r", 200, 180, 15, k >= 0.5 ? C.blue : C.muted, "center", 650);
    }
    if (c.i === 2 || c.i === 3) {
      const got = REW.slice(0, Math.max(0, Math.min(REW.length, Math.floor(f))));
      const G = got.reduce((a, b) => a + b, 0);
      hud.panel(40, 50, 330, 100);
      hud.text(c.i === 2 ? "Rewards so far" : "Return from the start", 60, 80, 15, C.muted, "left", 700);
      hud.text(got.length ? got.map((r) => (r > 0 ? "+10" : "−0.5")).join(" ") : "—", 60, 106, 14, C.ink);
      hud.text(`G = ${fmt(G)}`, 60, 136, 20, C.ink, "left", 750);
    }
    if (c.i === 4) {
      const shr = REW.map((r, k) => r * lerp(1, GAMMA ** k, smooth((t - 26.6 - k * 0.6) / 0.6)));
      const G = shr.reduce((a, b) => a + b, 0);
      hud.panel(40, 50, 380, 110);
      hud.text("G = r₁ + γ r₂ + γ² r₃ + …", 60, 82, 19, C.accent, "left", 700);
      hud.text(`undiscounted 6.5  →  discounted ${fmt(G, 2)}`, 60, 112, 16, C.ink, "left", 600);
      hud.text("γ < 1: sooner is better, and G stays finite", 60, 140, 14, C.muted);
    }
    if (c.i === 5) {
      hud.panel(40, 50, 340, 80);
      hud.text("V(s) = E[ G | start in s ]", 60, 84, 20, C.accent, "left", 700);
      hud.text("taller bar = better place to be", 60, 112, 15, C.muted);
    }
  };
}
