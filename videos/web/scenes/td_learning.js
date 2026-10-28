// TD vs Monte Carlo on two identical corridors: MC waits for the episode to end and updates every visited cell to
// the real return; TD updates each cell right after a step, from the next cell's current guess. Watch the +10
// propagate back one cell per episode under TD, all at once under MC.
import { C, THREE, cameraPath, clamp, dynLabel, fade, fmt, label, lerp, mat, robot, smooth, story } from "../lib.js";

export const meta = {
  title: "TD learning vs Monte Carlo", path: "rl", duration: 50,
  desc: "Two robots, two corridors: one learns after every step (TD), one waits for the end (Monte Carlo).",
  keywords: ["Monte Carlo Prediction", "TD(0)", "SARSA", "Q-Learning", "Temporal-Difference", "Function Approximation",
             "DQN Part 1", "DQN Part 2"],
  poster: 30,
};

const L = 6, R = 10, ALPHA = 0.5, MOVE = 0.55;        // states 0..5, goal after state 5, gamma = 1, reward only at goal
const EP = [10.5, 19, 27.5, 36];                      // episode start times
const CH = [
  { t: 0, title: "", text: "" },
  { t: 4.5, title: "Two learners, one task", text: "Same corridor twice. Reward +10 only at the end. Both start with every value at 0. Outlines show the true value (10)." },
  { t: 10.5, title: "Episode 1", text: "Monte Carlo (top) waits, then sets every cell toward the real return. TD (bottom) only learns at the last cell: it's the only step with new information." },
  { t: 19, title: "Episode 2", text: "TD's update: V(s) ← V(s) + α [ r + V(s′) − V(s) ]. The value of the last cell now flows one cell back." },
  { t: 27.5, title: "Episode 3", text: "Each episode, TD's knowledge creeps one cell further back. MC jumps straight to its averages." },
  { t: 36, title: "Episode 4", text: "TD bootstraps: it learns from its own guesses. Slower here, but it never needs the episode to finish." },
  { t: 44, title: "The trade-off", text: "MC: unbiased but noisy and must wait for the end. TD: learns every step with lower variance, from biased guesses. Q-learning and DQN are TD." },
];

// simulate value tables after each move, for both learners
function simulate() {
  const mc = [Array(L).fill(0)], td = [Array(L).fill(0)];
  for (let e = 0; e < EP.length; e++) {
    let v = td.at(-1).slice();
    for (let s = 0; s < L; s++) {
      const target = s === L - 1 ? R : v[s + 1];
      v = v.slice(); v[s] += ALPHA * (target - v[s]);
      td.push(v);
      mc.push(mc.at(-1).slice());            // MC doesn't change during the episode
    }
    const m = mc.at(-1).map((x) => x + ALPHA * (R - x));
    mc[mc.length - 1] = m;                   // MC updates everything at the end
  }
  return { mc, td };
}
const SIM = simulate();
const idxAt = (t) => {
  for (let e = EP.length - 1; e >= 0; e--) {
    if (t >= EP[e]) {
      const k = clamp((t - EP[e] - 0.6) / MOVE, 0, L);
      return { e, step: Math.floor(k), frac: k - Math.floor(k), moving: k < L, done: k >= L, k };
    }
  }
  return { e: -1, step: 0, frac: 0, moving: false, done: false, k: 0 };
};

export function setup({ scene, camera, render, hud }) {
  const X = (s) => (s - 2.5) * 1.6;
  const rows = [{ z: -1.9, name: "Monte Carlo", color: 0x2563eb, key: "mc" }, { z: 1.9, name: "TD(0)", color: 0x7c3aed, key: "td" }];
  rows.forEach((r) => {
    for (let s = 0; s < L; s++) {
      const tile = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.14, 1.5), mat(0xf1f5f9));
      tile.position.set(X(s), 0.07, r.z); tile.receiveShadow = true; scene.add(tile);
    }
    const g = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.14, 1.5), mat(0x86efac));
    g.position.set(X(L), 0.07, r.z); scene.add(g);
    const gl = label("+10", { size: 32, color: C.green }); gl.position.set(X(L), 0.9, r.z); scene.add(gl);
    const nm = label(r.name, { size: 34, color: r.key === "mc" ? C.blue : C.accent }); nm.position.set(X(-1) - 0.3, 0.8, r.z); scene.add(nm);
    r.bars = Array.from({ length: L }, (_, s) => {
      const b = new THREE.Mesh(new THREE.BoxGeometry(1.0, 1, 1.0), mat(r.color, { transparent: true, opacity: 0.88 }));
      b.castShadow = true; b.position.set(X(s), 0, r.z); scene.add(b);
      const gh = new THREE.Mesh(new THREE.BoxGeometry(1.15, 1, 1.15), new THREE.MeshBasicMaterial({ color: 0x0f172a, wireframe: true, transparent: true, opacity: 0.25 }));
      gh.scale.y = R * 0.28; gh.position.set(X(s), 0.14 + R * 0.14, r.z); scene.add(gh);
      const n = dynLabel({ size: 24 }); scene.add(n);
      return { b, n };
    });
    r.bot = robot(r.color); scene.add(r.bot);
    r.flash = dynLabel({ size: 30 }); scene.add(r.flash);
    r.beams = Array.from({ length: L }, () => {
      const m = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 1, 10), mat(0xf59e0b, { emissive: 0xf59e0b, emissiveIntensity: 0.6, transparent: true }));
      scene.add(m); return m;
    });
  });

  const KEYS = [[0, [0, 8, 13], [0, 0.6, 0]], [4.5, [0, 7.5, 11], [0, 0.8, 0]], [10.5, [-2, 6.5, 10], [0, 1, 0]],
                [27.5, [2, 6.5, 10], [0, 1, 0]], [44, [0, 9, 11], [0, 1, 0]], [50, [0, 9.5, 11.5], [0, 1, 0]]];

  return function renderAt(t) {
    const I = idxAt(t);
    // index into the simulated tables: (episode, move) -> row
    const base = I.e < 0 ? 0 : I.e * L;
    const live = I.e < 0 ? 0 : base + Math.min(L, I.step + (I.frac > 0.55 ? 1 : 0));
    const mcEnd = I.e >= 0 && I.done && t - EP[I.e] - 0.6 - L * MOVE > 0.3;
    rows.forEach((r) => {
      const table = r.key === "td" ? SIM.td[live] : SIM[r.key][I.e < 0 ? 0 : base + (mcEnd ? L : 0)];
      const prev = r.key === "td" ? SIM.td[Math.max(0, live - 1)] : SIM.mc[Math.max(0, base)];
      r.bars.forEach(({ b, n }, s) => {
        let v = table[s];
        if (r.key === "mc" && mcEnd) v = lerp(prev[s], table[s], smooth((t - EP[I.e] - 0.9 - L * MOVE) / 0.8));
        const h = Math.max(0.03, v * 0.28);
        b.scale.y = h; b.position.y = 0.14 + h / 2;
        n.userData.set(fmt(v, 1)); n.position.set(X(s), 0.14 + h + 0.35, r.z);
        n.visible = t > 4.5;
      });
      // robot walking
      const k = I.e < 0 ? 0 : I.k, s0 = Math.min(L, Math.floor(k)), fr = k - s0;
      const x = lerp(X(s0), X(Math.min(L, s0 + 1)), smooth(fr));
      r.bot.position.set(I.e < 0 ? X(0) : x, 0.16 + (I.moving ? Math.sin(fr * Math.PI) * 0.25 : 0), r.z + 0.85);
      r.bot.visible = t > 4.5;
      // TD: highlight the cell just updated; MC: beams on every cell at the end
      r.flash.visible = false;
      if (r.key === "td" && I.moving && I.frac > 0.55 && I.e >= 0) {
        const s = I.step, delta = SIM.td[base + s + 1][s] - SIM.td[base + s][s];
        if (Math.abs(delta) > 0.01) {
          r.flash.visible = true; r.flash.userData.set(`δ ${delta > 0 ? "+" : ""}${fmt(delta / ALPHA, 1)}`, C.amber);
          r.flash.position.set(X(s), 3.6, r.z);
        }
      }
      r.beams.forEach((m, s) => {
        const on = r.key === "mc" && mcEnd && t - EP[I.e] - 0.6 - L * MOVE < 1.6;
        m.visible = on;
        if (on) { m.scale.y = 3.2; m.position.set(X(s), 1.8, r.z); m.material.opacity = 0.5; }
      });
    });
    cameraPath(camera, KEYS, t);
    render();

    hud.clear();
    hud.title("TD learning vs Monte Carlo", "learn after every step, or wait for the ending?", fade(t, -1, 4.3, 0.7));
    const c = story(hud, CH, t);
    if (c.i >= 2 && c.i <= 5) {
      hud.panel(40, 46, 300, 116);
      hud.text(`Episode ${I.e + 1}`, 60, 78, 19, C.ink, "left", 750);
      hud.text("MC:  V ← V + α (G − V)", 60, 108, 15, C.blue, "left", 650);
      hud.text("TD:  V ← V + α (r + V(s′) − V)", 60, 136, 15, C.accent, "left", 650);
    }
  };
}
