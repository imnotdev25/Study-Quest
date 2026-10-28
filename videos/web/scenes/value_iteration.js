// Value iteration on a gridworld: each sweep backs values up from the goal, bars grow, then the greedy policy
// (arrows) falls out of the converged values.
import { C, THREE, cameraPath, clamp, fade, label, lerp, mat, smooth } from "../lib.js";

export const meta = {
  title: "Value iteration", path: "rl", duration: 32,
  desc: "Bellman backups spread value outward from the goal, sweep by sweep, until the greedy policy appears.",
  keywords: ["Value Iteration", "Value Functions", "MDPs and Discounted Return", "Bellman"],
  poster: 18,
};

const N = 5, GOAL = [4, 4], PIT = [2, 3], WALLS = [[1, 1], [2, 1], [3, 3]], GAMMA = 0.9, STEPC = -0.4;
const ACTS = [[1, 0, "→"], [-1, 0, "←"], [0, 1, "↑"], [0, -1, "↓"]];
const isWall = (x, y) => x < 0 || y < 0 || x >= N || y >= N || WALLS.some((w) => w[0] === x && w[1] === y);
const term = (x, y) => (x === GOAL[0] && y === GOAL[1] ? 10 : x === PIT[0] && y === PIT[1] ? -10 : null);

// Synchronous value iteration, deterministic moves; reward on entering a cell = STEPC, or the terminal value.
function sweeps(n) {
  const Vs = [Array.from({ length: N }, () => Array(N).fill(0))];
  for (let k = 0; k < n; k++) {
    const V = Vs.at(-1), nv = V.map((r) => r.slice());
    for (let x = 0; x < N; x++) for (let y = 0; y < N; y++) {
      if (isWall(x, y) || term(x, y) !== null) continue;
      nv[x][y] = Math.max(...ACTS.map(([dx, dy]) => q(V, x, y, dx, dy)));
    }
    Vs.push(nv);
  }
  return Vs;
}
function q(V, x, y, dx, dy) {
  let nx = x + dx, ny = y + dy;
  if (isWall(nx, ny)) { nx = x; ny = y; }
  const tv = term(nx, ny);
  return tv !== null ? tv : STEPC + GAMMA * V[nx][ny];
}
const VS = sweeps(9);
const delta = VS.slice(1).map((V, k) => Math.max(...V.flatMap((r, x) => r.map((v, y) => Math.abs(v - VS[k][x][y])))));
const SWEEP0 = 5, SWEEP_T = 1.7;           // sweep k runs from SWEEP0 + k*SWEEP_T
const P = (x, y) => new THREE.Vector3((x - 2) * 1.6, 0, (2 - y) * 1.6);

export function setup({ scene, camera, render, hud }) {
  const bars = [], arrows = [];
  for (let x = 0; x < N; x++) for (let y = 0; y < N; y++) {
    const wall = isWall(x, y), tv = term(x, y);
    const base = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.12, 1.5),
      mat(tv === 10 ? 0x86efac : tv === -10 ? 0xfca5a5 : wall ? 0x94a3b8 : 0xf1f5f9));
    base.position.copy(P(x, y)).setY(0.06); base.receiveShadow = true; scene.add(base);
    if (wall) {
      const w = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.9, 1.5), mat(0x94a3b8));
      w.position.copy(P(x, y)).setY(0.45); w.castShadow = true; scene.add(w);
      continue;
    }
    const bar = new THREE.Mesh(new THREE.BoxGeometry(1.1, 1, 1.1), mat(0x8b5cf6, { transparent: true, opacity: 0.92 }));
    bar.castShadow = true; bar.userData = { x, y }; scene.add(bar); bars.push(bar);
    if (tv === null) {
      const a = new THREE.Mesh(new THREE.ConeGeometry(0.22, 0.6, 16), mat(0x0f172a));
      a.userData = { x, y }; scene.add(a); arrows.push(a);
    }
  }
  const lg = label("goal +10", { color: C.green }); lg.position.copy(P(...GOAL)).setY(4.3); scene.add(lg);
  const lp = label("pit −10", { color: C.red }); lp.position.copy(P(...PIT)).setY(1.2); scene.add(lp);
  const scan = new THREE.Mesh(new THREE.BoxGeometry(8.4, 0.03, 0.12), mat(0xf59e0b, { emissive: 0xf59e0b, emissiveIntensity: 0.6 }));
  scene.add(scan);

  const KEYS = [[0, [10, 11, 12], [0, 1, 0]], [4.5, [7, 9, 10], [0, 1, 0]], [21, [-6, 10, 10], [0, 1, 0]], [32, [0, 14, 6], [0, 0, 0.5]]];
  const highlight = [1, 3];   // the cell whose backup we spell out

  return function renderAt(t) {
    const kf = clamp((t - SWEEP0) / SWEEP_T, 0, VS.length - 1);
    const k = Math.floor(kf), f = smooth(kf - k);
    for (const b of bars) {
      const { x, y } = b.userData;
      const v = lerp(VS[k][x][y], VS[Math.min(VS.length - 1, k + 1)][x][y], f);
      const h = Math.max(0.04, Math.abs(v) * 0.35);
      b.scale.y = h;
      b.position.copy(P(x, y)).setY(0.12 + h / 2);
      b.material.color.set(v >= 0 ? 0x8b5cf6 : 0xef4444);
      b.material.emissive.set(x === highlight[0] && y === highlight[1] && t > SWEEP0 && t < 21 ? 0xf59e0b : 0x000000);
      b.material.emissiveIntensity = 0.5;
    }
    const sweeping = t > SWEEP0 && kf < VS.length - 1;
    scan.visible = sweeping;
    scan.position.set(0, 0.2, lerp(4.2, -4.2, kf - k));
    const pa = smooth((t - 21.5) / 1.5);
    const V = VS.at(-1);
    for (const a of arrows) {
      const { x, y } = a.userData;
      const best = ACTS.reduce((bi, ac, i) => (q(V, x, y, ac[0], ac[1]) > q(V, x, y, ACTS[bi][0], ACTS[bi][1]) ? i : bi), 0);
      const [dx, dy] = ACTS[best];
      a.visible = pa > 0.01;
      a.scale.setScalar(pa);
      const h = Math.max(0.04, Math.abs(V[x][y]) * 0.35);
      a.position.copy(P(x, y)).setY(0.4 + h).add(new THREE.Vector3(dx * 0.25, 0, -dy * 0.25));
      a.rotation.set(0, 0, 0);
      if (dx) a.rotation.z = -dx * Math.PI / 2; else a.rotation.x = -dy * Math.PI / 2;
    }
    cameraPath(camera, KEYS, t);
    render();

    hud.clear();
    hud.title("Value iteration", "look one step ahead, everywhere, again and again", fade(t, -1, 4.2, 0.8));
    const a = fade(t, 4.5, 21.2);
    if (a > 0) {
      hud.alpha(a);
      hud.panel(30, 30, 430, 236);
      hud.text(`Sweep ${Math.min(VS.length - 1, k + 1)}`, 50, 62, 20, C.ink, "left", 700);
      hud.text("V(s) ← maxₐ [ r(s,a) + γ · V(s′) ]", 50, 96, 19, C.accent, "left", 650);
      hud.text(`γ = ${GAMMA}, each step costs ${STEPC}`, 50, 124, 15, C.muted);
      const [hx, hy] = highlight, Vk = VS[k];
      hud.text("Highlighted cell, this sweep:", 50, 156, 15, C.ink, "left", 600);
      ACTS.forEach(([dx, dy, s], i) => {
        const v = q(Vk, hx, hy, dx, dy);
        const best = Math.max(...ACTS.map((ac) => q(Vk, hx, hy, ac[0], ac[1])));
        hud.text(`${s}  ${v.toFixed(2)}`, 60 + (i % 2) * 190, 186 + Math.floor(i / 2) * 28, 16,
                 Math.abs(v - best) < 1e-9 ? C.green : C.muted, "left", Math.abs(v - best) < 1e-9 ? 700 : 500);
      });
      hud.text(`→ V = ${Math.max(...ACTS.map((ac) => q(Vk, hx, hy, ac[0], ac[1]))).toFixed(2)}`, 60, 248, 16, C.ink, "left", 700);
      hud.chart(940, 30, 310, 190, [{ data: delta, color: C.accent, upto: kf + 1 }],
                { title: "Largest change per sweep", n: delta.length, range: [0, Math.max(...delta)], digits: 1 });
      hud.alpha(1);
    }
    hud.caption("Sweep 1: only the cells next to the goal and the pit know anything.", fade(t, 5.2, 9.2, 0.4));
    hud.caption("Each sweep pushes value one step further out from the goal.", fade(t, 9.4, 15, 0.4));
    hud.caption("Changes shrink every sweep (γ < 1). Stop when they're tiny: that's convergence.", fade(t, 15.2, 21, 0.4));
    hud.caption("Policy for free: in each cell, step toward the highest-value neighbour.", fade(t, 22, 28, 0.4));
    hud.caption("Note the detour: no arrow leads past the pit.", fade(t, 28.2, 32.5, 0.4));
  };
}
