// Value iteration, taught by zooming in on one Bellman backup first, then sweeping the whole grid until the
// values stop changing, then reading off the policy and walking it around the pit.
import { C, THREE, cameraPath, clamp, dynLabel, ease, fade, fmt, floorArrow, label, lerp, mat, robot, smooth, story } from "../lib.js";

export const meta = {
  title: "Value iteration", path: "rl", duration: 52,
  desc: "One Bellman backup up close, then sweep after sweep until values converge and the best policy appears.",
  keywords: ["Value Iteration", "Bellman", "Dynamic Programming", "Policy Iteration"],
  poster: 30,
};

const N = 5, GOAL = [4, 4], PIT = [2, 3], GAMMA = 0.9, STEP = -0.5;
const ACTS = [[1, 0], [-1, 0], [0, 1], [0, -1]];
const term = (x, y) => (x === GOAL[0] && y === GOAL[1] ? 10 : x === PIT[0] && y === PIT[1] ? -10 : null);
const inside = (x, y) => x >= 0 && y >= 0 && x < N && y < N;
function q(Vk, x, y, dx, dy) {
  let nx = x + dx, ny = y + dy;
  if (!inside(nx, ny)) { nx = x; ny = y; }
  const tv = term(nx, ny);
  return tv !== null ? tv : STEP + GAMMA * Vk[nx][ny];
}
const VS = [Array.from({ length: N }, () => Array(N).fill(0))];
for (let k = 0; k < 10; k++) {
  const V = VS.at(-1), nv = V.map((r) => r.slice());
  for (let x = 0; x < N; x++) for (let y = 0; y < N; y++) if (term(x, y) === null) nv[x][y] = Math.max(...ACTS.map(([a, b]) => q(V, x, y, a, b)));
  VS.push(nv);
}
const DELTA = VS.slice(1).map((V, k) => Math.max(...V.flatMap((r, x) => r.map((v, y) => Math.abs(v - VS[k][x][y])))));
const P = (x, y) => new THREE.Vector3((x - 2) * 1.5, 0, (2 - y) * 1.5);
const H = (v) => Math.max(0.03, Math.abs(v) * 0.3);
const ZOOM = [3, 4];   // the cell whose backup we zoom into (left of the goal)
const CH = [
  { t: 0, title: "", text: "" },
  { t: 4.5, title: "Start knowing nothing", text: "Every cell's value V is 0. Green pays +10, red is a pit (−10), every move costs −0.5." },
  { t: 10.5, title: "One backup, up close", text: "For this cell, try all four moves: reward + 0.9 × value of where you land. Moving right reaches the goal: +10." },
  { t: 17.5, title: "Keep the best", text: "The new V is the maximum: V(s) ← maxₐ [ r + γ·V(s′) ]. One step of lookahead." },
  { t: 23, title: "Sweep the whole grid", text: "Do that for every cell, then again. Each sweep, value spreads one cell further from the goal." },
  { t: 34, title: "Convergence", text: "Changes shrink every sweep because γ < 1. When the biggest change is tiny, stop: the values are final." },
  { t: 40, title: "Read off the policy", text: "In each cell, step toward the neighbour with the highest value. The robot detours around the pit." },
];
const SWEEP0 = 23.3, SWEEP_T = 1.05;

export function setup({ scene, camera, render, hud }) {
  const bars = [], nums = [];
  for (let x = 0; x < N; x++) for (let y = 0; y < N; y++) {
    const tv = term(x, y);
    const tile = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.14, 1.4), mat(tv === 10 ? 0x86efac : tv === -10 ? 0xfca5a5 : 0xf1f5f9));
    tile.position.copy(P(x, y)).setY(0.07); tile.receiveShadow = true; tile.userData = { x, y }; scene.add(tile);
    if (tv !== null) {
      const l = label(tv > 0 ? "+10" : "−10", { size: 34, color: tv > 0 ? C.green : C.red }); l.position.copy(P(x, y)).setY(0.9); scene.add(l);
      continue;
    }
    const b = new THREE.Mesh(new THREE.BoxGeometry(0.9, 1, 0.9), mat(0x8b5cf6, { transparent: true, opacity: 0.88 }));
    b.castShadow = true; b.userData = { x, y, tile }; scene.add(b); bars.push(b);
    const n = dynLabel({ size: 24 }); n.userData.xy = [x, y]; scene.add(n); nums.push(n);
  }
  // backup visual: four arrows from the zoomed cell with candidate values
  const cand = ACTS.map(() => { const a = floorArrow(0x2563eb, 0.9); scene.add(a); const l = dynLabel({ size: 28 }); scene.add(l); return { a, l }; });
  const scan = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.05, 7.6), mat(0xf59e0b, { emissive: 0xf59e0b, emissiveIntensity: 0.5, transparent: true, opacity: 0.6 }));
  scene.add(scan);
  const arrows = [];
  for (let x = 0; x < N; x++) for (let y = 0; y < N; y++) if (term(x, y) === null) {
    const a = floorArrow(0x0f172a, 0.7); a.userData.xy = [x, y]; scene.add(a); arrows.push(a);
  }
  const bot = robot(); scene.add(bot);
  const V = VS.at(-1);
  const best = (x, y) => ACTS.reduce((bi, ac, i) => (q(V, x, y, ...ac) > q(V, x, y, ...ACTS[bi]) ? i : bi), 0);
  const route = [[0, 0]];   // greedy route from the bottom-left corner
  while (route.length < 20) {
    const [x, y] = route.at(-1); if (term(x, y) !== null) break;
    const [dx, dy] = ACTS[best(x, y)]; route.push([x + dx, y + dy]);
  }

  const KEYS = [[0, [8, 10, 11], [0, 0.5, 0]], [4.5, [0, 9, 9.5], [0, 0.5, 0.3]], [10.5, [3.5, 5, 3.5], [1.5, 0.6, -3]],
                [17.5, [3.5, 5, 3.5], [1.5, 0.6, -3]], [23, [-6, 9.5, 9], [0, 0.5, 0]], [34, [6, 9, 9], [0, 0.8, 0]],
                [40, [0, 12, 6.5], [0, 0, 0.2]], [52, [-3, 12, 7], [0, 0, 0.2]]];

  return function renderAt(t) {
    // which value table is on screen
    let Vk = VS[0], hl = null, sweep = 0;
    if (t >= 17.5 && t < SWEEP0) {
      // only the zoomed cell (then its neighbour two steps out) has been backed up
      Vk = VS[0].map((r) => r.slice());
      const k = smooth((t - 18) / 1.5);
      Vk[ZOOM[0]][ZOOM[1]] = lerp(0, VS[1][ZOOM[0]][ZOOM[1]], k);
      hl = ZOOM;
    } else if (t >= SWEEP0) {
      const kf = clamp((t - SWEEP0) / SWEEP_T, 0, VS.length - 1);
      sweep = Math.floor(kf);
      const fr = smooth(kf - sweep), A = VS[sweep], B = VS[Math.min(VS.length - 1, sweep + 1)];
      Vk = A.map((r, x) => r.map((v, y) => lerp(v, B[x][y], fr)));
      scan.visible = kf < VS.length - 1 && t < 34;
      scan.position.set((fr * 5 - 2.5) * 1.5, 0.18, 0);
    }
    if (!(t >= SWEEP0 && t < 34)) scan.visible = false;
    bars.forEach((b) => {
      const { x, y, tile } = b.userData, v = Vk[x][y], h = H(v);
      b.scale.y = h; b.position.copy(P(x, y)).setY(0.14 + h / 2);
      b.material.color.set(v >= -0.01 ? 0x8b5cf6 : 0xef4444);
      const on = hl && x === hl[0] && y === hl[1];
      tile.material.color.set(on ? 0xfde68a : 0xf1f5f9);
    });
    nums.forEach((n) => {
      const [x, y] = n.userData.xy, v = Vk[x][y];
      n.userData.set(fmt(v, 1), v < -0.01 ? C.red : C.ink);
      n.position.copy(P(x, y)).setY(0.14 + H(v) + 0.35);
      n.visible = t > 4.5 && t < 40.5;
    });
    // candidate arrows for the zoomed backup
    const cz = fade(t, 11, 22.8, 0.5);
    const zoomCell = ZOOM;
    cand.forEach(({ a, l }, i) => {
      const [dx, dy] = ACTS[i], v = q(VS[0], zoomCell[0], zoomCell[1], dx, dy);
      const k = ease.back((t - 11.5 - i * 0.9) / 0.5) * cz;
      a.visible = l.visible = k > 0.02;
      a.scale.setScalar(Math.max(0.01, k));
      a.userData.point(dx, -dy);
      a.position.copy(P(...zoomCell)).add(new THREE.Vector3(dx * 0.6, 0.25, -dy * 0.6));
      const isBest = i === 0;
      a.children.forEach((m) => m.material.color.set(t > 17.8 && isBest ? 0x16a34a : t > 17.8 ? 0xcbd5e1 : 0x2563eb));
      l.userData.set(`${fmt(v)}`, t > 17.8 && isBest ? C.green : C.ink);
      l.position.copy(P(...zoomCell)).add(new THREE.Vector3(dx * 1.05, 0.75, -dy * 1.05));
      l.material.opacity = k;
    });
    // policy arrows and the robot walking it
    arrows.forEach((a) => {
      const [x, y] = a.userData.xy, k = ease.back((t - 40.3 - (x + y) * 0.05) / 0.5);
      a.visible = k > 0.02; a.scale.setScalar(Math.max(0.01, k));
      const [dx, dy] = ACTS[best(x, y)]; a.userData.point(dx, -dy);
      a.position.copy(P(x, y)).setY(0.14 + H(V[x][y]) + 0.06);
    });
    const rk = clamp((t - 43) / 0.65, 0, route.length - 1), ri = Math.floor(rk), rf = rk - ri;
    bot.visible = t > 42.5;
    bot.position.copy(P(...route[ri]).lerp(P(...route[Math.min(route.length - 1, ri + 1)]), smooth(rf)))
      .setY(0.14 + H(V[route[ri][0]]?.[route[ri][1]] ?? 0) + Math.sin(rf * Math.PI) * 0.25);
    cameraPath(camera, KEYS, t);
    render();

    hud.clear();
    hud.title("Value iteration", "plan by looking one step ahead, everywhere, until nothing changes", fade(t, -1, 4.3, 0.7));
    const c = story(hud, CH, t);
    if (c.i === 2 || c.i === 3) {
      hud.panel(40, 50, 380, 210);
      hud.text("Backup for the yellow cell", 60, 80, 16, C.ink, "left", 700);
      [["→ goal", "+10", 10], ["← left", "−0.5 + 0.9×0", -0.5], ["↑ wall", "−0.5 + 0.9×0", -0.5], ["↓ down", "−0.5 + 0.9×0", -0.5]].forEach(([a, e, v], i) => {
        const on = c.i === 3 && i === 0;
        hud.text(a, 60, 112 + i * 30, 15, on ? C.green : C.ink, "left", on ? 750 : 500);
        hud.text(e, 160, 112 + i * 30, 15, C.muted);
        hud.text(fmt(v), 400, 112 + i * 30, 15, on ? C.green : C.ink, "right", 700);
      });
      if (c.i === 3) hud.text("V ← max = 10.0", 60, 244, 18, C.accent, "left", 750);
    }
    if (c.i === 4 || c.i === 5) {
      hud.panel(40, 50, 230, 64);
      hud.text(`Sweep ${Math.min(VS.length - 1, sweep + 1)} of ${VS.length - 1}`, 60, 90, 20, C.ink, "left", 750);
      hud.chart(960, 50, 290, 180, [{ data: DELTA, color: C.accent, upto: clamp((t - SWEEP0) / SWEEP_T, 0, DELTA.length) }],
        { title: "Biggest change per sweep", n: DELTA.length, range: [0, Math.max(...DELTA)], digits: 0 });
    }
  };
}
