// Policy gradients: a robot at a crossroads with four corridors. The policy is four probability bars; sampled
// actions that beat the baseline get pushed up, ones that don't get pushed down, until the policy prefers the
// corridor with the treasure.
import { C, THREE, cameraPath, clamp, coin, dynLabel, ease, fade, fmt, label, lerp, mat, robot, smooth, softmax, story } from "../lib.js";

export const meta = {
  title: "Policy gradients", path: "rl", duration: 50,
  desc: "A policy is a set of probabilities. Sample, score against a baseline, and shift probability toward what worked.",
  keywords: ["Policy Gradients", "REINFORCE", "Baselines and Advantage", "Actor-Critic", "GAE", "Leave-One-Out Baselines",
             "Critic-Free Baselines", "Entropy"],
  poster: 22,
};

const ACTS = [{ name: "→", d: [1, 0], r: 1 }, { name: "↑", d: [0, -1], r: 0.5 }, { name: "←", d: [-1, 0], r: 0 }, { name: "↓", d: [0, 1], r: 0 }];
const BASE = 0.4, LR = 1.1;
const CH = [
  { t: 0, title: "", text: "" },
  { t: 4.5, title: "A policy is probabilities", text: "Four corridors. The bars are π(a): the chance of picking each one. They start equal and always sum to 1." },
  { t: 11, title: "Sample and score", text: "Pick a corridor at random from the bars. → leads to a full chest (+1). That beat the baseline (0.4), so → goes up." },
  { t: 18, title: "Bad outcomes push down", text: "← and ↓ lead nowhere (0). Advantage = reward − baseline = −0.4, so their probabilities shrink." },
  { t: 25, title: "Why a baseline", text: "↑ pays 0.5: a little better than the baseline, so it grows a little. Without a baseline every reward would push up." },
  { t: 31.5, title: "Repeat many times", text: "Each update: θ ← θ + α · ∇log π(a) · A. Probability flows to actions that beat the baseline." },
  { t: 42, title: "The learned policy", text: "The robot now picks → most of the time. No model of the world, just trial, score and shift. PPO, GRPO and RLHF all build on this." },
];
// [time, action index]
const SAMPLES = [[12, 0], [15, 0], [19, 2], [21.5, 3], [25.5, 1], [28.3, 1],
  ...Array.from({ length: 16 }, (_, k) => [32 + k * 0.6, [0, 0, 1, 0, 2, 0, 0, 3, 0, 1, 0, 0, 0, 2, 0, 0][k]])];

function logitsAt(t) {
  const z = [0, 0, 0, 0];
  for (const [st, a] of SAMPLES) {
    const k = smooth((t - st - 1.6) / 0.7);
    if (k <= 0) continue;
    const p = softmax(z), A = ACTS[a].r - BASE;
    for (let j = 0; j < 4; j++) z[j] += k * LR * A * ((j === a ? 1 : 0) - p[j]);
  }
  return z;
}

export function setup({ scene, camera, render, hud }) {
  const ARM = 3.6;
  const floorM = mat(0xf1f5f9);
  const hub = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.14, 1.6), floorM); hub.position.y = 0.07; scene.add(hub);
  ACTS.forEach((A) => {
    const [dx, dz] = A.d;
    const c = new THREE.Mesh(new THREE.BoxGeometry(dx ? ARM : 1.4, 0.14, dz ? ARM : 1.4), floorM);
    c.position.set(dx * (0.8 + ARM / 2), 0.07, dz * (0.8 + ARM / 2)); c.receiveShadow = true; scene.add(c);
    const end = new THREE.Vector3(dx * (0.8 + ARM - 0.4), 0, dz * (0.8 + ARM - 0.4));
    A.end = end;
    if (A.r > 0) {
      const chest = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.55, 0.55), mat(0x92400e));
      chest.position.copy(end).setY(0.35); chest.castShadow = true; scene.add(chest);
      const lid = new THREE.Mesh(new THREE.BoxGeometry(0.82, 0.12, 0.57), mat(0xfbbf24, { metalness: 0.6 }));
      lid.position.copy(end).setY(0.68); scene.add(lid);
      const l = label(A.r === 1 ? "+1" : "+0.5", { size: 30, color: C.green }); l.position.copy(end).setY(1.4); scene.add(l);
    } else {
      const wall = new THREE.Mesh(new THREE.BoxGeometry(dx ? 0.2 : 1.4, 0.8, dz ? 0.2 : 1.4), mat(0xcbd5e1));
      wall.position.copy(end).add(new THREE.Vector3(dx * 0.4, 0.4, dz * 0.4)); scene.add(wall);
      const l = label("0", { size: 30, color: C.muted }); l.position.copy(end).setY(1.3); scene.add(l);
    }
  });
  // probability bars on a panel behind the hub
  const bars = ACTS.map((A, j) => {
    const b = new THREE.Mesh(new THREE.BoxGeometry(0.8, 1, 0.5), mat(j === 0 ? 0x7c3aed : 0xa78bfa));
    b.castShadow = true; scene.add(b);
    const l = label(A.name, { size: 36 }); scene.add(l);
    const v = dynLabel({ size: 26 }); scene.add(v);
    return { b, l, v, x: -2.7 + j * 1.8 };
  });
  const BZ = -6.2;
  const basePlane = new THREE.Mesh(new THREE.BoxGeometry(7.6, 0.02, 0.9), mat(0xf59e0b, { transparent: true, opacity: 0.35 }));
  scene.add(basePlane);
  const bot = robot(); scene.add(bot);
  const ball = new THREE.Mesh(new THREE.SphereGeometry(0.2, 18, 12), mat(0xf59e0b, { emissive: 0xf59e0b, emissiveIntensity: 0.4 }));
  scene.add(ball);
  const pop = dynLabel({ size: 34 }); scene.add(pop);
  const gold = coin(0.26); scene.add(gold);

  const KEYS = [[0, [7, 9, 10], [0, 0.5, -1.5]], [4.5, [0, 7.5, 7], [0, 1, -3]], [11, [5, 7, 7], [0, 1, -2]],
                [18, [-5, 7, 7], [0, 1, -2]], [25, [0, 8, 6], [0, 1, -2.5]], [31.5, [0, 8.5, 8.5], [0, 1, -2.5]],
                [42, [6, 8, 7], [0, 0.6, -1.5]], [50, [7, 8.5, 8], [0, 0.6, -1.5]]];

  return function renderAt(t) {
    const p = softmax(logitsAt(t));
    bars.forEach(({ b, l, v, x }, j) => {
      const h = Math.max(0.03, p[j] * 6);
      b.scale.y = h; b.position.set(x, h / 2, BZ);
      l.position.set(x, 0.3, BZ + 0.7);
      v.userData.set(p[j].toFixed(2)); v.position.set(x, h + 0.4, BZ);
    });
    basePlane.visible = t > 24 && t < 42;
    basePlane.position.set(0, 0.25, BZ);
    // current sample
    const cur = [...SAMPLES].reverse().find(([st]) => t >= st && t < st + (st >= 32 ? 0.6 : 2.6));
    let pos = new THREE.Vector3(0, 0, 0);
    if (cur) {
      const [st, a] = cur, dur = st >= 32 ? 0.6 : 2.6, k = (t - st) / dur;
      const A = ACTS[a];
      const go = smooth(clamp(k * 2.2 - 0.25)), back = smooth(clamp((k - 0.75) * 4));
      pos = new THREE.Vector3().lerpVectors(new THREE.Vector3(), A.end, go * (1 - back));
      ball.visible = k < 0.3;
      ball.position.set(bars[a].x, lerp(5.5, p[a] * 6 + 0.2, ease.out(k / 0.3)), BZ);
      pop.visible = gold.visible = k > 0.45 && k < 0.95 && dur > 1;
      const adv = A.r - BASE;
      pop.userData.set(`A = ${adv > 0 ? "+" : ""}${fmt(adv)}`, adv > 0 ? C.green : C.red);
      pop.position.copy(A.end).setY(2.1);
      gold.visible = gold.visible && A.r > 0;
      gold.position.copy(A.end).setY(1.0 + (k - 0.45) * 2); gold.rotation.z = t * 6;
      bot.rotation.y = Math.atan2(A.d[0], A.d[1]);
    } else { ball.visible = pop.visible = gold.visible = false; bot.rotation.y = t * 0.3; }
    if (t > 42) {   // final: greedy-ish walks to the chest
      const k = ((t - 42) / 2.4) % 1, go = smooth(clamp(k * 2)), back = smooth(clamp((k - 0.6) * 3));
      pos = new THREE.Vector3().lerpVectors(new THREE.Vector3(), ACTS[0].end, go * (1 - back));
      bot.rotation.y = Math.PI / 2;
    }
    bot.position.copy(pos).setY(0.15);
    cameraPath(camera, KEYS, t);
    render();

    hud.clear();
    hud.title("Policy gradients", "learn the probabilities directly from trial and error", fade(t, -1, 4.3, 0.7));
    const c = story(hud, CH, t);
    if (c.i >= 2 && c.i <= 5) {
      hud.panel(40, 46, 330, 128);
      hud.text("Update after each sample", 60, 76, 15, C.muted, "left", 700);
      hud.text("advantage A = reward − baseline", 60, 104, 16, C.ink, "left", 650);
      hud.text("A > 0: chosen action up", 60, 132, 15, C.green, "left", 650);
      hud.text("A < 0: chosen action down", 60, 156, 15, C.red, "left", 650);
    }
  };
}
