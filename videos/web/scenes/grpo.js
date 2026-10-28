// GRPO: one prompt, a group of sampled answers, a verifier scores them, advantages are rewards minus the group
// mean (over the std), and the policy shifts probability toward the above-average answers.
import { C, THREE, cameraPath, card, fade, label, lerp, mat, smooth } from "../lib.js";

export const meta = {
  title: "GRPO in one group", path: "rl", duration: 34,
  desc: "Sample a group of answers, score them, compare each to the group average, and push up the better ones.",
  keywords: ["GRPO", "Group Reward Standardization", "Verifiable Rewards", "DAPO", "Dr. GRPO", "Critic-Free Baselines",
             "RL with Verifiable Rewards", "Group Baselines"],
  poster: 17,
};

const ANSWERS = [
  ["17 × 3 = 51", 1], ["17 × 3 = 41", 0], ["3 × 17 → 51", 1], ["about 50", 0],
  ["10×3 + 7×3 = 51", 1], ["17 + 3 = 20", 0], ["51", 1], ["5 1?", 0],
];
const G = ANSWERS.length;
const R = ANSWERS.map((a) => a[1]);
const MEAN = R.reduce((a, b) => a + b, 0) / G;
const STD = Math.sqrt(R.reduce((a, r) => a + (r - MEAN) ** 2, 0) / G);
const ADV = R.map((r) => (r - MEAN) / (STD + 1e-6));
const P0 = [0.16, 0.14, 0.1, 0.14, 0.08, 0.14, 0.1, 0.14];

export function setup({ scene, camera, render, hud }) {
  const prompt = card(["Prompt:", "What is 17 × 3?"], { w: 3.4, h: 1.5, border: "#7c3aed", size: 52 });
  prompt.position.set(-7.5, 1.6, 0); scene.add(prompt);
  const model = new THREE.Mesh(new THREE.IcosahedronGeometry(0.9, 1), mat(0x8b5cf6, { flatShading: true }));
  model.position.set(-4.4, 1.6, 0); model.castShadow = true; scene.add(model);
  const ml = label("policy π", { color: C.accent }); ml.position.set(-4.4, 3.0, 0); scene.add(ml);

  const cards = [], balls = [], pbars = [];
  ANSWERS.forEach(([txt], i) => {
    const c = card(txt, { w: 2.5, h: 0.9, size: 46 });
    scene.add(c); cards.push(c);
    const b = new THREE.Mesh(new THREE.SphereGeometry(0.22, 20, 14), mat(R[i] ? 0x16a34a : 0xdc2626));
    b.castShadow = true; scene.add(b); balls.push(b);
    const p = new THREE.Mesh(new THREE.BoxGeometry(0.5, 1, 0.5), mat(0x60a5fa));
    p.castShadow = true; scene.add(p); pbars.push(p);
  });
  const plane = new THREE.Mesh(new THREE.BoxGeometry(12, 0.02, 3.2), mat(0xf59e0b, { transparent: true, opacity: 0.35 }));
  scene.add(plane);
  const planeLabel = label(`group mean = ${MEAN.toFixed(2)}`, { color: C.amber }); scene.add(planeLabel);

  const slot = (i) => new THREE.Vector3(-1 + (i % 4) * 2.9, 0, i < 4 ? -1.4 : 1.4);
  const KEYS = [[0, [-2, 7, 15], [-1, 1, 0]], [4.5, [-2, 6, 13], [-1.5, 1.2, 0]], [13, [3.5, 6.5, 11], [3, 1.2, 0]],
                [24, [3.5, 9, 12], [3, 1, 0]], [34, [3, 10, 13.5], [3, 1, 0]]];

  return function renderAt(t) {
    model.rotation.y = t * 0.8; model.rotation.x = t * 0.3;
    const reward = smooth((t - 10.5) / 1.5), advT = smooth((t - 15) / 2), upd = smooth((t - 21) / 2.5);
    cards.forEach((c, i) => {
      const k = smooth((t - 5 - i * 0.45) / 1.2);
      const target = slot(i).setY(1.1 + ADV[i] * 0.55 * advT);
      c.position.lerpVectors(model.position, target, k);
      c.scale.setScalar(lerp(0.2, 1, k));
      c.visible = k > 0.001;
      const border = reward > 0.5 ? (R[i] ? "#16a34a" : "#dc2626") : "#cbd5e1";
      if (c.userData.border !== border) { c.userData.redraw(border); c.userData.border = border; }
      balls[i].visible = reward > 0.01;
      balls[i].scale.setScalar(reward);
      balls[i].position.copy(target).add(new THREE.Vector3(1.45, 0.25, 0));
      const p = lerp(P0[i], P0[i] * Math.exp(0.9 * ADV[i]) / P0.reduce((s, q, j) => s + q * Math.exp(0.9 * ADV[j]), 0), upd);
      const h = Math.max(0.02, p * 9);
      pbars[i].visible = t > 19.5;
      pbars[i].scale.y = h * smooth((t - 19.5) / 1);
      pbars[i].position.copy(slot(i)).add(new THREE.Vector3(1.45, 0, 0)).setY(0.01 + (h * smooth((t - 19.5) / 1)) / 2);
      pbars[i].material.color.set(upd > 0.05 ? (ADV[i] > 0 ? 0x16a34a : 0xdc2626) : 0x60a5fa);
    });
    plane.visible = advT > 0.01;
    plane.material.opacity = 0.35 * advT;
    plane.position.set(3.35, 1.1, 0);
    planeLabel.position.set(9.9, 1.5, 0);
    planeLabel.material.opacity = advT;
    cameraPath(camera, KEYS, t);
    render();

    hud.clear();
    hud.title("GRPO in one group", "no critic: compare each answer with its siblings", fade(t, -1, 4.2, 0.8));
    hud.caption("1. Sample a group of G = 8 answers to the same prompt.", fade(t, 5, 10.3, 0.4));
    hud.caption("2. A verifier scores each one: 1 if correct, 0 if not.", fade(t, 10.5, 14.8, 0.4));
    hud.caption("3. Advantage = (reward − group mean) / group std. Above the plane: better than average.", fade(t, 15, 20.8, 0.4));
    hud.caption("4. Update: raise the probability of positive-advantage answers, lower the rest.", fade(t, 21, 27, 0.4));
    const a = fade(t, 15.2, 27.2);
    if (a > 0) {
      hud.alpha(a);
      hud.panel(30, 30, 330, 300);
      hud.text("Group statistics", 50, 62, 19, C.ink, "left", 700);
      hud.text(`mean r = ${MEAN.toFixed(2)}   std = ${STD.toFixed(2)}`, 50, 92, 16, C.muted);
      ANSWERS.forEach(([txt], i) => {
        const y = 122 + i * 25;
        hud.text(txt, 50, y, 14, C.ink);
        hud.text(`r=${R[i]}  A=${ADV[i] >= 0 ? "+" : ""}${ADV[i].toFixed(2)}`, 340, y, 14, ADV[i] > 0 ? C.green : C.red, "right", 650);
      });
      hud.alpha(1);
    }
    const z = fade(t, 27.5, 40, 0.6);
    if (z > 0) {
      hud.alpha(z);
      hud.panel(250, 200, 780, 250);
      hud.text("When every answer gets the same reward…", 640, 244, 23, C.ink, "center", 700);
      hud.text("all 8 correct → mean 1, every advantage 0", 640, 288, 19, C.muted, "center");
      hud.text("all 8 wrong → mean 0, every advantage 0", 640, 318, 19, C.muted, "center");
      hud.text("…the group teaches nothing. That's why DAPO filters such prompts out", 640, 368, 18, C.accent, "center", 650);
      hud.text("and why good RL data sits at the edge of what the model can solve.", 640, 396, 18, C.accent, "center", 650);
      hud.alpha(1);
    }
  };
}
