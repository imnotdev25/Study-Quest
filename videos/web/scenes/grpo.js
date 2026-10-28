// GRPO: one prompt, a group of 8 sampled answers, a verifier marks them, the group mean becomes the baseline,
// answers above it gain probability, the next round improves, and an all-correct group teaches nothing.
import { C, THREE, cameraPath, card, clamp, dynLabel, ease, fade, label, lerp, mat, smooth, softmax, story } from "../lib.js";

export const meta = {
  title: "GRPO: learn from the group", path: "rl", duration: 50,
  desc: "Sample a group of answers, score them, compare each with the group average, and push up the better ones.",
  keywords: ["GRPO", "Group Reward Standardization", "Verifiable Rewards", "DAPO", "Dr. GRPO", "Critic-Free Baselines",
             "RL with Verifiable Rewards", "Group Baselines", "GSPO", "Rejection Sampling"],
  poster: 22,
};

const ANS = ["51", "41", "51", "50", "17×3=51", "20", "54", "57"];
const R1 = [1, 0, 1, 0, 1, 0, 0, 0];
const R2 = [1, 1, 1, 0, 1, 1, 1, 0];
const stats = (r) => { const m = r.reduce((a, b) => a + b, 0) / r.length; const s = Math.sqrt(r.reduce((a, x) => a + (x - m) ** 2, 0) / r.length); return { m, s, adv: r.map((x) => (s > 0 ? (x - m) / s : 0)) }; };
const S1 = stats(R1), S2 = stats(R2);
const CH = [
  { t: 0, title: "", text: "" },
  { t: 4.5, title: "One prompt, a group of answers", text: "“What is 17 × 3?” The model samples G = 8 answers to the same question." },
  { t: 11, title: "A verifier scores them", text: "No human needed: the answer is checkable. 3 of 8 are correct (reward 1), 5 are wrong (0)." },
  { t: 17, title: "The group is the baseline", text: "Group mean = 0.375. Advantage = (reward − mean) / std. Correct answers rise above the plane, wrong ones sink." },
  { t: 24, title: "Update", text: "Above-average answers gain probability (bars), below-average ones lose it. No value network, unlike PPO." },
  { t: 31, title: "Next round", text: "The model now gets 6 of 8 right. The mean rises to 0.75, so the bar to beat rises with it." },
  { t: 38, title: "When everyone agrees", text: "All 8 correct: every advantage is 0, and the prompt teaches nothing. DAPO filters such prompts out." },
  { t: 44.5, title: "The recipe", text: "Sample a group, verify, subtract the group mean, update. Good RL data sits at the edge of what the model can solve." },
];

export function setup({ scene, camera, render, hud }) {
  const prompt = card(["17 × 3 = ?"], { w: 3.0, h: 1.2, border: "#7c3aed", size: 60 });
  prompt.position.set(0, 1.0, 5.0); prompt.rotation.x = -0.35; scene.add(prompt);
  const slot = (i) => new THREE.Vector3((i - 3.5) * 1.55, 0, 0);
  const cards = ANS.map((a) => { const c = card(a, { w: 1.35, h: 0.8, size: 48 }); scene.add(c); return c; });
  const stamps = ANS.map(() => { const l = dynLabel({ size: 34 }); scene.add(l); return l; });
  const advs = ANS.map(() => { const l = dynLabel({ size: 24 }); scene.add(l); return l; });
  const bars = ANS.map(() => { const b = new THREE.Mesh(new THREE.BoxGeometry(0.6, 1, 0.6), mat(0x60a5fa)); b.castShadow = true; scene.add(b); return b; });
  const plane = new THREE.Mesh(new THREE.BoxGeometry(13, 0.03, 1.6), mat(0xf59e0b, { transparent: true, opacity: 0.3 }));
  scene.add(plane);
  const pl = dynLabel({ size: 28, color: C.amber }); scene.add(pl);
  const zero = label("advantage 0 for all: no signal", { size: 30, color: C.red }); scene.add(zero);
  const P0 = ANS.map(() => 1 / 8);

  const KEYS = [[0, [0, 6, 13], [0, 1, 1.5]], [4.5, [0, 5, 11], [0, 1.2, 1.5]], [11, [0, 4.5, 9], [0, 1.5, 0]],
                [17, [-5, 5, 9], [0, 1.5, 0]], [24, [5, 5, 9], [0, 1, -0.5]], [31, [0, 5.5, 10], [0, 1.3, 0]],
                [38, [0, 5, 9.5], [0, 1.3, 0]], [50, [0, 6.5, 11], [0, 1.3, 0]]];

  return function renderAt(t) {
    const round2 = t >= 31, allOk = t >= 38;
    const R = allOk ? ANS.map(() => 1) : round2 ? R2 : R1;
    const St = allOk ? stats(R) : round2 ? S2 : S1;
    const t0 = allOk ? 38 : round2 ? 31 : 4.5;
    const verify = smooth((t - (round2 ? t0 + 1.5 : 11.3)) / 0.6);
    const lift = smooth((t - (round2 ? t0 + 3 : 17.5)) / 1.2);
    const upd = round2 ? 0 : smooth((t - 24.5) / 2.5);
    const logits = St.adv.map((a) => a * 0.9 * upd);
    const p = softmax(logits);
    cards.forEach((c, i) => {
      const k = ease.back((t - t0 - 0.6 - i * 0.25) / 0.5);
      const target = slot(i).setY(1.4 + St.adv[i] * 0.55 * lift);
      c.position.lerpVectors(prompt.position, target, clamp(k));
      c.scale.setScalar(Math.max(0.05, lerp(0.3, 1, clamp(k))));
      const ok = R[i] === 1;
      const border = verify > 0.5 ? (ok ? "#16a34a" : "#dc2626") : "#cbd5e1";
      if (c.userData.border !== border) { c.userData.redraw(border); c.userData.border = border; }
      stamps[i].visible = verify > 0.05;
      stamps[i].userData.set(ok ? "✓ 1" : "✗ 0", ok ? C.green : C.red);
      stamps[i].position.copy(target).add(new THREE.Vector3(0, 0.75, 0));
      stamps[i].material.opacity = verify;
      advs[i].visible = lift > 0.05;
      advs[i].userData.set(`A ${St.adv[i] >= 0 ? "+" : ""}${St.adv[i].toFixed(2)}`, St.adv[i] > 0 ? C.green : St.adv[i] < 0 ? C.red : C.muted);
      advs[i].position.copy(target).add(new THREE.Vector3(0, -0.65, 0));
      const h = Math.max(0.02, (round2 ? P0[i] : p[i]) * 9);
      bars[i].visible = t > 24 && t < 31;
      bars[i].scale.y = h; bars[i].position.set(slot(i).x, h / 2, -1.6);
      bars[i].material.color.set(upd > 0.1 ? (St.adv[i] > 0 ? 0x16a34a : 0xdc2626) : 0x60a5fa);
    });
    plane.visible = pl.visible = lift > 0.02;
    plane.position.set(0, 1.4, 0); plane.material.opacity = 0.3 * lift;
    pl.userData.set(`group mean ${St.m.toFixed(3)}`); pl.position.set(6.9, 1.85, 0);
    zero.visible = allOk && t > 41; zero.position.set(0, 3.3, 0);
    cameraPath(camera, KEYS, t);
    render();

    hud.clear();
    hud.title("GRPO: learn from the group", "no critic: compare each answer with its siblings", fade(t, -1, 4.3, 0.7));
    const c = story(hud, CH, t);
    if (c.i >= 3 && c.i <= 6) {
      hud.panel(40, 46, 300, 104);
      hud.text(`round ${round2 ? 2 : 1}${allOk ? " (all correct)" : ""}`, 60, 76, 15, C.muted, "left", 700);
      hud.text(`mean ${St.m.toFixed(3)}   std ${St.s.toFixed(3)}`, 60, 104, 17, C.ink, "left", 650);
      hud.text("A = (r − mean) / std", 60, 132, 16, C.accent, "left", 700);
    }
  };
}
