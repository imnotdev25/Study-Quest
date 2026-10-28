// RLHF: preference pairs -> reward model -> RL against the reward model, held near the reference model by a KL
// leash, and what happens when you optimise the proxy too hard.
import { C, THREE, cameraPath, card, clamp, dynLabel, fade, label, lerp, mat, smooth, story } from "../lib.js";

export const meta = {
  title: "RLHF in four steps", path: "rl", duration: 50,
  desc: "Human preferences train a reward model, RL climbs it, a KL leash keeps the model sane, and pushing too hard backfires.",
  keywords: ["RLHF", "Reward Modeling", "KL Regularization", "Reward Overoptimization", "Best-of-N", "Reward Models",
             "Preference Optimization", "DPO", "Adaptive KL Control", "What Alignment Means"],
  poster: 20,
};

const CH = [
  { t: 0, title: "", text: "" },
  { t: 4.5, title: "1. People compare answers", text: "A rater sees two answers to the same prompt and picks the better one. Thousands of these pairs are collected." },
  { t: 12, title: "2. Train a reward model", text: "The reward model learns to give the preferred answer the higher score: P(A beats B) = σ(r(A) − r(B))." },
  { t: 19, title: "3. RL against the reward model", text: "The policy writes answers, the reward model scores them, and PPO or GRPO pushes toward higher scores." },
  { t: 27, title: "The KL leash", text: "A penalty β·KL(π ‖ π_ref) ties the policy to the original model. Without it, the policy drifts into odd text the reward model happens to like." },
  { t: 36, title: "4. Don't push too hard", text: "The reward model is only a proxy. Its score keeps rising, but true quality peaks and then falls. That's reward hacking (Goodhart's law)." },
  { t: 45, title: "In practice", text: "Stop near the peak, keep the leash, refresh the reward model, and read samples. DPO skips the reward model and learns from the pairs directly." },
];

export function setup({ scene, camera, render, hud }) {
  const A = card(["Answer A", "clear, correct"], { w: 2.6, h: 1.4, size: 40 }), B = card(["Answer B", "vague, rambling"], { w: 2.6, h: 1.4, size: 40 });
  A.position.set(-1.7, 1.3, 2.5); B.position.set(1.7, 1.3, 2.5); scene.add(A, B);
  const rater = new THREE.Group();
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.32, 20, 14), mat(0xfcd34d)); head.position.y = 1.55; rater.add(head);
  const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.3, 0.6, 8, 14), mat(0x2563eb)); body.position.y = 0.75; rater.add(body);
  rater.position.set(0, 0, 4.6); scene.add(rater);
  const pick = label("✓ preferred", { size: 30, color: C.green }); scene.add(pick);
  const rm = new THREE.Mesh(new THREE.BoxGeometry(2.4, 1.6, 1.6), mat(0x0d9488)); rm.position.set(0, 0.8, -1.5); rm.castShadow = true; scene.add(rm);
  const rml = label("reward model", { size: 32, color: C.teal }); rml.position.set(0, 2.1, -1.5); scene.add(rml);
  const sA = new THREE.Mesh(new THREE.BoxGeometry(0.5, 1, 0.5), mat(0x16a34a)), sB = new THREE.Mesh(new THREE.BoxGeometry(0.5, 1, 0.5), mat(0xdc2626));
  scene.add(sA, sB);
  const scA = dynLabel({ size: 26 }), scB = dynLabel({ size: 26 }); scene.add(scA, scB);
  // policy vs reference on a 2D "behaviour plane" (KL chapter)
  const ref = new THREE.Mesh(new THREE.SphereGeometry(0.35, 20, 14), mat(0x94a3b8)); scene.add(ref);
  const pol = new THREE.Mesh(new THREE.SphereGeometry(0.35, 20, 14), mat(0x7c3aed)); scene.add(pol);
  const free = new THREE.Mesh(new THREE.SphereGeometry(0.3, 20, 14), mat(0xdc2626, { transparent: true, opacity: 0.7 })); scene.add(free);
  const leash = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 1, 8), mat(0xf59e0b)); scene.add(leash);
  const refL = label("reference model", { size: 26, color: C.muted }); scene.add(refL);
  const polL = label("policy + KL", { size: 26, color: C.accent }); scene.add(polL);
  const freeL = label("no KL: drifts off", { size: 26, color: C.red }); scene.add(freeL);
  const KL0 = new THREE.Vector3(-6.5, 0.4, 0);

  const KEYS = [[0, [5, 6, 11], [0, 1.2, 1]], [4.5, [0, 4.5, 9.5], [0, 1.2, 2.5]], [12, [4, 5, 7], [0, 1, 0]],
                [19, [-1, 5.5, 8], [0, 1, 0]], [27, [-6, 8, 6], [-6, 0, -1]], [36, [0, 6, 10], [0, 1.5, 0]], [50, [1, 6.5, 10.5], [0, 1.5, 0]]];

  return function renderAt(t) {
    // chapter 1-2: cards, rater, scores
    const c12 = fade(t, 4.5, 26.5, 0.6);
    A.visible = B.visible = rater.visible = c12 > 0.01;
    const prefer = t > 8 && t < 19;
    A.position.y = 1.3 + (prefer ? 0.25 * smooth((t - 8) / 0.6) : 0);
    const border = prefer ? "#16a34a" : "#cbd5e1";
    if (A.userData.border !== border) { A.userData.redraw(border); A.userData.border = border; }
    pick.visible = prefer; pick.position.set(-1.7, 2.5, 2.5);
    rater.rotation.y = t > 7 ? -0.4 : Math.sin(t * 1.5) * 0.4;
    const train = smooth((t - 13) / 4);
    const hA = lerp(0.5, 2.4, train), hB = lerp(0.5, 0.7, train);
    sA.visible = sB.visible = scA.visible = scB.visible = t > 12 && t < 27;
    sA.scale.y = hA; sA.position.set(-0.6, 1.6 + hA / 2, -1.5); sB.scale.y = hB; sB.position.set(0.6, 1.6 + hB / 2, -1.5);
    scA.userData.set(`r(A) = ${(hA * 1.2).toFixed(1)}`, C.green); scA.position.set(-0.6, 1.9 + hA, -1.5);
    scB.userData.set(`r(B) = ${(hB * 1.2).toFixed(1)}`, C.red); scB.position.set(0.6, 1.9 + hB, -1.5);
    rm.material.emissive.set(t > 13 && t < 17 && Math.sin(t * 8) > 0 ? 0x14b8a6 : 0x000000);
    // KL chapter
    const kl = fade(t, 26.5, 36.3, 0.5);
    [ref, pol, free, leash, refL, polL, freeL].forEach((o) => (o.visible = kl > 0.01));
    if (kl > 0.01) {
      const k = clamp((t - 27.5) / 7);
      ref.position.copy(KL0);
      const drift = new THREE.Vector3(Math.cos(k * 5) * k * 4.5, 0, -k * 5.5 + Math.sin(k * 7) * k);
      free.position.copy(KL0).add(drift).setY(0.35);
      const tied = drift.clone().multiplyScalar(0.28);
      pol.position.copy(KL0).add(tied).setY(0.4);
      const d = pol.position.clone().sub(ref.position);
      leash.scale.y = Math.max(0.01, d.length()); leash.position.copy(ref.position).add(d.clone().multiplyScalar(0.5));
      leash.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.normalize());
      refL.position.copy(ref.position).setY(1.1); polL.position.copy(pol.position).setY(1.1); freeL.position.copy(free.position).setY(1.0);
    }
    cameraPath(camera, KEYS, t);
    render();

    hud.clear();
    hud.title("RLHF in four steps", "how human preferences become a training signal", fade(t, -1, 4.3, 0.7));
    const c = story(hud, CH, t);
    if (c.i === 3) {
      const k = clamp((t - 19.5) / 6);
      const T = 40, d = Array.from({ length: T }, (_, s) => 1 - Math.exp(-s / 10));
      hud.chart(860, 46, 390, 220, [{ data: d.map((v) => v * 3), color: C.teal, upto: k * T }],
        { title: "Reward-model score during RL", n: T, range: [0, 3], digits: 1 });
    }
    if (c.i === 5 || c.i === 6) {
      const k = clamp((t - 36.5) / 6), T = 60;
      const proxy = Array.from({ length: T }, (_, s) => 3 * (1 - Math.exp(-s / 14)) + s * 0.012);
      const tru = Array.from({ length: T }, (_, s) => { const x = s / 22; return 2.6 * x * Math.exp(1 - x); });
      hud.chart(250, 70, 780, 330, [{ data: proxy, color: C.teal, upto: k * T }, { data: tru, color: C.red, upto: k * T }],
        { title: "Optimisation pressure (KL from the start) →", n: T, range: [0, 4], digits: 0 });
      hud.text("reward-model score (proxy)", 1010, 130, 15, C.teal, "right", 700);
      hud.text("true quality", 1010, 300, 15, C.red, "right", 700);
      if (k > 0.45) hud.text("▲ stop here", 250 + 44 + (22 / (T - 1)) * 716, 140, 14, C.green, "center", 750);
    }
  };
}
