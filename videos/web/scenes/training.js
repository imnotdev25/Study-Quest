// How an LLM learns: predict the next token, measure surprise (cross-entropy), follow the gradient downhill on the
// loss landscape, and watch predictions sharpen as the loss falls.
import { C, THREE, cameraPath, clamp, fade, label, lerp, mat, smooth, softmax, story } from "../lib.js";

export const meta = {
  title: "How an LLM learns", path: "llm", duration: 48,
  desc: "Predict the next token, score the surprise, step downhill on the loss, repeat billions of times.",
  keywords: ["Next-Token Loss", "Where Losses Come From", "How to Actually Calculate a Loss", "Softmax Cross-Entropy",
             "Learning Rate, Curvature", "Momentum and Optimizer State", "AdamW", "The Pre-Training Loop", "One Neuron, Full Backpropagation",
             "Supervised Fine-Tuning"],
  poster: 24,
};

const CANDS = ["sat", "ran", "the", "blue", "slept"];
const CH = [
  { t: 0, title: "", text: "" },
  { t: 4.5, title: "Predict the next word", text: "Training text: “The cat ___”. The model outputs a probability for every candidate. The true next word is “sat”." },
  { t: 11, title: "Loss = surprise", text: "Cross-entropy loss = −log p(true word). The model gave “sat” only 0.15, so the loss is 1.9: very surprised." },
  { t: 17.5, title: "The loss landscape", text: "Every setting of the weights has a loss. Picture it as terrain: the ball is the model, height is how wrong it is." },
  { t: 23.5, title: "Follow the gradient downhill", text: "Backpropagation gives the slope at the ball. Step a little downhill: θ ← θ − η · ∇loss. Momentum keeps it rolling." },
  { t: 32, title: "Predictions sharpen", text: "As the loss falls, p(“sat”) rises. The same step happens for every token in every batch." },
  { t: 40, title: "Repeat billions of times", text: "Pre-training does this on trillions of tokens. Fine-tuning and RL reuse the same loop with different data and targets." },
];
const f = (x, y) => 0.16 * (x * x + 1.8 * y * y) + 0.45 * Math.sin(x * 1.2) * Math.cos(y * 1.3) + 0.6;

export function setup({ scene, camera, render, hud }) {
  const geo = new THREE.PlaneGeometry(11, 11, 64, 64);
  const pos = geo.attributes.position, cols = [];
  for (let k = 0; k < pos.count; k++) {
    const z = f(pos.getX(k), pos.getY(k)); pos.setZ(k, z);
    const c = new THREE.Color().setHSL(lerp(0.72, 0.02, clamp(z / 4)), 0.6, 0.72); cols.push(c.r, c.g, c.b);
  }
  geo.setAttribute("color", new THREE.Float32BufferAttribute(cols, 3));
  geo.computeVertexNormals();
  const surf = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.7, side: THREE.DoubleSide }));
  surf.rotation.x = -Math.PI / 2; surf.receiveShadow = true; surf.position.set(4, 0, 0); scene.add(surf);
  const wire = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: 0x475569, wireframe: true, transparent: true, opacity: 0.08 }));
  wire.rotation.x = -Math.PI / 2; wire.position.copy(surf.position); scene.add(wire);
  // descent path with momentum
  const path = [];
  let x = -4.2, y = 3.6, vx = 0, vy = 0;
  for (let k = 0; k < 120; k++) {
    const e = 1e-3, gx = (f(x + e, y) - f(x - e, y)) / (2 * e), gy = (f(x, y + e) - f(x, y - e)) / (2 * e);
    vx = 0.82 * vx - 0.08 * gx; vy = 0.82 * vy - 0.08 * gy; x += vx; y += vy;
    path.push({ p: new THREE.Vector3(x + 4, f(x, y) + 0.25, -y), loss: f(x, y), g: new THREE.Vector3(-gx, 0, gy) });
  }
  const LMAX = path[0].loss, LMIN = Math.min(...path.map((q) => q.loss));
  const ball = new THREE.Mesh(new THREE.SphereGeometry(0.25, 24, 16), mat(0xf59e0b, { emissive: 0xf59e0b, emissiveIntensity: 0.3 }));
  ball.castShadow = true; scene.add(ball);
  const trail = path.map(() => { const m = new THREE.Mesh(new THREE.SphereGeometry(0.05, 8, 6), mat(0x0f172a)); scene.add(m); return m; });
  const arrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 1.4, 0x16a34a, 0.4, 0.25); scene.add(arrow);
  const ml = label("model (current weights)", { size: 26, color: C.amber }); scene.add(ml);
  // prediction bars on the left
  const bars = CANDS.map((w, j) => {
    const b = new THREE.Mesh(new THREE.BoxGeometry(0.7, 1, 0.7), mat(j === 0 ? 0x16a34a : 0xa78bfa)); b.castShadow = true; scene.add(b);
    const l = label(w, { size: 30 }); l.position.set(-7 + j * 1.1, 0.25, 1.0); scene.add(l);
    return b;
  });
  const prompt = label("“The cat ___”", { size: 40 }); prompt.position.set(-4.8, 5.0, 0); scene.add(prompt);

  const KEYS = [[0, [2, 8, 15], [0, 1.5, 0]], [4.5, [-5, 4.5, 9], [-4.8, 2, 0]], [17.5, [10, 9, 11], [4, 1, 0]],
                [23.5, [6, 7, 9], [3, 1, 0]], [32, [0, 8, 13], [0, 1.5, 0]], [40, [1, 10, 15], [0, 1, 0]], [48, [2, 10.5, 15.5], [0, 1, 0]]];

  return function renderAt(t) {
    const prog = clamp((t - 24) / 14);                       // descent progress
    const k = Math.min(path.length - 1, Math.floor(prog * (path.length - 1)));
    const q = path[k];
    ball.position.copy(q.p); ball.visible = t > 17.5;
    trail.forEach((m, j) => { m.visible = t > 24 && j < k; m.position.copy(path[j].p); });
    arrow.visible = t > 24 && t < 39;
    arrow.position.copy(q.p).add(new THREE.Vector3(0, 0.3, 0));
    if (q.g.length() > 1e-4) arrow.setDirection(q.g.clone().normalize());
    ml.position.copy(q.p).setY(q.p.y + 0.9); ml.material.opacity = fade(t, 18, 26, 0.5);
    // predictions improve as loss falls
    const learned = clamp((LMAX - q.loss) / (LMAX - LMIN));
    const z = [lerp(0.2, 3.4, t > 24 ? learned : 0), 0.9, 0.6, -0.6, 0.4];
    const p = softmax(z);
    bars.forEach((b, j) => { const h = Math.max(0.03, p[j] * 6); b.scale.y = h; b.position.set(-7 + j * 1.1, 0.5 + h / 2, 0); });
    cameraPath(camera, KEYS, t);
    render();

    hud.clear();
    hud.title("How an LLM learns", "predict, measure the surprise, step downhill, repeat", fade(t, -1, 4.3, 0.7));
    const c = story(hud, CH, t);
    if (c.i >= 1) {
      hud.panel(40, 46, 300, 112);
      hud.text(`p(“sat”) = ${p[0].toFixed(2)}`, 60, 80, 18, C.green, "left", 750);
      hud.text(`loss = −log p = ${(-Math.log(p[0])).toFixed(2)}`, 60, 110, 18, C.ink, "left", 700);
      hud.text(c.i >= 4 ? `step ${k + 1}` : "before training", 60, 138, 14, C.muted);
    }
    if (c.i >= 4) {
      const L = path.map((x) => x.loss);
      hud.chart(950, 46, 300, 180, [{ data: L, color: C.accent, upto: k + 1 }],
        { title: "Loss over training steps", n: L.length, range: [Math.floor(LMIN * 10) / 10, Math.ceil(LMAX)], digits: 1 });
    }
  };
}
