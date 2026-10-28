// Attention: "it" sends a query, every earlier token answers with a key, dot products become weights via softmax,
// and "it" collects a weighted mix of values. Then the full causal attention pattern for several heads.
import { C, THREE, cameraPath, card, fade, label, lerp, mat, smooth, softmax } from "../lib.js";

export const meta = {
  title: "Attention, step by step", path: "llm", duration: 34,
  desc: "Queries meet keys, softmax turns scores into weights, and each token gathers a weighted mix of values.",
  keywords: ["Scaled Dot-Product Attention", "Multi-Head Attention", "Backpropagation Through Attention",
             "Cheaper Attention", "Assemble a Pre-Norm Decoder Block"],
  poster: 15,
};

const WORDS = ["The", "cat", "sat", "on", "the", "mat", "because", "it", "was", "tired"];
const Q = 7;                                                          // "it"
const SCORES = [0.2, 3.1, 0.6, -0.4, 0.1, 1.2, 0.3, 0.9];             // it · k_j / √d for j ≤ 7
const WEIGHTS = softmax(SCORES);
const HEADS = [
  ["head 1: previous token", (i, j) => (j === i - 1 ? 4 : j === i ? 1 : 0)],
  ["head 2: who is 'it'?", (i, j) => (i === Q && j === 1 ? 4 : j === i ? 1.5 : j === 1 ? 0.8 : 0)],
  ["head 3: first token", (i, j) => (j === 0 ? 3 : 0)],
];
const X = (i) => (i - 4.5) * 1.55;

export function setup({ scene, camera, render, hud }) {
  const toks = WORDS.map((w, i) => {
    const c = card(w, { w: 1.4, h: 0.75, size: 44, border: i === Q ? "#7c3aed" : "#94a3b8" });
    c.position.set(X(i), 0.8, 3); scene.add(c); return c;
  });
  // q / k / v cubes above each token
  const mk = (col) => WORDS.map(() => { const m = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), mat(col)); m.castShadow = true; scene.add(m); return m; });
  const qs = mk(0x7c3aed), ks = mk(0x2563eb), vs = mk(0x16a34a);
  const lq = label("query", { color: C.accent, size: 28 }), lk = label("keys", { color: C.blue, size: 28 }), lv = label("values", { color: C.green, size: 28 });
  [lq, lk, lv].forEach((l) => scene.add(l));
  // weight bars and arcs from "it" to each earlier token
  const bars = WORDS.map(() => { const b = new THREE.Mesh(new THREE.BoxGeometry(0.5, 1, 0.5), mat(0x7c3aed)); b.castShadow = true; scene.add(b); return b; });
  const arcs = WORDS.slice(0, Q + 1).map((_, j) => {
    const a = new THREE.Vector3(X(Q), 1.3, 3), b = new THREE.Vector3(X(j), 1.3, 3);
    const mid = a.clone().add(b).multiplyScalar(0.5).setY(1.3 + Math.abs(Q - j) * 0.45 + 0.4);
    const curve = new THREE.QuadraticBezierCurve3(a, mid, b);
    const r = 0.02 + 0.12 * WEIGHTS[j] / Math.max(...WEIGHTS);   // thickness = attention weight
    const m = new THREE.Mesh(new THREE.TubeGeometry(curve, 40, r, 8), mat(0x7c3aed, { transparent: true, opacity: 0.85 }));
    scene.add(m); return { m, curve, j };
  });
  // flying value packets
  const packets = WORDS.slice(0, Q + 1).map(() => { const p = new THREE.Mesh(new THREE.SphereGeometry(0.16, 16, 12), mat(0x16a34a)); scene.add(p); return p; });
  // attention matrix (heads)
  const cells = [];
  for (let i = 0; i < WORDS.length; i++) for (let j = 0; j < WORDS.length; j++) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(0.5, 1, 0.5), mat(0x7c3aed));
    m.position.set((j - 4.5) * 0.6, 0, -2 - i * 0.6); m.castShadow = true; m.userData = { i, j }; scene.add(m); cells.push(m);
  }
  const headLabels = HEADS.map(([name]) => { const l = label(name, { color: C.accent }); l.position.set(0, 3.2, -4.5); scene.add(l); return l; });
  const maskLabel = label("future tokens masked (causal)", { color: C.muted, size: 28 }); maskLabel.position.set(4.2, 1.2, -1.6); scene.add(maskLabel);

  const KEYS = [[0, [0, 7, 15], [0, 1, 1]], [4.5, [1, 5.5, 12], [0, 1.4, 2.5]], [11, [3, 6, 11], [1, 2, 2]],
                [20, [3, 6, 11], [1, 2, 2]], [23, [0, 11, 7], [0, 0, -4.5]], [34, [-3, 10, 6], [0, 0, -4.5]]];

  return function renderAt(t) {
    const qkv = smooth((t - 5) / 1.5), score = smooth((t - 9) / 1.5), sm = smooth((t - 13) / 1.5), mix = smooth((t - 17) / 2);
    const mat0 = smooth((t - 22) / 1.5);
    WORDS.forEach((_, i) => {
      const show = qkv * (1 - mat0);
      qs[i].visible = ks[i].visible = vs[i].visible = show > 0.01;
      qs[i].position.set(X(i) - 0.4, 1.5, 3); ks[i].position.set(X(i), 1.5, 3); vs[i].position.set(X(i) + 0.4, 1.5, 3);
      [qs[i], ks[i], vs[i]].forEach((m) => m.scale.setScalar(show));
      qs[i].material.emissive.set(i === Q ? 0x7c3aed : 0x000000); qs[i].material.emissiveIntensity = 0.4;
      toks[i].visible = mat0 < 0.99;
      const w = i <= Q ? lerp(Math.max(0, SCORES[i]) / 3.1, WEIGHTS[i] / Math.max(...WEIGHTS), sm) : 0;
      const h = Math.max(0.01, w * 2.2 * score * (1 - mat0));
      bars[i].visible = score > 0.01 && i <= Q && mat0 < 0.99;
      bars[i].scale.y = h; bars[i].position.set(X(i), 0.01 + h / 2, 4.6);
    });
    lq.position.set(X(Q) - 0.4, 2.1, 3); lk.position.set(X(2), 2.1, 3); lv.position.set(X(4) + 0.4, 2.1, 3);
    [lq, lk, lv].forEach((l) => (l.material.opacity = qkv * (t < 12 ? 1 : 0)));
    arcs.forEach(({ m, j }) => {
      const w = WEIGHTS[j] / Math.max(...WEIGHTS);
      m.visible = sm > 0.01 && mat0 < 0.99;
      m.material.opacity = 0.15 + 0.75 * w * sm;
    });
    packets.forEach((p, j) => {
      p.visible = mix > 0.01 && mix < 0.99 && mat0 < 0.01;
      p.position.copy(arcs[j].curve.getPoint(1 - mix));
      p.scale.setScalar(0.4 + 2 * WEIGHTS[j]);
    });
    // heads
    const hi = Math.min(HEADS.length - 1, Math.floor(Math.max(0, t - 23) / 3.6));
    const hk = smooth((t - 23 - hi * 3.6) / 0.8);
    const prev = Math.max(0, hi - 1);
    cells.forEach((m) => {
      const { i, j } = m.userData;
      const masked = j > i;
      const row = (h) => softmax(WORDS.map((_, jj) => (jj <= i ? HEADS[h][1](i, jj) : -1e9)));
      const w = masked ? 0 : lerp(row(prev)[j], row(hi)[j], hk);
      const h = masked ? 0.04 : 0.06 + w * 2.0;
      m.visible = mat0 > 0.01;
      m.scale.set(1, Math.max(0.01, h * mat0), 1);
      m.position.y = (h * mat0) / 2;
      m.material.color.set(masked ? 0xe2e8f0 : 0x7c3aed);
      m.material.transparent = true; m.material.opacity = masked ? 0.6 : 0.35 + 0.65 * Math.min(1, w * 1.5);
    });
    maskLabel.visible = mat0 > 0.01;
    maskLabel.material.opacity = mat0;
    headLabels.forEach((l, h) => { l.visible = mat0 > 0.01 && h === hi; l.material.opacity = mat0 * hk; });
    cameraPath(camera, KEYS, t);
    render();

    hud.clear();
    hud.title("Attention, step by step", "how “it” finds out what it refers to", fade(t, -1, 4.2, 0.8));
    hud.caption("1. Every token makes a query (what I need), a key (what I have) and a value (what I pass on).", fade(t, 5, 8.8, 0.4));
    hud.caption("2. Score = query of “it” · key of each earlier token, divided by √d.", fade(t, 9, 12.8, 0.4));
    hud.caption("3. Softmax turns scores into weights that sum to 1. “cat” wins.", fade(t, 13, 16.8, 0.4));
    hud.caption("4. “it” takes the weighted sum of values: its new vector now knows about the cat.", fade(t, 17, 21.8, 0.4));
    hud.caption("Every token does this at once: rows = queries, columns = keys. Each head learns its own pattern.", fade(t, 22.5, 34.5, 0.4));
    const a = fade(t, 9.2, 21.8);
    if (a > 0) {
      hud.alpha(a);
      hud.panel(30, 30, 360, 300);
      hud.text(t < 13 ? "scores  q·k / √d" : "weights  softmax(scores)", 50, 62, 18, C.ink, "left", 700);
      WORDS.slice(0, Q + 1).forEach((w, j) => {
        const v = t < 13 ? SCORES[j] : WEIGHTS[j];
        const y = 88 + j * 29;
        hud.text(w, 50, y + 14, 15, C.ink);
        hud.bar(140, y, (t < 13 ? Math.max(0, v) / 3.1 : v) * 170, 18, j === 1 ? C.accent : "#c4b5fd");
        hud.text(t < 13 ? v.toFixed(1) : v.toFixed(2), 370, y + 14, 14, C.muted, "right");
      });
      hud.alpha(1);
    }
  };
}
