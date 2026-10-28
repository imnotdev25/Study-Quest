// Inside one decoder block: token vectors flow up through RMSNorm, attention (tokens exchange information),
// a residual add, RMSNorm, the MLP (each token thinks alone), another residual add, then N blocks stacked.
import { C, THREE, cameraPath, card, clamp, fade, label, lerp, mat, smooth, story } from "../lib.js";

export const meta = {
  title: "Inside a transformer block", path: "llm", duration: 50,
  desc: "Follow five token vectors through norm, attention, residual, MLP and residual, then stack the block N times.",
  keywords: ["Assemble a Pre-Norm Decoder Block", "LayerNorm, Residual Paths", "The LLaMA Recipe", "Multi-Head Attention",
             "Mixture of Experts", "GPT-1 to GPT-3", "The Architecture Landscape"],
  poster: 20,
};

const WORDS = ["The", "cat", "sat", "on", "it"];
const D = 8;    // vector length shown as a column of cubes
const LEVELS = { embed: 0.6, norm1: 2.4, attn: 4.4, add1: 6.2, norm2: 7.8, mlp: 9.8, add2: 11.6 };
const CH = [
  { t: 0, title: "", text: "" },
  { t: 4.5, title: "Tokens become vectors", text: "Each token is now a column of numbers (its embedding). Colour = value. Five tokens, five columns." },
  { t: 10, title: "RMSNorm", text: "Rescale each vector to a steady size so the next step behaves the same at every depth." },
  { t: 15, title: "Attention: tokens talk", text: "Each token gathers information from earlier ones (arcs). “it” pulls mostly from “cat”. This is the only place tokens mix." },
  { t: 22, title: "Residual add", text: "The attention output is added to the original vector, not swapped in. The input always has a straight path upward." },
  { t: 27.5, title: "MLP: each token thinks alone", text: "Each vector expands to 4× the size, goes through a nonlinearity, and shrinks back. Most of the model's parameters live here." },
  { t: 34, title: "Residual add again", text: "Add the MLP output to the stream. One block = norm → attention → add → norm → MLP → add." },
  { t: 39.5, title: "Stack it N times", text: "Real models repeat this block 30–100 times. Early blocks learn grammar-like patterns, later ones more abstract features." },
  { t: 45, title: "Then predict", text: "After the last block, a final norm and the unembedding turn the last column into a score for every word in the vocabulary." },
];

function vecs(stage, seed = 0) {
  // deterministic pseudo-vectors per token per stage
  return WORDS.map((_, i) => Array.from({ length: D }, (_, j) => Math.sin(i * 1.7 + j * 0.9 + stage * 1.3 + seed) * (stage === 0 ? 1.4 : 1)));
}
const V = { embed: vecs(0), norm1: vecs(0).map((v) => { const r = Math.sqrt(v.reduce((a, x) => a + x * x, 0) / D); return v.map((x) => x / r); }),
            attn: vecs(2), add1: vecs(3), norm2: vecs(4), mlp: vecs(5), add2: vecs(6) };
const colorOf = (x) => new THREE.Color().setHSL(x > 0 ? 0.72 : 0.0, 0.7, lerp(0.92, 0.5, clamp(Math.abs(x) / 1.5)));

export function setup({ scene, camera, render, hud }) {
  const X = (i) => (i - 2) * 1.9;
  // block frame: translucent slabs for each sublayer
  const slab = (y, h, color, name) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(10.5, h, 2.2), mat(color, { transparent: true, opacity: 0.22 }));
    m.position.set(0, y, 0); scene.add(m);
    const l = label(name, { size: 28, color: C.accent }); l.position.set(-6.6, y, 0); scene.add(l);
    return m;
  };
  const S = { norm1: slab(LEVELS.norm1, 0.5, 0x94a3b8, "RMSNorm"), attn: slab(LEVELS.attn, 1.3, 0x7c3aed, "Attention"),
              add1: slab(LEVELS.add1, 0.4, 0xf59e0b, "+ residual"), norm2: slab(LEVELS.norm2, 0.5, 0x94a3b8, "RMSNorm"),
              mlp: slab(LEVELS.mlp, 1.3, 0x0d9488, "MLP"), add2: slab(LEVELS.add2, 0.4, 0xf59e0b, "+ residual") };
  WORDS.forEach((w, i) => { const c = card(w, { w: 1.3, h: 0.6, size: 44 }); c.position.set(X(i), 0.3, 1.6); c.rotation.x = -0.5; scene.add(c); });
  // the moving columns: D cubes per token
  const cols = WORDS.map((_, i) => Array.from({ length: D }, (_, j) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.16, 0.42), mat(0xffffff)); m.castShadow = true; scene.add(m); return m;
  }));
  // residual bypass lines on the left of each column
  const bypass = WORDS.map((_, i) => {
    const m = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, LEVELS.add2 - LEVELS.embed, 8), mat(0xf59e0b, { transparent: true, opacity: 0.0 }));
    m.position.set(X(i) - 0.55, (LEVELS.add2 + LEVELS.embed) / 2 + 0.5, 0); scene.add(m); return m;
  });
  // attention arcs from "it" to earlier tokens
  const W = [0.05, 0.62, 0.12, 0.06, 0.15];
  const arcs = W.map((w, j) => {
    const a = new THREE.Vector3(X(4), LEVELS.attn + 0.5, 0.6), b = new THREE.Vector3(X(j), LEVELS.attn + 0.5, 0.6);
    const mid = a.clone().add(b).multiplyScalar(0.5).setY(LEVELS.attn + 0.9 + Math.abs(4 - j) * 0.35);
    const m = new THREE.Mesh(new THREE.TubeGeometry(new THREE.QuadraticBezierCurve3(a, mid, b), 40, 0.02 + 0.12 * w, 8),
      mat(0x7c3aed, { transparent: true, opacity: 0.8 }));
    scene.add(m); return m;
  });
  // MLP expansion: a wide translucent box per token
  const wide = WORDS.map((_, i) => { const m = new THREE.Mesh(new THREE.BoxGeometry(1.5, 1.2, 1.0), mat(0x14b8a6, { transparent: true, opacity: 0.35 })); m.position.set(X(i), LEVELS.mlp, 0); scene.add(m); return m; });
  // stack of N blocks (chapter 7)
  const stack = Array.from({ length: 6 }, (_, k) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(10.5, 1.6, 2.2), mat(k % 2 ? 0xddd6fe : 0xc7d2fe, { transparent: true, opacity: 0.55 }));
    m.position.set(0, 1 + k * 1.9, -5); scene.add(m); return m;
  });

  // which stage the columns are at, as a function of time
  const STAGES = [["embed", 4.5], ["norm1", 10.5], ["attn", 16], ["add1", 22.5], ["norm2", 27.5], ["mlp", 29], ["add2", 34.5]];
  function stageAt(t) {
    let i = 0;
    while (i + 1 < STAGES.length && t >= STAGES[i + 1][1]) i++;
    const next = STAGES[Math.min(STAGES.length - 1, i + 1)];
    return { a: STAGES[i][0], b: next[0], k: i + 1 < STAGES.length ? smooth((t - next[1] + 1.4) / 1.4) : 0 };
  }

  const KEYS = [[0, [9, 6, 13], [0, 3, 0]], [4.5, [0, 2.5, 9], [0, 1.2, 0]], [10, [0, 4, 9.5], [0, 2.6, 0]],
                [15, [3, 6.5, 8], [0, 4.6, 0]], [22, [-4, 7, 9], [0, 6, 0]], [27.5, [3, 9.5, 9.5], [0, 9.2, 0]],
                [34, [0, 10.5, 10], [0, 10.5, 0]], [39.5, [14, 9, 14], [0, 5, -3]], [50, [15, 10, 15], [0, 5, -3]]];

  return function renderAt(t) {
    const { a, b, k } = stageAt(t);
    const ya = LEVELS[a], yb = LEVELS[b];
    cols.forEach((col, i) => col.forEach((m, j) => {
      const va = V[a][i][j], vb = V[b][i][j];
      m.material.color.copy(colorOf(lerp(va, vb, k)));
      const y = lerp(ya, yb, k) + (j - D / 2) * 0.17 + 0.6;
      m.position.set(X(i) + (a === "mlp" || b === "mlp" ? Math.sin(j) * 0.25 * Math.min(1, b === "mlp" ? k : 1 - k) : 0), y, 0);
      m.visible = t > 4.5 && t < 39.5;
    }));
    bypass.forEach((m) => { m.material.opacity = 0.7 * (fade(t, 22, 39.5, 0.6)); m.visible = m.material.opacity > 0.01; });
    arcs.forEach((m, j) => { m.visible = t > 15.5 && t < 22.5; m.material.opacity = 0.15 + 0.75 * W[j] / 0.62 * fade(t, 15.5, 22.5, 0.6); });
    wide.forEach((m) => { m.visible = t > 28 && t < 34.5; m.scale.x = 1 + 1.2 * fade(t, 28.5, 34, 1); });
    Object.values(S).forEach((m) => (m.visible = t < 39.5));
    stack.forEach((m, k2) => { const s = smooth((t - 39.8 - k2 * 0.4) / 0.6); m.visible = s > 0.01; m.scale.setScalar(Math.max(0.01, s)); });
    cameraPath(camera, KEYS, t);
    render();

    hud.clear();
    hud.title("Inside a transformer block", "the same few steps, repeated dozens of times", fade(t, -1, 4.3, 0.7));
    const c = story(hud, CH, t);
    if (c.i >= 1 && c.i <= 6) {
      hud.panel(40, 46, 280, 64);
      hud.text("x ← x + Attn(Norm(x))", 58, 74, 16, c.i >= 2 && c.i <= 4 ? C.accent : C.muted, "left", 700);
      hud.text("x ← x + MLP(Norm(x))", 58, 98, 16, c.i >= 5 ? C.teal : C.muted, "left", 700);
    }
    if (c.i === 8) {
      hud.panel(860, 46, 380, 120);
      hud.text("logits = Unembed(Norm(x_last))", 880, 80, 16, C.ink, "left", 700);
      hud.text("one score per vocabulary token", 880, 108, 15, C.muted);
      hud.text("softmax → next-token probabilities", 880, 136, 15, C.accent, "left", 650);
    }
  };
}
