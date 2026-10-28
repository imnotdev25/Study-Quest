// Next-token prediction: the prompt goes through the model, out come logits, softmax (with temperature) turns
// them into probabilities, one token is sampled and appended, and the KV cache means only the new token is
// processed on the next step.
import { C, THREE, cameraPath, card, fade, label, lerp, mat, smooth, softmax, story } from "../lib.js";

export const meta = {
  title: "Predicting the next token", path: "llm", duration: 34,
  desc: "Logits, softmax and temperature, sampling one token, and why the KV cache makes the next step cheap.",
  keywords: ["Autoregressive Decoding and the KV Cache", "Next-Token Loss", "One Token, Prefill and Decode",
             "Softmax Cross-Entropy", "Speculative Decoding", "PagedAttention"],
  poster: 14,
};

const PROMPT = ["The", "capital", "of", "France", "is"];
const CANDS = ["Paris", "a", "the", "located", "Lyon", "known"];
const LOGITS = [6.2, 3.1, 2.9, 2.4, 1.6, 1.4];
const TEMP = (t) => (t < 15 ? 1 : t < 17.5 ? lerp(1, 0.3, smooth((t - 15) / 1.5)) : t < 20.5 ? lerp(0.3, 2.2, smooth((t - 17.5) / 1.5)) : lerp(2.2, 1, smooth((t - 20.5) / 1)));
const X = (i) => -6.5 + i * 1.8;

const CH = [
  { t: 0, title: "", text: "" },
  { t: 4.5, title: "Prefill", text: "Prefill: the whole prompt goes through the model in one parallel pass." },
  { t: 7.5, title: "Logits", text: "The last position outputs a logit (a raw score) for every token in the vocabulary." },
  { t: 11.0, title: "Softmax", text: "Softmax turns logits into probabilities that sum to 1." },
  { t: 14.5, title: "Temperature", text: "Low temperature sharpens the distribution, high temperature flattens it." },
  { t: 21.5, title: "Sample and append", text: "Sample one token and append it to the text." },
  { t: 26.5, title: "The KV cache", text: "Next step: only the new token runs through the model. Earlier keys and values come from the cache." },
];

export function setup({ scene, camera, render, hud }) {
  const toks = [...PROMPT, "Paris"].map((w, i) => {
    const c = card(w, { w: 1.6, h: 0.8, size: 46, border: i === PROMPT.length ? "#16a34a" : "#94a3b8" });
    c.position.set(X(i), 0.7, 3.4); scene.add(c); return c;
  });
  // the model: a stack of translucent layers
  const model = new THREE.Group();
  for (let l = 0; l < 6; l++) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(10.5, 0.32, 2.2), mat(0xc4b5fd, { transparent: true, opacity: 0.55 }));
    m.position.y = 1.7 + l * 0.42; m.castShadow = true; model.add(m);
  }
  model.position.set(-2, 0, 0.6); scene.add(model);
  const ml = label("transformer: N layers of attention + MLP", { color: C.accent }); ml.position.set(-2, 4.6, 0.6); scene.add(ml);
  // logit / probability bars on the right
  const bars = CANDS.map((w, i) => {
    const b = new THREE.Mesh(new THREE.BoxGeometry(0.7, 1, 0.7), mat(0x7c3aed)); b.castShadow = true; scene.add(b);
    const l = label(w, { size: 30, scale: 0.85 }); scene.add(l);
    return { b, l, x: 6.5 + i * 1.05 };
  });
  // KV cache slabs: one K (blue) and V (green) per processed token, stacked behind the model
  const kv = [...PROMPT, "Paris"].map((_, i) => {
    const g = new THREE.Group();
    const k = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.2, 0.5), mat(0x60a5fa)); k.position.y = 0.0;
    const v = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.2, 0.5), mat(0x4ade80)); v.position.y = 0.26;
    g.add(k, v); g.position.set(X(i), 0.25, -1.6); scene.add(g); return g;
  });
  const kvl = label("KV cache: keys and values kept from earlier tokens", { color: C.blue, size: 28 }); kvl.position.set(-2, 1.2, -2.4); scene.add(kvl);
  const pulse = new THREE.Mesh(new THREE.SphereGeometry(0.25, 20, 14), mat(0xf59e0b, { emissive: 0xf59e0b, emissiveIntensity: 0.8 }));
  scene.add(pulse);

  const KEYS = [[0, [1, 7, 16], [1, 2, 1]], [4.5, [-1, 6, 13], [-1, 2.3, 1]], [9, [6, 6, 12], [5, 2, 1]],
                [22, [6, 6, 12], [5, 2, 1]], [25, [-2, 7, 12], [-2, 1.5, 0]], [34, [-5, 8, 12], [-2, 1.5, 0]]];

  return function renderAt(t) {
    const T = TEMP(t), probs = softmax(LOGITS, T);
    const prefill = smooth((t - 5) / 2.5), logit = smooth((t - 8) / 1.5), sm = smooth((t - 11.5) / 1.5);
    const sample = smooth((t - 22) / 1.5), append = smooth((t - 23.5) / 1.5), decode = smooth((t - 27) / 2.5);
    toks.forEach((c, i) => {
      const isNew = i === PROMPT.length;
      c.visible = !isNew || sample > 0.01;
      if (isNew) {
        const from = new THREE.Vector3(bars[0].x, 2.5, 0.6), to = new THREE.Vector3(X(i), 0.7, 3.4);
        c.position.lerpVectors(from, to, append);
        c.scale.setScalar(lerp(0.6, 1, append));
      }
    });
    // prefill: all prompt tokens rise through the model together; decode: only the new token does
    const rising = t < 25 ? prefill : decode;
    pulse.visible = (t > 5 && t < 8) || (t > 27 && t < 29.5);
    pulse.position.set(t < 25 ? X(2) : X(PROMPT.length), lerp(0.9, 4.2, rising), 1.5);
    pulse.scale.set(t < 25 ? 18 : 3, 0.6, 1);
    kv.forEach((g, i) => {
      const k = i < PROMPT.length ? prefill : decode;
      g.visible = k > 0.01; g.scale.setScalar(k);
    });
    bars.forEach(({ b, l, x }, i) => {
      const v = lerp(LOGITS[i] / 6.2, probs[i] / Math.max(...softmax(LOGITS, 0.3)), sm);
      const h = Math.max(0.02, v * 3.6 * logit);
      b.visible = logit > 0.01;
      b.scale.y = h; b.position.set(x, h / 2, 0.6);
      b.material.color.set(i === 0 && sample > 0.5 ? 0x16a34a : sm > 0.5 ? 0x7c3aed : 0x94a3b8);
      l.position.set(x, -0.0 + 0.35, 1.6); l.material.opacity = logit;
    });
    cameraPath(camera, KEYS, t);
    render();

    hud.clear();
    hud.title("Predicting the next token", "one forward pass, one probability for every token in the vocabulary", fade(t, -1, 4.2, 0.8));
    story(hud, CH, t);
    const a = fade(t, 11.8, 26.5);
    if (a > 0) {
      hud.alpha(a);
      hud.panel(30, 30, 340, 260);
      hud.text(`p = softmax(z / T),  T = ${T.toFixed(1)}`, 50, 62, 18, C.ink, "left", 700);
      CANDS.forEach((w, i) => {
        const y = 86 + i * 30;
        hud.text(w, 50, y + 14, 15, C.ink);
        hud.bar(140, y, probs[i] * 190, 18, i === 0 ? C.accent : "#c4b5fd");
        hud.text((probs[i] * 100).toFixed(1) + "%", 350, y + 14, 14, C.muted, "right");
      });
      const H = -probs.reduce((s, p) => s + p * Math.log(p), 0);
      hud.text(`entropy ${H.toFixed(2)} nats`, 50, 276, 15, C.muted);
      hud.alpha(1);
    }
    const b = fade(t, 27.5, 40);
    if (b > 0) {
      hud.alpha(b);
      hud.panel(930, 30, 320, 130);
      hud.text("Cost per new token", 950, 62, 18, C.ink, "left", 700);
      hud.text("without cache: reprocess all T tokens", 950, 92, 15, C.muted);
      hud.text("with KV cache: 1 token + read the cache", 950, 120, 15, C.green, "left", 650);
      hud.alpha(1);
    }
  };
}
