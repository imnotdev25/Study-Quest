// PPO: why big policy jumps are dangerous, the probability ratio, the clip band, the flat (zero-gradient) region
// for good and bad actions, and many small safe rounds instead of one leap.
import { C, THREE, cameraPath, clamp, dynLabel, fade, label, lerp, mat, smooth, softmax, story } from "../lib.js";

export const meta = {
  title: "PPO: small, safe steps", path: "rl", duration: 50,
  desc: "The probability ratio, the clip band, and why PPO stops pushing once a step has gone far enough.",
  keywords: ["PPO", "Trust Regions", "Importance Sampling", "Clip", "PPO for LLMs", "Token-Level PPO Ratios", "PPO KL Early Stopping",
             "PPO Epoch Reuse", "PPO’s Clipped Surrogate"],
  poster: 24,
};

const EPS = 0.2;
const CH = [
  { t: 0, title: "", text: "" },
  { t: 4.5, title: "The danger of big jumps", text: "Five answers. One update with a large step piles everything onto answer A, and the others collapse. One noisy batch just broke the policy." },
  { t: 12, title: "Rewind: measure the change", text: "PPO compares new and old probabilities for each sampled action: the ratio r = π_new(a) / π_old(a)." },
  { t: 18.5, title: "The clip band", text: "For a good action (A > 0) PPO only rewards raising r up to 1 + ε = 1.2. Past that line the objective goes flat." },
  { t: 26, title: "Flat means no push", text: "Inside the band the gradient pushes up. At the edge it drops to zero, so the update stops on its own." },
  { t: 32, title: "Bad actions too", text: "For a bad action (A < 0) the probability may fall to 1 − ε = 0.8 of its old value, and no further in one step." },
  { t: 38.5, title: "Many small rounds", text: "Each round the old policy is refreshed and the band moves. Several safe steps reach a good policy without the collapse." },
];
const BASEZ = [0.6, 0.1, 0.0, -0.2, -0.3];
const NAMES = ["A", "B", "C", "D", "E"];

export function setup({ scene, camera, render, hud }) {
  const X = (j) => (j - 2) * 1.6;
  const bars = NAMES.map((n, j) => {
    const b = new THREE.Mesh(new THREE.BoxGeometry(1.0, 1, 1.0), mat(j === 0 ? 0x7c3aed : j === 3 ? 0xdb2777 : 0xa78bfa));
    b.castShadow = true; scene.add(b);
    const old = new THREE.Mesh(new THREE.BoxGeometry(1.25, 0.05, 1.25), mat(0x0f172a));
    scene.add(old);
    const l = label(n, { size: 36 }); l.position.set(X(j), 0.3, 1.0); scene.add(l);
    const v = dynLabel({ size: 26 }); scene.add(v);
    return { b, old, v };
  });
  const band = new THREE.Mesh(new THREE.BoxGeometry(1.5, 1, 1.5), mat(0x16a34a, { transparent: true, opacity: 0.18 }));
  scene.add(band);
  const bandRed = new THREE.Mesh(new THREE.BoxGeometry(1.5, 1, 1.5), mat(0xdc2626, { transparent: true, opacity: 0.18 }));
  scene.add(bandRed);
  const ghost = new THREE.Mesh(new THREE.BoxGeometry(1.05, 1, 1.05), new THREE.MeshBasicMaterial({ color: 0xdc2626, wireframe: true, transparent: true, opacity: 0.5 }));
  scene.add(ghost);
  const stop = label("gradient = 0", { size: 28, color: C.red }); scene.add(stop);
  const S = 7;   // bar height scale

  // logits over time
  function state(t) {
    const z = BASEZ.slice();
    let oldz = BASEZ.slice(), focus = -1;
    if (t >= 6 && t < 12) z[0] += 3.2 * smooth((t - 6) / 2);                    // the big jump
    if (t >= 12 && t < 18.5) z[0] += 3.2 * (1 - smooth((t - 12) / 1)) + 0.22 * smooth((t - 14.5) / 1.5);
    if (t >= 18.5 && t < 32) { z[0] += lerp(0.22, 0.285, smooth((t - 19) / 4)); focus = 0; }
    if (t >= 32 && t < 38.5) { z[0] += 0.285; z[3] -= 0.13 * smooth((t - 33) / 3); focus = 3; }
    if (t >= 38.5) {
      const rounds = clamp((t - 39) / 2.2, 0, 4);
      const done = Math.floor(rounds), fr = smooth(rounds - done);
      const per = 0.3;
      oldz = BASEZ.slice(); oldz[0] += per * done; oldz[3] -= 0.25 * done;
      z.splice(0, 5, ...oldz); z[0] += per * fr; z[3] -= 0.25 * fr;
      focus = 0;
    }
    return { p: softmax(z), old: softmax(oldz), focus };
  }

  const KEYS = [[0, [6, 7, 11], [0, 1.5, 0]], [4.5, [0, 5.5, 10], [0, 1.8, 0]], [12, [-3, 4.5, 7.5], [-2.4, 1.5, 0]],
                [26, [-3, 4.5, 7.5], [-2.4, 1.5, 0]], [32, [2.5, 4.5, 7.5], [2, 1.2, 0]], [38.5, [0, 6, 10], [0, 1.6, 0]], [50, [1, 6.5, 10.5], [0, 1.6, 0]]];

  return function renderAt(t) {
    const { p, old, focus } = state(t);
    bars.forEach(({ b, old: o, v }, j) => {
      const h = Math.max(0.03, p[j] * S);
      b.scale.y = h; b.position.set(X(j), h / 2, 0);
      o.position.set(X(j), old[j] * S, 0); o.visible = t > 12;
      v.userData.set(p[j].toFixed(2)); v.position.set(X(j), h + 0.4, 0);
    });
    // clip band around the focused action
    const showGood = focus === 0 && t >= 18.5 && t < 32 || t >= 38.5, showBad = focus === 3 && t >= 32 && t < 38.5;
    band.visible = showGood; bandRed.visible = showBad;
    if (showGood) { const lo = old[0] * S, hi = old[0] * (1 + EPS) * S; band.scale.y = hi - lo; band.position.set(X(0), (lo + hi) / 2, 0); }
    if (showBad) { const lo = old[3] * (1 - EPS) * S, hi = old[3] * S; bandRed.scale.y = hi - lo; bandRed.position.set(X(3), (lo + hi) / 2, 0); }
    const r0 = p[0] / old[0];
    stop.visible = t >= 26 && t < 32 && r0 > 1 + EPS - 0.02;
    stop.position.set(X(0), old[0] * (1 + EPS) * S + 1.3, 0);
    ghost.visible = t > 41 && t < 50;
    const big = softmax(BASEZ.map((z, j) => z + (j === 0 ? 3.2 : 0)))[0] * S;
    ghost.scale.y = big; ghost.position.set(X(0), big / 2, 0);
    cameraPath(camera, KEYS, t);
    render();

    hud.clear();
    hud.title("PPO: small, safe steps", "improve the policy without wrecking it", fade(t, -1, 4.3, 0.7));
    const c = story(hud, CH, t);
    if (c.i >= 2 && c.i <= 5) {
      const j = c.i === 5 ? 3 : 0, r = p[j] / old[j], A = j === 0 ? 1 : -1;
      hud.panel(860, 46, 390, 290);
      hud.text(`action ${NAMES[j]}: ${A > 0 ? "good (A > 0)" : "bad (A < 0)"}`, 880, 76, 16, A > 0 ? C.green : C.red, "left", 750);
      hud.text(`r = ${p[j].toFixed(3)} / ${old[j].toFixed(3)} = ${r.toFixed(2)}`, 880, 104, 16, C.ink, "left", 650);
      // objective plot: min(rA, clip(r)A)
      const px = 900, py = 130, pw = 320, ph = 170, rx = (v) => px + ((v - 0.5) / 1.0) * pw;
      const obj = (v) => Math.min(v * A, clamp(v, 1 - EPS, 1 + EPS) * A);
      const yy = (v) => py + ph / 2 - (obj(v) - (A > 0 ? 1 : -1)) * (ph / 1.2);
      const g = hud.g;
      g.fillStyle = A > 0 ? "rgba(22,163,74,.12)" : "rgba(220,38,38,.12)";
      g.fillRect(rx(1 - EPS), py, rx(1 + EPS) - rx(1 - EPS), ph);
      g.strokeStyle = C.accent; g.lineWidth = 3; g.beginPath();
      for (let k = 0; k <= 60; k++) { const v = 0.5 + k / 60; k ? g.lineTo(rx(v), yy(v)) : g.moveTo(rx(v), yy(v)); }
      g.stroke();
      g.fillStyle = C.amber; g.beginPath(); g.arc(rx(clamp(r, 0.5, 1.5)), yy(clamp(r, 0.5, 1.5)), 8, 0, Math.PI * 2); g.fill();
      ["0.8", "1.0", "1.2"].forEach((s, k) => hud.text(s, rx([0.8, 1, 1.2][k]), py + ph + 18, 13, C.muted, "center"));
      hud.text("ratio r", px + pw, py + ph + 18, 13, C.muted, "right");
      hud.text("objective  min(r·A, clip(r)·A)", px, py - 6, 13, C.muted);
    }
  };
}
