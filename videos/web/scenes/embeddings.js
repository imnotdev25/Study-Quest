// Tokens become vectors: text splits into tokens, each token id looks up a row of the embedding table, and the
// vectors land in a space where similar meanings sit close together (and directions carry meaning).
import { C, THREE, cameraPath, card, fade, hash, label, lerp, mat, smooth, story } from "../lib.js";

export const meta = {
  title: "From tokens to embeddings", path: "llm", duration: 32,
  desc: "Text is split into tokens, each id picks a row of the embedding table, and meaning becomes geometry.",
  keywords: ["Embeddings", "Tokenization", "Weight Tying", "Core Intuitions: Dot Products", "Notation, Shapes"],
  poster: 20,
};

const SENT = [["The", 464], [" king", 5822], [" and", 290], [" queen", 16599], [" ate", 15063], [" apples", 22514]];
// [word, cluster, x, y, z]
const POINTS = [
  ["king", "royalty", 3.2, 2.6, -1.0], ["queen", "royalty", 3.4, 2.4, 0.9], ["prince", "royalty", 2.6, 2.9, -0.2],
  ["man", "people", 0.6, 1.2, -1.0], ["woman", "people", 0.8, 1.0, 0.9], ["child", "people", 0.2, 1.5, 0.0],
  ["apple", "food", -3.0, 1.0, 2.2], ["apples", "food", -3.2, 1.2, 2.5], ["bread", "food", -3.6, 0.8, 1.6],
  ["ate", "verbs", -2.4, 3.0, -2.2], ["eat", "verbs", -2.7, 3.2, -2.5], ["ran", "verbs", -1.8, 3.4, -2.8],
  ["Paris", "places", 1.5, 3.8, 2.8], ["Rome", "places", 2.0, 4.1, 3.1], ["Tokyo", "places", 1.1, 4.0, 3.4],
];
const COL = { royalty: 0x7c3aed, people: 0x2563eb, food: 0x16a34a, verbs: 0xd97706, places: 0xdb2777 };
const OFF = new THREE.Vector3(4, 0, 0);

const CH = [
  { t: 0, title: "", text: "" },
  { t: 4.5, title: "Split into tokens", text: "A tokenizer splits text into pieces it knows. Each piece has an integer id." },
  { t: 11.5, title: "Look up a row", text: "Each id selects one row of a learned table: a vector of d numbers." },
  { t: 17.5, title: "Meaning becomes distance", text: "Training arranges those vectors so similar meanings end up close together." },
  { t: 24.5, title: "Directions carry meaning", text: "Directions carry meaning too: king → queen points like man → woman." },
];

export function setup({ scene, camera, render, hud }) {
  // sentence tokens
  const toks = SENT.map(([w, id], i) => {
    const c = card([w.trim(), `id ${id}`], { w: 1.6, h: 1.0, border: "#94a3b8", size: 40 });
    c.userData.home = new THREE.Vector3(-6 + i * 1.75, 1.0, 5);
    scene.add(c); return c;
  });
  // embedding table: a tall stack of thin rows
  const table = new THREE.Group();
  for (let r = 0; r < 28; r++) {
    const row = new THREE.Mesh(new THREE.BoxGeometry(3.4, 0.12, 0.6), mat(r % 2 ? 0xe0e7ff : 0xede9fe));
    row.position.y = 0.3 + r * 0.16; row.castShadow = true; table.add(row);
  }
  table.position.set(-9.5, 0, -1); scene.add(table);
  const tl = label("embedding table  (vocab × d)", { color: C.accent }); tl.position.set(-9.5, 5.3, -1); scene.add(tl);
  // points in embedding space
  const pts = POINTS.map(([w, cl, x, y, z]) => {
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.2, 20, 14), mat(COL[cl]));
    m.position.set(x, y, z).add(OFF); m.castShadow = true; scene.add(m);
    const l = label(w, { size: 30, color: "#" + new THREE.Color(COL[cl]).getHexString(), scale: 0.8 });
    l.position.copy(m.position).add(new THREE.Vector3(0, 0.45, 0)); scene.add(l);
    return { m, l, w };
  });
  const axes = new THREE.Group();
  for (const [dir, col] of [[[1, 0, 0], 0xef4444], [[0, 1, 0], 0x16a34a], [[0, 0, 1], 0x2563eb]]) {
    const a = new THREE.ArrowHelper(new THREE.Vector3(...dir), new THREE.Vector3(-4.5, 0.02, -3.5).add(OFF), 3, col, 0.3, 0.18);
    axes.add(a);
  }
  scene.add(axes);
  // analogy arrows: man -> woman and king -> queen point the same way
  const at = (w) => pts.find((p) => p.w === w).m.position;
  const mkArrow = (a, b, col) => {
    const d = at(b).clone().sub(at(a));
    const ar = new THREE.ArrowHelper(d.clone().normalize(), at(a), d.length(), col, 0.35, 0.2);
    scene.add(ar); return ar;
  };
  const a1 = mkArrow("man", "woman", 0x0f172a), a2 = mkArrow("king", "queen", 0x0f172a);

  const KEYS = [[0, [0, 6, 16], [-2, 1.5, 2]], [4.5, [-2, 5, 13], [-3, 1.5, 2.5]], [12, [-5, 5, 10], [-5, 2.5, 1]],
                [18, [9, 7, 12], [4, 2.5, 0]], [25, [7, 4.5, 8.5], [5, 2, 0]], [32, [12, 8, 9], [4, 2.5, 0]]];

  return function renderAt(t) {
    const split = smooth((t - 5) / 1.5), look = smooth((t - 12) / 2.5), fly = smooth((t - 14.5) / 2.5);
    toks.forEach((c, i) => {
      const home = c.userData.home.clone();
      home.x += (i - 2.5) * 0.35 * split;                       // tokens separate
      const row = table.position.clone().add(new THREE.Vector3(0, 0.3 + (8 + i * 3) * 0.16, 0.6));
      const p = home.clone().lerp(row, look);
      c.position.copy(p);
      c.scale.setScalar(lerp(1, 0.35, look));
      c.visible = fly < 0.98;
    });
    pts.forEach(({ m, l, w }, i) => {
      const sentence = SENT.some(([s]) => s.trim() === w);
      const k = sentence ? fly : smooth((t - 15 - hash(i) * 2) / 2);
      m.scale.setScalar(k);
      l.material.opacity = k;
      m.visible = k > 0.01;
    });
    axes.visible = t > 14.5;
    const an = smooth((t - 25) / 1.5);
    for (const a of [a1, a2]) { a.visible = an > 0.01; a.setLength(Math.max(0.01, at("man").distanceTo(at("woman")) * an), 0.35, 0.2); }
    a2.setLength(Math.max(0.01, at("king").distanceTo(at("queen")) * an), 0.35, 0.2);
    cameraPath(camera, KEYS, t);
    render();

    hud.clear();
    hud.title("From tokens to embeddings", "how a model turns text into numbers it can compute with", fade(t, -1, 4.2, 0.8));
    story(hud, CH, t);
    const a = fade(t, 12.5, 24.5);
    if (a > 0) {
      hud.alpha(a);
      hud.panel(940, 30, 310, 150);
      hud.text("Shapes", 960, 62, 19, C.ink, "left", 700);
      hud.text("ids: (T)          e.g. T = 6", 960, 94, 16, C.muted);
      hud.text("table E: (V × d)  V ≈ 150k, d ≈ 4k", 960, 122, 16, C.muted);
      hud.text("x = E[ids]: (T × d)", 960, 150, 16, C.accent, "left", 650);
      hud.alpha(1);
    }
    const b = fade(t, 25.5, 40);
    if (b > 0) {
      hud.alpha(b);
      hud.panel(30, 30, 380, 110);
      hud.text("vec(king) − vec(man) + vec(woman)", 50, 70, 18, C.ink, "left", 650);
      hud.text("≈ vec(queen)   (cosine similarity)", 50, 104, 18, C.accent, "left", 650);
      hud.alpha(1);
    }
  };
}
