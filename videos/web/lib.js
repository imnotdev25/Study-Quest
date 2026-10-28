// Shared stage for every explainer scene: light theme, renderer + 2D HUD overlay, labels, captions, charts.
// Scenes are pure functions of time t (seconds) so record.mjs can render them frame by frame.
import * as THREE from "three";

export { THREE };
export const W = 1280, H = 720;
export const C = {
  bg: 0xf6f7fb, floor: 0xffffff, grid: 0xdde2ea, ink: "#0f172a", muted: "#64748b", panel: "rgba(255,255,255,0.9)",
  line: "rgba(15,23,42,0.12)", accent: "#7c3aed", blue: "#2563eb", green: "#16a34a", red: "#dc2626", amber: "#d97706",
  teal: "#0d9488", pink: "#db2777",
};
export const hex = (c) => new THREE.Color(c);

export const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
export const smooth = (x) => { x = clamp(x); return x * x * (3 - 2 * x); };
export const lerp = (a, b, k) => a + (b - a) * k;
/** 0 before a, ramps to 1, holds, ramps back to 0 after b. */
export const fade = (t, a, b, f = 0.5) => clamp(Math.min((t - a) / f, (b - t) / f));
export const fract = (x) => x - Math.floor(x);
export const hash = (i) => fract(Math.sin(i * 127.1 + 311.7) * 43758.5453);

export function makeStage({ fov = 40 } = {}) {
  const host = document.getElementById("stage");
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(1);
  renderer.setSize(W, H);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  host.appendChild(renderer.domElement);
  const hudCanvas = document.createElement("canvas");
  hudCanvas.width = W; hudCanvas.height = H;
  host.appendChild(hudCanvas);
  const g = hudCanvas.getContext("2d");

  const scene = new THREE.Scene();
  scene.background = hex(C.bg);
  scene.fog = new THREE.Fog(C.bg, 26, 60);
  const camera = new THREE.PerspectiveCamera(fov, W / H, 0.1, 200);
  scene.add(new THREE.HemisphereLight(0xffffff, 0xdfe6f0, 1.4));
  const sun = new THREE.DirectionalLight(0xffffff, 1.5);
  sun.position.set(8, 16, 10);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -16, right: 16, top: 16, bottom: -16 });
  sun.shadow.radius = 4;
  scene.add(sun);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), new THREE.MeshStandardMaterial({ color: C.floor, roughness: 1 }));
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  scene.add(floor);
  const grid = new THREE.GridHelper(200, 200, C.grid, C.grid);
  grid.position.y = 0.003;
  grid.material.transparent = true; grid.material.opacity = 0.6;
  scene.add(grid);

  // ---------------------------------------------------------------- HUD helpers
  const hud = {
    clear() { g.clearRect(0, 0, W, H); },
    alpha(a) { g.globalAlpha = clamp(a); },
    text(s, x, y, size = 20, color = C.ink, align = "left", weight = 500) {
      g.font = `${weight} ${size}px -apple-system, "Segoe UI", system-ui, sans-serif`;
      g.fillStyle = color; g.textAlign = align; g.textBaseline = "alphabetic";
      g.fillText(s, x, y);
    },
    width(s, size = 20, weight = 500) {
      g.font = `${weight} ${size}px -apple-system, "Segoe UI", system-ui, sans-serif`;
      return g.measureText(s).width;
    },
    panel(x, y, w, h, fill = C.panel) {
      g.save();
      g.shadowColor = "rgba(15,23,42,0.10)"; g.shadowBlur = 18; g.shadowOffsetY = 4;
      g.fillStyle = fill; g.beginPath(); g.roundRect(x, y, w, h, 14); g.fill();
      g.restore();
      g.strokeStyle = C.line; g.lineWidth = 1; g.beginPath(); g.roundRect(x, y, w, h, 14); g.stroke();
    },
    bar(x, y, w, h, color, r = 5) { g.fillStyle = color; g.beginPath(); g.roundRect(x, y, Math.max(0, w), h, r); g.fill(); },
    caption(s, a) {
      if (a <= 0) return;
      g.globalAlpha = clamp(a);
      const w = hud.width(s, 23) + 48;
      hud.panel(W / 2 - w / 2, H - 86, w, 50);
      hud.text(s, W / 2, H - 53, 23, C.ink, "center");
      g.globalAlpha = 1;
    },
    title(title, sub, a) {
      if (a <= 0) return;
      g.globalAlpha = clamp(a);
      g.fillStyle = "rgba(246,247,251,0.72)"; g.fillRect(0, 0, W, H);
      hud.text(title, W / 2, 330, 52, C.ink, "center", 750);
      hud.text(sub, W / 2, 378, 23, C.accent, "center", 600);
      hud.text("Study Quest", W / 2, 650, 16, C.muted, "center", 600);
      g.globalAlpha = 1;
    },
    /** series: [{data, color, upto}], opts: {title, n, range:[lo,hi], digits} */
    chart(x, y, w, h, series, opts) {
      hud.panel(x, y, w, h);
      hud.text(opts.title, x + 16, y + 28, 16, C.ink, "left", 650);
      const px = x + 44, py = y + 42, pw = w - 60, ph = h - 64, [lo, hi] = opts.range;
      const X = (i) => px + (i / Math.max(1, opts.n - 1)) * pw, Y = (v) => py + ph - ((clamp(v, lo, hi) - lo) / (hi - lo)) * ph;
      g.strokeStyle = C.line; g.lineWidth = 1;
      for (let k = 0; k <= 3; k++) {
        const yy = py + (ph * k) / 3; g.beginPath(); g.moveTo(px, yy); g.lineTo(px + pw, yy); g.stroke();
        hud.text((hi - ((hi - lo) * k) / 3).toFixed(opts.digits ?? 1), px - 8, yy + 4, 12, C.muted, "right");
      }
      for (const s of series) {
        g.strokeStyle = s.color; g.lineWidth = 3; g.beginPath();
        const m = Math.min(s.data.length, Math.floor(s.upto ?? s.data.length));
        for (let i = 0; i < m; i++) i ? g.lineTo(X(i), Y(s.data[i])) : g.moveTo(X(i), Y(s.data[i]));
        g.stroke();
      }
    },
    g,
  };

  return {
    THREE, scene, camera, renderer, hud,
    render() { renderer.render(scene, camera); },
  };
}

/** A text sprite that always faces the camera. */
export function label(text, { size = 34, color = C.ink, bg = "rgba(255,255,255,0.92)", scale = 1 } = {}) {
  const c = document.createElement("canvas"), x = c.getContext("2d");
  const font = `600 ${size}px -apple-system, "Segoe UI", system-ui, sans-serif`;
  x.font = font;
  const w = Math.ceil(x.measureText(text).width) + 36;
  c.width = w; c.height = size + 26;
  x.font = font;
  if (bg) {
    x.fillStyle = bg; x.strokeStyle = "rgba(15,23,42,0.15)"; x.lineWidth = 2;
    x.beginPath(); x.roundRect(1, 1, w - 2, c.height - 2, 12); x.fill(); x.stroke();
  }
  x.fillStyle = color; x.textBaseline = "middle"; x.textAlign = "center"; x.fillText(text, w / 2, c.height / 2 + 1);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
  s.scale.set((w / 110) * scale, (c.height / 110) * scale, 1);
  s.renderOrder = 10;
  return s;
}

/** A flat card (plane) with text drawn on it, e.g. a token or a model response. */
export function card(lines, { w = 2.4, h = 1.2, color = "#ffffff", border = "#cbd5e1", ink = C.ink, size = 44 } = {}) {
  const px = 220, c = document.createElement("canvas"), x = c.getContext("2d");
  c.width = Math.round(w * px); c.height = Math.round(h * px);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const draw = (bd = border, bgc = color) => {
    x.clearRect(0, 0, c.width, c.height);
    x.fillStyle = bgc; x.strokeStyle = bd; x.lineWidth = 14;
    x.beginPath(); x.roundRect(8, 8, c.width - 16, c.height - 16, 36); x.fill(); x.stroke();
    x.fillStyle = ink; x.textAlign = "center"; x.textBaseline = "middle";
    x.font = `600 ${size}px -apple-system, "Segoe UI", system-ui, sans-serif`;
    const ls = Array.isArray(lines) ? lines : [lines];
    ls.forEach((l, i) => x.fillText(l, c.width / 2, c.height / 2 + (i - (ls.length - 1) / 2) * size * 1.25));
    tex.needsUpdate = true;
  };
  draw();
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, 0.08),
    [0, 0, 0, 0, 0, 0].map((_, i) => i === 4 ? new THREE.MeshStandardMaterial({ map: tex, roughness: 0.6 })
                                             : new THREE.MeshStandardMaterial({ color: 0xe2e8f0, roughness: 0.6 })));
  m.castShadow = true;
  m.userData.redraw = draw;
  return m;
}

/** Interpolate camera position/target along keyframes [[time, [x,y,z], [tx,ty,tz]], ...]. */
export function cameraPath(camera, keys, t) {
  let i = keys.findIndex((k) => k[0] > t);
  if (i < 0) i = keys.length - 1;
  if (i === 0) i = 1;
  const [ta, pa, la] = keys[i - 1], [tb, pb, lb] = keys[i];
  const k = smooth((t - ta) / Math.max(1e-3, tb - ta));
  camera.position.set(lerp(pa[0], pb[0], k), lerp(pa[1], pb[1], k), lerp(pa[2], pb[2], k));
  camera.lookAt(lerp(la[0], lb[0], k), lerp(la[1], lb[1], k), lerp(la[2], lb[2], k));
}

export function softmax(z, T = 1) {
  const m = Math.max(...z), e = z.map((v) => Math.exp((v - m) / T)), s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export const mat = (color, extra = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.45, metalness: 0.05, ...extra });

// ---------------------------------------------------------------- storytelling helpers (v2 scenes)
/** Chapters: [{t: start seconds, title, text}]. Returns {i, k, local} for time t. */
export function chapterAt(chapters, t) {
  let i = 0;
  while (i + 1 < chapters.length && chapters[i + 1].t <= t) i++;
  const end = i + 1 < chapters.length ? chapters[i + 1].t : chapters[i].t + 8;
  return { i, local: t - chapters[i].t, k: clamp((t - chapters[i].t) / (end - chapters[i].t)), end };
}

function wrap(hud, text, size, maxW) {
  const words = text.split(" "), lines = [];
  let line = "";
  for (const w of words) {
    const test = line ? line + " " + w : w;
    if (hud.width(test, size) > maxW && line) { lines.push(line); line = w; } else line = test;
  }
  if (line) lines.push(line);
  return lines;
}

/** Draw chapter progress (top) and the narration card (bottom) for time t. Skips chapter 0 (title card). */
export function story(hud, chapters, t, { skipFirst = true } = {}) {
  const c = chapterAt(chapters, t);
  const g = hud.g;
  // progress pills
  const n = chapters.length - (skipFirst ? 1 : 0);
  if (!(skipFirst && c.i === 0)) {
    const w = 46, gap = 8, x0 = W / 2 - (n * w + (n - 1) * gap) / 2;
    for (let j = 0; j < n; j++) {
      const ci = j + (skipFirst ? 1 : 0), x = x0 + j * (w + gap);
      g.fillStyle = "rgba(15,23,42,0.10)"; g.beginPath(); g.roundRect(x, 18, w, 6, 3); g.fill();
      const fill = ci < c.i ? 1 : ci === c.i ? c.k : 0;
      g.fillStyle = C.accent; g.beginPath(); g.roundRect(x, 18, w * fill, 6, 3); g.fill();
    }
  }
  const ch = chapters[c.i];
  if (!ch.text || (skipFirst && c.i === 0)) return c;
  const a = clamp(Math.min(c.local / 0.45, (c.end - (chapters[c.i].t + c.local)) / 0.35 + 0.0001));
  g.globalAlpha = clamp(a);
  const lines = wrap(hud, ch.text, 21, 820);
  const tw = Math.max(hud.width(ch.title, 15, 700), ...lines.map((l) => hud.width(l, 21))) + 52;
  const h = 46 + lines.length * 28;
  hud.panel(W / 2 - tw / 2, H - 26 - h, tw, h);
  hud.text(ch.title.toUpperCase(), W / 2, H - 26 - h + 28, 14, C.accent, "center", 750);
  lines.forEach((l, j) => hud.text(l, W / 2, H - 26 - h + 56 + j * 28, 21, C.ink, "center"));
  g.globalAlpha = 1;
  return c;
}

/** A sprite label whose text can change (texture rebuilt only when it does). */
export function dynLabel(opts = {}) {
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ transparent: true, depthTest: false }));
  s.renderOrder = 11;
  let cur = null;
  s.userData.set = (text, color = opts.color || C.ink) => {
    const key = text + color;
    if (key === cur) return;
    cur = key;
    const l = label(text, { ...opts, color });
    s.material.map?.dispose();
    s.material.map = l.material.map; s.material.needsUpdate = true;
    s.scale.copy(l.scale);
  };
  return s;
}

export const ease = { out: (x) => 1 - (1 - clamp(x)) ** 3, back: (x) => { x = clamp(x); const c = 1.7; return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2; } };

export function robot(color = 0x7c3aed) {
  const g = new THREE.Group();
  const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.32, 0.36, 8, 16), mat(color));
  body.position.y = 0.55; body.castShadow = true; g.add(body);
  for (const s of [-1, 1]) {
    const e = new THREE.Mesh(new THREE.SphereGeometry(0.065, 12, 8), mat(0xffffff, { emissive: 0xffffff, emissiveIntensity: 0.6 }));
    e.position.set(s * 0.12, 0.72, 0.27); g.add(e);
  }
  const ant = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.25), mat(0x334155));
  ant.position.y = 1.08; g.add(ant);
  const tip = new THREE.Mesh(new THREE.SphereGeometry(0.06, 10, 8), mat(0xf59e0b, { emissive: 0xf59e0b, emissiveIntensity: 0.6 }));
  tip.position.y = 1.22; g.add(tip);
  return g;
}

export function coin(r = 0.28) {
  const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, 0.07, 32), mat(0xfbbf24, { metalness: 0.6, roughness: 0.3, emissive: 0xb45309, emissiveIntensity: 0.15 }));
  m.rotation.x = Math.PI / 2; m.castShadow = true;
  return m;
}

/** A flat arrow lying on the floor pointing along (dx, dz). */
export function floorArrow(color = 0x0f172a, len = 0.7) {
  const g = new THREE.Group();
  const shaft = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.05, len * 0.6), mat(color));
  shaft.position.z = -len * 0.1; g.add(shaft);
  const head = new THREE.Mesh(new THREE.ConeGeometry(0.17, len * 0.4, 16), mat(color));
  head.rotation.x = -Math.PI / 2; head.position.z = -len * 0.4 - len * 0.1; g.add(head);
  g.userData.point = (dx, dz) => { g.rotation.y = Math.atan2(-dx, -dz); };
  return g;
}

export const fmt = (v, d = 1) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(d);
