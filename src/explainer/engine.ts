// Live 3D explainer for one lesson: a light-themed Three.js scene (one of ten motifs) driven by a timeline of
// beats from public/content/explainers.json. Everything is a function of time t, so seeking is exact.
import * as THREE from "three";

export type Beat = { kind: "title" | "analogy" | "section" | "formula" | "ideas" | "pitfall" | "outro";
  title: string; text: string; tex?: string; items?: string[] };
export type Script = { motif: string; labels: string[]; beats: Beat[] };

const DUR: Record<Beat["kind"], number> = { title: 4, analogy: 6.5, section: 6.5, formula: 6, ideas: 7.5, pitfall: 5.5, outro: 4 };
export function timeline(beats: Beat[]) {
  let t = 0;
  const starts = beats.map((b) => { const s = t; t += DUR[b.kind]; return s; });
  return { starts, duration: t, at(time: number) {
    let i = starts.length - 1;
    while (i > 0 && starts[i] > time) i--;
    return { i, frac: Math.min(1, (time - starts[i]) / DUR[beats[i].kind]) };
  } };
}

const clamp = (x: number, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const smooth = (x: number) => { x = clamp(x); return x * x * (3 - 2 * x); };
const hash = (i: number) => { const x = Math.sin(i * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };
const mat = (color: number, extra: THREE.MeshStandardMaterialParameters = {}) =>
  new THREE.MeshStandardMaterial({ color, roughness: 0.45, metalness: 0.05, ...extra });
const PALETTE = [0x7c3aed, 0x2563eb, 0x0d9488, 0x16a34a, 0xd97706, 0xdb2777];

function label(text: string, color = "#0f172a", size = 30) {
  const c = document.createElement("canvas"), x = c.getContext("2d")!;
  const font = `600 ${size}px -apple-system, "Segoe UI", system-ui, sans-serif`;
  x.font = font;
  const w = Math.ceil(x.measureText(text).width) + 32;
  c.width = w; c.height = size + 22;
  x.font = font;
  x.fillStyle = "rgba(255,255,255,0.94)"; x.strokeStyle = "rgba(15,23,42,0.15)"; x.lineWidth = 2;
  x.beginPath(); x.roundRect(1, 1, w - 2, c.height - 2, 10); x.fill(); x.stroke();
  x.fillStyle = color; x.textAlign = "center"; x.textBaseline = "middle"; x.fillText(text, w / 2, c.height / 2 + 1);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
  s.scale.set(w / 110, c.height / 110, 1);
  s.renderOrder = 10;
  return s;
}

type Ctx = { scene: THREE.Scene; labels: string[]; seed: number; accent: number };
/** update(t, beatIndex, beatFraction, overallProgress) */
type Update = (t: number, i: number, f: number, p: number) => void;

// ------------------------------------------------------------------------------------------- motifs
const MOTIFS: Record<string, (c: Ctx) => Update> = {
  loop({ scene, labels, accent }) {
    const names = labels.length ? labels : ["Agent", "Action", "Environment", "Reward"];
    const ring = new THREE.Mesh(new THREE.TorusGeometry(3.2, 0.06, 12, 96), mat(0xcbd5e1));
    ring.rotation.x = Math.PI / 2; ring.position.y = 1.2; scene.add(ring);
    const nodes = names.map((n, k) => {
      const a = (k / names.length) * Math.PI * 2;
      const m = new THREE.Mesh(k % 2 ? new THREE.SphereGeometry(0.45, 24, 16) : new THREE.BoxGeometry(1.1, 1.1, 1.1),
        mat(k === 0 ? accent : PALETTE[(k + 1) % PALETTE.length]));
      m.position.set(Math.cos(a) * 3.2, 1.2, Math.sin(a) * 3.2); m.castShadow = true; scene.add(m);
      const l = label(n); l.position.copy(m.position).add(new THREE.Vector3(0, 1.0, 0)); scene.add(l);
      return m;
    });
    const dots = Array.from({ length: 10 }, () => { const d = new THREE.Mesh(new THREE.SphereGeometry(0.12, 12, 8), mat(0xf59e0b)); scene.add(d); return d; });
    return (t) => {
      nodes.forEach((m, k) => { m.rotation.y = t * 0.6 + k; m.scale.setScalar(1 + 0.12 * Math.max(0, Math.sin(t * 2 - k * 1.5))); });
      dots.forEach((d, k) => { const a = t * 0.9 + (k / dots.length) * Math.PI * 2; d.position.set(Math.cos(a) * 3.2, 1.2, Math.sin(a) * 3.2); });
    };
  },

  grid({ scene, accent }) {
    const n = 6, bars: THREE.Mesh[] = [];
    for (let x = 0; x < n; x++) for (let z = 0; z < n; z++) {
      const tile = new THREE.Mesh(new THREE.BoxGeometry(0.95, 0.08, 0.95), mat(x === n - 1 && z === 0 ? 0x86efac : 0xf1f5f9));
      tile.position.set(x - n / 2 + 0.5, 0.04, z - n / 2 + 0.5); tile.receiveShadow = true; scene.add(tile);
      const b = new THREE.Mesh(new THREE.BoxGeometry(0.7, 1, 0.7), mat(accent, { transparent: true, opacity: 0.9 }));
      b.position.copy(tile.position); b.castShadow = true; b.userData.d = (n - 1 - x) + z; scene.add(b); bars.push(b);
    }
    return (t, _i, _f, p) => {
      const wave = p * 14;
      bars.forEach((b) => {
        const d = b.userData.d as number, v = Math.pow(0.85, d) * smooth(wave - d * 0.9);
        const h = Math.max(0.02, v * 3); b.scale.y = h; b.position.y = 0.08 + h / 2;
        b.material instanceof THREE.MeshStandardMaterial && b.material.emissive.set(Math.abs(wave - d * 0.9 - 0.5) < 0.5 ? 0xf59e0b : 0);
      });
      void t;
    };
  },

  bars({ scene, accent, seed }) {
    const n = 8, logits = Array.from({ length: n }, (_, k) => 2.5 * hash(seed + k) + (k === 2 ? 1.5 : 0));
    const bars = logits.map((_, k) => {
      const b = new THREE.Mesh(new THREE.BoxGeometry(0.7, 1, 0.7), mat(k === 2 ? accent : 0xa5b4fc));
      b.position.set((k - n / 2 + 0.5) * 1.05, 0, 0); b.castShadow = true; scene.add(b); return b;
    });
    return (t) => {
      const T = 1.4 + Math.sin(t * 0.5) * 1.0, z = logits.map((v) => Math.exp(v / T)), s = z.reduce((a, b) => a + b, 0);
      bars.forEach((b, k) => { const h = Math.max(0.03, (z[k] / s) * 9); b.scale.y = h; b.position.y = h / 2; });
    };
  },

  dist({ scene, accent, seed }) {
    const bins = 15, cols = Array.from({ length: bins }, () => 0), cubes: THREE.Mesh[] = [];
    const N = 220, samples = Array.from({ length: N }, (_, k) => {
      const u1 = hash(seed + k * 2) || 1e-6, u2 = hash(seed + k * 2 + 1);
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    });
    const slot = samples.map((v) => {
      const b = Math.max(0, Math.min(bins - 1, Math.floor((v + 3) / 6 * bins)));
      return [b, cols[b]++] as const;
    });
    samples.forEach((_, k) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.18, 0.42), mat(k % 9 === 0 ? 0xf59e0b : accent));
      m.castShadow = true; scene.add(m); cubes.push(m);
    });
    const curve = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(Array.from({ length: 40 }, (_, k) => {
      // expected stack height for value x: N * pdf(x) * binWidth(0.4) * cubeHeight(0.19); bin spacing maps x -> 1.2x
      const x = -3 + (6 * k) / 39;
      return new THREE.Vector3(x * 1.2, N * 0.4 * 0.19 * Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI) + 0.05, -0.4);
    })), 80, 0.04, 8), mat(0x0f172a));
    scene.add(curve);
    return (t, _i, _f, p) => {
      const shown = Math.floor(clamp(p * 1.6) * N);
      cubes.forEach((m, k) => {
        const [b, h] = slot[k], landed = k < shown;
        const x = (b - bins / 2 + 0.5) * 0.48, y = 0.09 + h * 0.19;
        const fall = landed ? 0 : 6 + hash(k) * 2;
        m.visible = k < shown + 6;
        m.position.set(x, y + fall * (1 - smooth((p * 1.6 * N - k) / 6)), 0);
      });
      curve.visible = p > 0.6;
      void t;
    };
  },

  tokens({ scene, accent, seed }) {
    const n = 7, toks = Array.from({ length: n }, (_, k) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(1.0, 0.5, 0.6), mat(k === n - 2 ? accent : 0xe0e7ff));
      m.position.set((k - n / 2 + 0.5) * 1.3, 0.3, 1.2); m.castShadow = true; scene.add(m); return m;
    });
    const q = n - 2, w = Array.from({ length: q }, (_, k) => 0.2 + hash(seed + k) * (k === 1 ? 3 : 1));
    const arcs = Array.from({ length: q }, (_, k) => {
      const a = toks[q].position.clone().setY(0.6), b = toks[k].position.clone().setY(0.6);
      const mid = a.clone().add(b).multiplyScalar(0.5).setY(0.8 + Math.abs(q - k) * 0.45);
      const m = new THREE.Mesh(new THREE.TubeGeometry(new THREE.QuadraticBezierCurve3(a, mid, b), 40, 0.02 + 0.05 * w[k], 8),
        mat(accent, { transparent: true, opacity: 0.8 }));
      scene.add(m); return m;
    });
    const layers = Array.from({ length: 4 }, (_, k) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(9.5, 0.12, 1.4), mat(0xc4b5fd, { transparent: true, opacity: 0.35 }));
      m.position.set(0, 3.2 + k * 0.45, -1.2); scene.add(m); return m;
    });
    return (t) => {
      arcs.forEach((a, k) => { (a.material as THREE.MeshStandardMaterial).opacity = 0.15 + 0.7 * Math.max(0, Math.sin(t * 1.2 - k * 0.5)); });
      toks.forEach((m, k) => { m.position.y = 0.3 + 0.12 * Math.max(0, Math.sin(t * 2 - k * 0.7)); });
      layers.forEach((m, k) => { (m.material as THREE.MeshStandardMaterial).opacity = 0.25 + 0.3 * Math.max(0, Math.sin(t * 1.5 - k)); });
    };
  },

  stack({ scene, accent }) {
    const L = 6, layers = Array.from({ length: L }, (_, k) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(5.5, 0.35, 3), mat(k % 2 ? 0xddd6fe : 0xc7d2fe, { transparent: true, opacity: 0.8 }));
      m.position.y = 0.4 + k * 0.7; m.castShadow = true; scene.add(m); return m;
    });
    const gpus = Array.from({ length: 4 }, (_, k) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.6, 1.2), mat(0x0f172a));
      m.position.set(-4.8 + 0 * k, 0.3 + k * 1.05, 2.6); m.castShadow = true; scene.add(m); return m;
    });
    const parts = Array.from({ length: 30 }, () => { const d = new THREE.Mesh(new THREE.SphereGeometry(0.1, 10, 8), mat(accent)); scene.add(d); return d; });
    return (t) => {
      parts.forEach((d, k) => {
        const y = ((t * 1.2 + k * 0.37) % (L * 0.7 + 0.6));
        d.position.set(-2.4 + (k % 6) * 0.95, 0.2 + y, -1 + (k % 3) * 1);
      });
      layers.forEach((m, k) => { m.position.x = Math.sin(t * 0.8 + k) * 0.05; });
      gpus.forEach((m, k) => (m.material as THREE.MeshStandardMaterial).emissive.set(Math.sin(t * 3 + k) > 0.6 ? 0x22c55e : 0));
    };
  },

  landscape({ scene, accent }) {
    const geo = new THREE.PlaneGeometry(10, 10, 48, 48);
    const f = (x: number, y: number) => 0.18 * (x * x + 2.2 * y * y) / 4 + 0.35 * Math.sin(x * 1.3) * Math.cos(y * 1.1);
    const pos = geo.attributes.position;
    for (let k = 0; k < pos.count; k++) pos.setZ(k, f(pos.getX(k), pos.getY(k)));
    geo.computeVertexNormals();
    const surf = new THREE.Mesh(geo, mat(0xc4b5fd, { side: THREE.DoubleSide, flatShading: false, transparent: true, opacity: 0.9 }));
    surf.rotation.x = -Math.PI / 2; surf.receiveShadow = true; scene.add(surf);
    const wire = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: 0x7c3aed, wireframe: true, transparent: true, opacity: 0.12 }));
    wire.rotation.x = -Math.PI / 2; scene.add(wire);
    // gradient descent path with momentum from a fixed start
    const path: THREE.Vector3[] = [];
    let x = -4, y = 3.2, vx = 0, vy = 0;
    for (let k = 0; k < 140; k++) {
      const e = 1e-3, gx = (f(x + e, y) - f(x - e, y)) / (2 * e), gy = (f(x, y + e) - f(x, y - e)) / (2 * e);
      vx = 0.85 * vx - 0.12 * gx; vy = 0.85 * vy - 0.12 * gy; x += vx; y += vy;
      path.push(new THREE.Vector3(x, f(x, y) + 0.22, -y));
    }
    const ball = new THREE.Mesh(new THREE.SphereGeometry(0.22, 24, 16), mat(accent === 0x7c3aed ? 0xf59e0b : accent));
    ball.castShadow = true; scene.add(ball);
    const trail = path.map((p) => { const d = new THREE.Mesh(new THREE.SphereGeometry(0.05, 8, 6), mat(0x0f172a)); d.position.copy(p); scene.add(d); return d; });
    return (_t, _i, _f, p) => {
      const k = Math.min(path.length - 1, Math.floor(clamp(p * 1.3) * (path.length - 1)));
      ball.position.copy(path[k]);
      trail.forEach((d, j) => (d.visible = j < k));
    };
  },

  pipeline({ scene, labels, accent }) {
    const names = labels.length ? labels : ["Input", "Process", "Store", "Serve", "Output"];
    const boxes = names.map((n, k) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.9 + (k % 2) * 0.4, 1.3), mat(k === 0 ? accent : PALETTE[k % PALETTE.length]));
      m.position.set((k - (names.length - 1) / 2) * 2.4, 0.5, 0); m.castShadow = true; scene.add(m);
      const l = label(n); l.position.copy(m.position).add(new THREE.Vector3(0, 1.35, 0)); scene.add(l);
      return m;
    });
    for (let k = 0; k < boxes.length - 1; k++) {
      const a = boxes[k].position, b = boxes[k + 1].position;
      const pipe = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, b.x - a.x - 1.5, 10), mat(0x94a3b8));
      pipe.rotation.z = Math.PI / 2; pipe.position.set((a.x + b.x) / 2, 0.5, 0); scene.add(pipe);
    }
    const x0 = boxes[0].position.x, x1 = boxes.at(-1)!.position.x;
    const packets = Array.from({ length: 12 }, () => { const d = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.2, 0.2), mat(0xf59e0b)); d.castShadow = true; scene.add(d); return d; });
    return (t) => {
      packets.forEach((d, k) => {
        const u = ((t * 0.18 + k / packets.length) % 1);
        d.position.set(x0 + (x1 - x0) * u, 0.5 + 0.15 * Math.sin(u * 40), 0.75 * ((k % 3) - 1) * 0.4);
      });
      boxes.forEach((m, k) => { const u = ((t * 0.18) % 1) * (boxes.length - 1); m.scale.y = 1 + 0.12 * Math.max(0, 1 - Math.abs(u - k)); });
    };
  },

  graph({ scene, accent }) {
    const pts: THREE.Vector3[] = [], edges: [number, number][] = [];
    const levels = [1, 2, 4, 6];
    let id = 0;
    const ids: number[][] = levels.map((n, l) => Array.from({ length: n }, (_, k) => {
      pts.push(new THREE.Vector3((k - (n - 1) / 2) * (6 / Math.max(1, n)) * 1.2, 4 - l * 1.25, 0)); return id++;
    }));
    for (let l = 1; l < levels.length; l++) ids[l].forEach((c, k) => edges.push([ids[l - 1][Math.floor((k * ids[l - 1].length) / ids[l].length)], c]));
    const nodes = pts.map((p) => { const m = new THREE.Mesh(new THREE.SphereGeometry(0.32, 20, 14), mat(0xe0e7ff)); m.position.copy(p); m.castShadow = true; scene.add(m); return m; });
    edges.forEach(([a, b]) => {
      const d = pts[b].clone().sub(pts[a]);
      const m = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, d.length(), 8), mat(0x94a3b8));
      m.position.copy(pts[a]).add(d.clone().multiplyScalar(0.5));
      m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.normalize()); scene.add(m);
    });
    return (_t, _i, _f, p) => {
      const visited = Math.floor(clamp(p * 1.4) * nodes.length);   // BFS order = creation order
      nodes.forEach((m, k) => (m.material as THREE.MeshStandardMaterial).color.set(k < visited ? accent : k === visited ? 0xf59e0b : 0xe0e7ff));
    };
  },

  array({ scene, accent, seed }) {
    const n = 12, vals = Array.from({ length: n }, (_, k) => 0.5 + hash(seed + k) * 3.5);
    const bars = vals.map((v, k) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(0.6, 1, 0.6), mat(0xc7d2fe)); m.scale.y = v;
      m.position.set((k - n / 2 + 0.5) * 0.8, v / 2, 0); m.castShadow = true; scene.add(m); return m;
    });
    const ptr = [0, 1].map((k) => { const c = new THREE.Mesh(new THREE.ConeGeometry(0.22, 0.5, 16), mat(k ? 0xf59e0b : accent)); c.rotation.x = Math.PI; scene.add(c); return c; });
    return (t) => {
      const w = 3, start = Math.floor((t * 0.8) % (n - w + 1));
      bars.forEach((m, k) => (m.material as THREE.MeshStandardMaterial).color.set(k >= start && k < start + w ? accent : 0xc7d2fe));
      ptr[0].position.set((start - n / 2 + 0.5) * 0.8, 4.6, 0);
      ptr[1].position.set((start + w - 1 - n / 2 + 0.5) * 0.8, 4.6, 0);
    };
  },
};

// ------------------------------------------------------------------------------------------- stage
export function createExplainer(canvas: HTMLCanvasElement, script: Script, opts: { seed: number; accent: string; still: boolean }) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf6f7fb);
  scene.fog = new THREE.Fog(0xf6f7fb, 22, 48);
  const camera = new THREE.PerspectiveCamera(40, 16 / 9, 0.1, 100);
  scene.add(new THREE.HemisphereLight(0xffffff, 0xdfe6f0, 1.4));
  const sun = new THREE.DirectionalLight(0xffffff, 1.5);
  sun.position.set(8, 14, 10); sun.castShadow = true; sun.shadow.mapSize.set(1024, 1024);
  Object.assign(sun.shadow.camera, { left: -12, right: 12, top: 12, bottom: -12 });
  scene.add(sun);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(120, 120), new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1 }));
  floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; scene.add(floor);
  const grid = new THREE.GridHelper(120, 120, 0xdde2ea, 0xdde2ea);
  grid.position.y = 0.003; scene.add(grid);

  const accent = new THREE.Color(opts.accent || "#7c3aed").getHex();
  const update = (MOTIFS[script.motif] ?? MOTIFS.loop)({ scene, labels: script.labels, seed: opts.seed, accent });
  const tl = timeline(script.beats);

  function resize() {
    const w = canvas.clientWidth || 640, h = canvas.clientHeight || 360;
    renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  resize();

  function renderAt(t: number) {
    const { i, frac } = tl.at(t), p = clamp(t / tl.duration);
    update(t, i, frac, p);
    const kind = script.beats[i].kind;
    const a = (opts.still ? 0.6 : 0.6 + t * 0.07), r = kind === "title" || kind === "outro" ? 13 : 11;
    const h = kind === "formula" || kind === "ideas" ? 8 : 6.2;
    camera.position.set(Math.sin(a) * r, h, Math.cos(a) * r);
    camera.lookAt(0, 1.4, 0);
    renderer.render(scene, camera);
  }

  return {
    duration: tl.duration, starts: tl.starts, at: tl.at, renderAt, resize,
    dispose() {
      scene.traverse((o) => {
        const m = o as THREE.Mesh;
        m.geometry?.dispose();
        const ms = Array.isArray(m.material) ? m.material : m.material ? [m.material] : [];
        ms.forEach((x) => { (x as THREE.MeshBasicMaterial).map?.dispose(); x.dispose(); });
      });
      renderer.dispose();
      renderer.forceContextLoss();
    },
  };
}
