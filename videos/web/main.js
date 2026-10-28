// Loads one scene (?scene=<id>) and exposes window.renderAt(t) for the recorder.
import { makeStage } from "./lib.js";

export const SCENES = ["rl_loop", "value_iteration", "grpo", "embeddings", "attention", "next_token"];

const q = new URLSearchParams(location.search);
const id = SCENES.includes(q.get("scene")) ? q.get("scene") : SCENES[0];
const RECORD = q.has("record");
if (RECORD) document.body.classList.add("record");

const mod = await import(`./scenes/${id}.js`);
const stage = makeStage(mod.stageOptions || {});
const renderAt = mod.setup(stage);
const meta = mod.meta;

window.SCENES = SCENES;
window.META = { id, ...meta };
window.renderAt = renderAt;
window.viewerReady = true;

if (!RECORD) {
  const pick = document.getElementById("pick");
  pick.innerHTML = SCENES.map((s) => `<option ${s === id ? "selected" : ""}>${s}</option>`).join("");
  pick.onchange = () => { location.search = `?scene=${pick.value}`; };
  let t = 0, last = performance.now(), playing = true;
  const play = document.getElementById("play"), seek = document.getElementById("seek"), time = document.getElementById("time");
  play.onclick = () => { playing = !playing; play.textContent = playing ? "Pause" : "Play"; };
  seek.oninput = () => { t = (seek.value / 1000) * meta.duration; };
  const loop = (now) => {
    if (playing) t = (t + (now - last) / 1000) % meta.duration;
    last = now;
    seek.value = String(Math.round((t / meta.duration) * 1000));
    time.textContent = `${t.toFixed(1)} / ${meta.duration}s`;
    renderAt(t);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
} else {
  renderAt(0);
}
