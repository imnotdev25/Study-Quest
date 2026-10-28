// Videos: short 3D explainers for RL and LLMs. Click a poster to play; each links to the lessons it explains.
import { useState } from "react";
import { PATHS, type Data } from "../lib/data";
import { fmtDuration, useBlobSrc, useVideos, type Video } from "../lib/videos";
import { go } from "../App";

export function VideoPlayer({ v, autoPlay = false }: { v: Video; autoPlay?: boolean }) {
  const [on, setOn] = useState(autoPlay);
  const src = useBlobSrc(v.src, on);
  if (!on) {
    return (
      <button className="video-poster" onClick={() => setOn(true)} aria-label={`Play ${v.title}`}>
        <img src={v.poster} alt="" loading="lazy" />
        <span className="video-play" aria-hidden="true">▶</span>
        <span className="video-time">{fmtDuration(v.duration)}</span>
      </button>
    );
  }
  return (
    <div className="video-frame">
      {src ? <video src={src} poster={v.poster} controls autoPlay playsInline aria-label={v.title} />
           : <div className="loading">Loading video…</div>}
    </div>
  );
}

export function VideoCard({ v, data }: { v: Video; data: Data }) {
  const lessons = data.order.map((id) => data.topics.get(id)!)
    .filter((t) => v.keywords.some((k) => t.title.toLowerCase().includes(k.toLowerCase()))).slice(0, 3);
  return (
    <section className="card video-card">
      <VideoPlayer v={v} />
      <h3>{v.title}</h3>
      <p className="muted small">{v.desc}</p>
      {lessons.length > 0 && (
        <div className="row wrap">
          {lessons.map((t) => <button key={t.id} className="chip" onClick={() => go({ v: "topic", id: t.id })}>📖 {t.title}</button>)}
        </div>
      )}
    </section>
  );
}

export function Videos({ data }: { data: Data }) {
  const all = useVideos();
  const paths = PATHS.filter((p) => all.some((v) => v.path === p.id));
  return (
    <div className="page">
      <h1>Videos</h1>
      <p className="muted">Short 3D explainers. Each one also shows up inside the lessons it explains.</p>
      {!all.length && <p className="muted">No videos found. Render them with <code>cd videos && npm run record</code>.</p>}
      {paths.map((p) => (
        <div key={p.id}>
          <h2>{p.icon} {p.long}</h2>
          <div className="video-grid">{all.filter((v) => v.path === p.id).map((v) => <VideoCard key={v.id} v={v} data={data} />)}</div>
        </div>
      ))}
    </div>
  );
}
