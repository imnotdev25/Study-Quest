// 3D explainer videos (rendered from videos/web/scenes by videos/record.mjs into public/videos/).
import { useEffect, useState } from "react";
import type { Topic } from "./data";

export type Video = { id: string; title: string; desc: string; path: string; keywords: string[]; duration: number;
  src: string; poster: string };

let cache: Promise<Video[]> | null = null;
export function loadVideos(): Promise<Video[]> {
  cache ??= fetch("videos/videos.json").then((r) => (r.ok ? r.json() : []), () => []);
  return cache;
}

export function useVideos() {
  const [v, setV] = useState<Video[]>([]);
  useEffect(() => { loadVideos().then(setV); }, []);
  return v;
}

/** Videos whose keywords appear in the topic title (case-insensitive). */
export function videosFor(t: Topic, all: Video[]) {
  const title = t.title.toLowerCase();
  return all.filter((v) => v.keywords.some((k) => title.includes(k.toLowerCase())));
}

/** Load the mp4 as a blob URL: works the same in the browser and in Tauri's asset protocol (no range requests). */
export function useBlobSrc(src: string, on: boolean) {
  const [url, setUrl] = useState<string>();
  useEffect(() => {
    if (!on) return;
    let u: string | undefined, live = true;
    fetch(src).then((r) => r.blob()).then((b) => { if (live) setUrl((u = URL.createObjectURL(b))); }).catch(() => setUrl(src));
    return () => { live = false; if (u) URL.revokeObjectURL(u); };
  }, [src, on]);
  return url;
}

export const fmtDuration = (s: number) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;
