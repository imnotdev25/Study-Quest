// In-app replacement for window.confirm / window.prompt, which Tauri's web views don't show.
import { useEffect, useRef, useState } from "react";

type Req = { title: string; body?: string; input?: string; ok: string; danger?: boolean; resolve: (v: string | null) => void };
let push: ((r: Req) => void) | null = null;

/** Resolves true if confirmed. */
export const confirmDialog = (title: string, body?: string, opts: { ok?: string; danger?: boolean } = {}) =>
  new Promise<boolean>((res) => {
    if (!push) return res(false);                        // no host mounted
    push({ title, body, ok: opts.ok ?? "OK", danger: opts.danger, resolve: (v) => res(v !== null) });
  });

/** Resolves the typed text, or null if cancelled. */
export const promptDialog = (title: string, initial = "", ok = "Create") =>
  new Promise<string | null>((res) => {
    if (!push) return res(null);
    push({ title, input: initial, ok, resolve: res });
  });

export function DialogHost() {
  const [req, setReq] = useState<Req | null>(null);
  const [text, setText] = useState("");
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => { push = (r) => { setText(r.input ?? ""); setReq(r); }; return () => { push = null; }; }, []);
  useEffect(() => { if (req) setTimeout(() => (input.current ?? document.querySelector<HTMLButtonElement>(".dialog .primary, .dialog .danger-solid"))?.focus(), 30); }, [req]);
  if (!req) return null;
  const close = (v: string | null) => { req.resolve(v); setReq(null); };
  const ok = () => close(req.input !== undefined ? (text.trim() || null) : "ok");
  return (
    <div className="overlay" onClick={() => close(null)} onKeyDown={(e) => { if (e.key === "Escape") close(null); }}>
      <div className="dialog" role="dialog" aria-modal="true" aria-labelledby="dlg-title" onClick={(e) => e.stopPropagation()}>
        <h2 id="dlg-title">{req.title}</h2>
        {req.body && <p className="muted">{req.body}</p>}
        {req.input !== undefined && (
          <input ref={input} value={text} onChange={(e) => setText(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") ok(); }} />
        )}
        <div className="row dialog-actions">
          <button onClick={() => close(null)}>Cancel</button>
          <button className={req.danger ? "danger-solid" : "primary"} onClick={ok}>{req.ok}</button>
        </div>
      </div>
    </div>
  );
}
