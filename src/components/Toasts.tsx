// Byte the otter's speech bubbles: XP, badges, nudges and trolls.
import { useEffect, useState } from "react";
import { onToast, type Toast } from "../lib/store";

const FACE: Record<Toast["kind"], string> = { xp: "🦦", nudge: "🦦", badge: "🏅", praise: "🥳", info: "🦦", error: "😵" };

export function Toasts() {
  const [list, setList] = useState<Toast[]>([]);
  const [burst, setBurst] = useState(0);
  useEffect(() => onToast((t) => {
    setList((l) => [...l.slice(-3), t]);
    if (t.kind === "praise" || t.kind === "badge") setBurst(Date.now());
    const life = t.kind === "xp" ? 2600 : t.kind === "nudge" ? 9000 : 5500;
    setTimeout(() => setList((l) => l.filter((x) => x.id !== t.id)), life);
  }), []);
  return (
    <>
      <div className="toasts" aria-live="polite">
        {list.map((t) => (
          <div key={t.id} className={`toast toast-${t.kind}`} role="status" onClick={() => setList((l) => l.filter((x) => x.id !== t.id))}>
            <span className="toast-face" aria-hidden>{FACE[t.kind]}</span>
            <div><div className="toast-title">{t.title}</div>{t.body && <div className="toast-body">{t.body}</div>}</div>
          </div>
        ))}
      </div>
      {burst > 0 && <Confetti key={burst} />}
    </>
  );
}

function Confetti() {
  const bits = Array.from({ length: 28 }, (_, i) => i);
  const emoji = ["🎉", "✨", "⭐", "🦦", "🔥", "💜"];
  return (
    <div className="confetti" aria-hidden>
      {bits.map((i) => (
        <span key={i} style={{ left: `${(i * 37) % 100}%`, animationDelay: `${(i % 7) * 0.08}s`,
                               fontSize: `${14 + (i % 4) * 5}px` }}>{emoji[i % emoji.length]}</span>
      ))}
    </div>
  );
}
