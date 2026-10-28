// Small built-in line-icon set (24×24, stroke = currentColor), plus a progress ring. No icon dependency needed.
import type { CSSProperties, ReactNode } from "react";

const P: Record<string, ReactNode> = {
  home: <><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" /><path d="M9 22V12h6v10" /></>,
  search: <><circle cx="11" cy="11" r="8" /><path d="m21 21-4.3-4.3" /></>,
  flask: <><path d="M9 3h6" /><path d="M10 3v6L4.5 19a1.5 1.5 0 0 0 1.3 2h12.4a1.5 1.5 0 0 0 1.3-2L14 9V3" /><path d="M7 15h10" /></>,
  notebook: <><path d="M2 4h6a4 4 0 0 1 4 4v13a3 3 0 0 0-3-3H2z" /><path d="M22 4h-6a4 4 0 0 0-4 4v13a3 3 0 0 1 3-3h7z" /></>,
  gamepad: <><rect x="2" y="6" width="20" height="12" rx="5" /><path d="M6 12h4" /><path d="M8 10v4" /><circle cx="15.5" cy="13" r=".6" /><circle cx="17.5" cy="11" r=".6" /></>,
  calendar: <><rect x="3" y="4" width="18" height="18" rx="2" /><path d="M16 2v4" /><path d="M8 2v4" /><path d="M3 10h18" /></>,
  sliders: <><path d="M21 4h-7" /><path d="M10 4H3" /><path d="M21 12h-9" /><path d="M8 12H3" /><path d="M21 20h-5" /><path d="M12 20H3" /><path d="M14 2v4" /><path d="M8 10v4" /><path d="M16 18v4" /></>,
  target: <><circle cx="12" cy="12" r="10" /><circle cx="12" cy="12" r="6" /><circle cx="12" cy="12" r="2" /></>,
  cpu: <><rect x="4" y="4" width="16" height="16" rx="2" /><rect x="9" y="9" width="6" height="6" rx="1" /><path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3" /></>,
  layers: <><path d="m12 2 10 5-10 5L2 7z" /><path d="m2 17 10 5 10-5" /><path d="m2 12 10 5 10-5" /></>,
  shield: <><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /><path d="m9 12 2 2 4-4" /></>,
  book: <><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" /><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" /></>,
  chevron: <path d="m9 18 6-6-6-6" />,
  check: <path d="M20 6 9 17l-5-5" />,
  play: <path d="M7 4v16l13-8z" />,
  arrow: <><path d="M5 12h14" /><path d="m12 5 7 7-7 7" /></>,
  flame: <path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.4-.5-2-1-3-1.1-2.1-.2-4 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.2.4-2.3 1-3a2.5 2.5 0 0 0 2.5 2.5z" />,
  film: <><rect x="2" y="3" width="20" height="18" rx="2" /><path d="M7 3v18M17 3v18M2 9h5M2 15h5M17 9h5M17 15h5" /></>,
  pen: <><path d="M12 20h9" /><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z" /></>,
  sparkle: <path d="M12 3l1.9 5.6L19.5 10.5 13.9 12.4 12 18l-1.9-5.6L4.5 10.5l5.6-1.9z" />,
};
export type IconName = keyof typeof P;

export function Icon({ name, size = 16, stroke = 2, style, className }: { name: IconName; size?: number; stroke?: number;
  style?: CSSProperties; className?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={stroke}
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className={`icon ${className ?? ""}`} style={style}>
      {P[name]}
    </svg>
  );
}

/** Circular progress, 0..1. Shows a check when complete. */
export function Ring({ value, size = 36, stroke = 4, color = "var(--accent)", label }: { value: number; size?: number;
  stroke?: number; color?: string; label?: ReactNode }) {
  const r = (size - stroke) / 2, c = 2 * Math.PI * r, v = Math.max(0, Math.min(1, value));
  return (
    <span className="ring" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden="true">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--line)" strokeWidth={stroke} />
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth={stroke} strokeLinecap="round"
          strokeDasharray={`${c * v} ${c}`} transform={`rotate(-90 ${size / 2} ${size / 2})`} style={{ transition: "stroke-dasharray .4s" }} />
      </svg>
      <span className="ring-label">{v >= 1 ? <Icon name="check" size={size * 0.42} stroke={3} style={{ color }} /> : label}</span>
    </span>
  );
}

/** Topic status dot for the timeline: done, started, or not started. */
export function StatusDot({ state }: { state: "done" | "started" | "todo" }) {
  return (
    <span className={`dot dot-${state}`} aria-label={state === "done" ? "done" : state === "started" ? "started" : "not started"}>
      {state === "done" && <Icon name="check" size={10} stroke={3.5} />}
    </span>
  );
}

/** The app mark: the same bear face as the app icon (src-tauri/icons/source.png), drawn as SVG. */
export function Logo({ size = 26 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 1024 1024" aria-hidden="true" className="logo-svg">
      <defs>
        <linearGradient id="sq-logo-bg" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" style={{ stopColor: "var(--accent)" }} /><stop offset="1" style={{ stopColor: "color-mix(in srgb, var(--accent) 75%, black)" }} />
        </linearGradient>
      </defs>
      <rect x="40" y="40" width="944" height="944" rx="220" fill="url(#sq-logo-bg)" />
      <circle cx="300" cy="330" r="80" fill="#faf0e6" />
      <circle cx="724" cy="330" r="80" fill="#faf0e6" />
      <circle cx="512" cy="560" r="270" fill="#faf0e6" />
      <circle cx="420" cy="520" r="38" fill="#281e3c" />
      <circle cx="604" cy="520" r="38" fill="#281e3c" />
      <circle cx="512" cy="620" r="34" fill="#281e3c" />
    </svg>
  );
}
