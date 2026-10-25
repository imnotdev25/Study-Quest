// Light / dark / system appearance and the accent colour, applied to <html>.
import { useEffect } from "react";

export type ThemeMode = "system" | "light" | "dark";

export const ACCENTS: { id: string; name: string; color: string }[] = [
  { id: "violet", name: "Violet", color: "#7c5cf6" },
  { id: "blue", name: "Blue", color: "#2f7bf6" },
  { id: "teal", name: "Teal", color: "#0f9f8f" },
  { id: "green", name: "Green", color: "#16a34a" },
  { id: "amber", name: "Amber", color: "#d97706" },
  { id: "rose", name: "Rose", color: "#e11d48" },
  { id: "pink", name: "Pink", color: "#db2777" },
  { id: "slate", name: "Graphite", color: "#475569" },
];

const dark = () => window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;

export function applyTheme(mode: ThemeMode, accent: string) {
  const root = document.documentElement;
  root.dataset.theme = mode === "system" ? (dark() ? "dark" : "light") : mode;
  root.style.setProperty("--accent", ACCENTS.find((a) => a.id === accent)?.color ?? accent ?? ACCENTS[0].color);
}

/** Keep <html> in sync with settings, and with the OS when the mode is "system". */
export function useTheme(mode: ThemeMode, accent: string) {
  useEffect(() => {
    applyTheme(mode, accent);
    if (mode !== "system") return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const on = () => applyTheme(mode, accent);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, [mode, accent]);
}
