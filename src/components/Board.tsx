// An Excalidraw drawing board. Scenes are saved to localStorage per board id, and can be exported as .excalidraw.
// Loaded lazily (it's a large bundle) by Diagrams.tsx and the "Design it" card in System Design lessons.
import { useRef } from "react";
import { Excalidraw, serializeAsJSON } from "@excalidraw/excalidraw";
import "@excalidraw/excalidraw/index.css";
import { boardKey, touchBoard } from "../lib/boards";

export default function Board({ id, title, height = 600 }: { id: string; title: string; height?: number | string }) {
  const theme = document.documentElement.dataset.theme === "dark" ? "dark" : "light";
  const timer = useRef<number | undefined>(undefined);
  let initial: Record<string, unknown> | undefined;
  try {
    const raw = localStorage.getItem(boardKey(id));
    if (raw) initial = JSON.parse(raw);
  } catch { /* corrupted scene: start empty */ }

  return (
    <div className="board" style={{ height }}>
      <Excalidraw
        theme={theme}
        initialData={initial ?? { appState: { viewBackgroundColor: theme === "dark" ? "#121212" : "#ffffff" } }}
        onChange={(elements, appState, files) => {
          window.clearTimeout(timer.current);
          timer.current = window.setTimeout(() => {
            localStorage.setItem(boardKey(id), serializeAsJSON(elements, appState, files, "local"));
            touchBoard(id, title, elements.filter((e) => !e.isDeleted).length);
          }, 500);
        }}
      />
    </div>
  );
}
