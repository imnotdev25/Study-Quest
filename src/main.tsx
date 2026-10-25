import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "katex/dist/katex.min.css";
import "highlight.js/styles/atom-one-dark.css";
import "./styles.css";
import "./lib/platform";

// Excalidraw loads its fonts from here (copied into public/fonts by scripts/copy-excalidraw-assets.mjs).
(window as unknown as { EXCALIDRAW_ASSET_PATH: string }).EXCALIDRAW_ASSET_PATH = "/";
import { applyTheme } from "./lib/theme";
import { getState } from "./lib/store";

// Apply the saved theme before first paint, so dark mode never flashes white.
applyTheme(getState().settings.themeMode, getState().settings.accent);
import App from "./App";

createRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>);
