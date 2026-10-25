// Which OS the app is running on, and the words/keys that differ between them.
const ua = typeof navigator === "undefined" ? "" : navigator.userAgent;

export const OS: "mac" | "windows" | "linux" = /Mac/i.test(ua) ? "mac" : /Win/i.test(ua) ? "windows" : "linux";
export const isMac = OS === "mac";

/** Modifier key label: ⌘ on macOS, Ctrl elsewhere. */
export const MOD = isMac ? "⌘" : "Ctrl+";
export const FILE_MANAGER = isMac ? "Finder" : OS === "windows" ? "Explorer" : "file manager";
export const TRASH = OS === "windows" ? "Recycle Bin" : "Trash";
export const TRAY = isMac ? "menu bar" : "system tray";

/** How to dictate into any text field when the web view has no speech recognizer. */
export const DICTATION = isMac
  ? { title: "🎙 Use macOS Dictation", body: "The cursor is in the box: press 🌐 (or Fn) twice and talk. Turn it on in System Settings → Keyboard → Dictation." }
  : OS === "windows"
  ? { title: "🎙 Use Windows voice typing", body: "The cursor is in the box: press Win + H and talk." }
  : { title: "🎙 Voice typing", body: "This Linux web view has no speech recognizer. Use your desktop's dictation tool (e.g. Speech Note or nerd-dictation), with the cursor in the box." };

// Lets CSS adapt (macOS gets the translucent sidebar; others get a solid one).
if (typeof document !== "undefined") document.documentElement.classList.add(`os-${OS}`);
