// Thin wrappers over Tauri commands. In a plain browser (pnpm dev without Tauri) they fail with a clear message.
import { invoke } from "@tauri-apps/api/core";

export const inTauri = () => typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

export async function call<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  if (!inTauri()) throw new Error("This needs the desktop app (run `pnpm tauri dev`).");
  return invoke<T>(cmd, args);
}

export type RunOutput = { stdout: string; stderr: string; timed_out: boolean; ms: number };

export async function addCalendarEvent(title: string, notes: string, start: number, minutes: number,
                                       calendar: string, alarm: boolean) {
  return call<{ calendar: string; uid: string }>("add_calendar_event", { title, notes, startMs: start, minutes, calendar, alarm });
}

/** Removes the event (and any duplicates with the same title and start). Resolves to the number removed. */
export async function deleteCalendarEvent(uid: string, title: string, start: number, calendar: string) {
  return call<number>("delete_calendar_event", { uid, title, startMs: start, calendar });
}
