// Index of drawing boards (scenes themselves live under sq.draw.<id> in localStorage).
export type BoardMeta = { id: string; title: string; updated: number; shapes: number };

const INDEX = "sq.draw.index";
export const boardKey = (id: string) => `sq.draw.${id}`;

export function listBoards(): BoardMeta[] {
  try { return JSON.parse(localStorage.getItem(INDEX) ?? "[]"); } catch { return []; }
}

const save = (list: BoardMeta[]) => localStorage.setItem(INDEX, JSON.stringify(list));

export function touchBoard(id: string, title: string, shapes: number) {
  const list = listBoards().filter((b) => b.id !== id);
  save([{ id, title, updated: Date.now(), shapes }, ...list]);
}

export function newBoardId() {
  return `free-${Date.now().toString(36)}`;
}

export function deleteBoard(id: string) {
  localStorage.removeItem(boardKey(id));
  save(listBoards().filter((b) => b.id !== id));
}

/** System Design and DSA lessons get a design board. */
export const hasBoard = (topicId: string) => topicId.startsWith("sd:") || topicId.startsWith("dsa:");
export const topicBoardId = (topicId: string) => `topic-${topicId.replace(":", "-")}`;
