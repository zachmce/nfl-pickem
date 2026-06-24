/** Current-week context-bar read (GET /api/current-week). */
import { api } from "./api";

export type WindowState = "not_yet_open" | "open" | "locked" | "closed";

export interface CurrentWeek {
  season: number;
  week: number;
  window_state: WindowState;
  window_closes_at: string;
}

/** Fetch the current week + its pick-window state for the context bar. */
export function getCurrentWeek(): Promise<CurrentWeek> {
  return api<CurrentWeek>("/api/current-week");
}
