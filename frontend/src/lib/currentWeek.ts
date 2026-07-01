/** Current-week context-bar read (GET /api/current-week). */
import { api } from "./api";

export type WindowState = "not_yet_open" | "open" | "locked" | "closed";

export interface CurrentWeek {
  season: number;
  week: number;
  window_state: WindowState;
  window_closes_at: string;
  /**
   * True ONLY when every game in the season is FINAL (mirrors the backend
   * CurrentWeekResponse.season_complete and SeasonStandingsResponse.season_complete
   * in ./results). The ContextBar uses this to render the season-over copy with
   * no "closes" clause; false for an in-progress season and a zero-game season.
   */
  season_complete: boolean;
  /**
   * The COMPUTED week-level freeze predicate, mirrored from the backend
   * CurrentWeekResponse.odds_frozen and the SlateResponse.odds_frozen in
   * ./results/slate. The ContextBar renders "lines locked" when true / "lines
   * live" when false on the in-progress line only (never in season-complete,
   * loading, or error states, nor in the compact WeekChip).
   */
  odds_frozen: boolean;
}

/** Fetch the current week + its pick-window state for the context bar. */
export function getCurrentWeek(): Promise<CurrentWeek> {
  return api<CurrentWeek>("/api/current-week");
}
