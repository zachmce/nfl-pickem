/**
 * Typed client for the /api/results read endpoints (season scoreboard + weekly).
 *
 * Mirrors the small, one-function-per-endpoint style of lib/picks.ts: every call
 * routes through the CSRF-aware `api<T>()` wrapper (credentials:include +
 * X-CSRF-Token on unsafe methods) — never a raw fetch. These are read-only GETs,
 * so no CSRF token is attached, but they still go through api() for the
 * credentialed fetch + ApiError handling.
 *
 * Wire shapes match the backend exactly (see backend/app/schemas/results.py).
 * The Weekly screen will later extend this file with `getWeekResults`.
 */
import { api } from "./api";

/**
 * One user's cumulative season standing row.
 *
 * `weekly_scores` maps a week number to that week's integer score. The KEYS are
 * STRINGS, not numbers: the backend type is `dict[int, int]`
 * (backend/app/schemas/results.py SeasonStandingRow), but JSON object keys are
 * always strings, so week numbers serialize as string keys (`"1"`, `"2"`, …).
 * Read a cell with `weekly_scores[String(week)]`.
 */
export interface SeasonStandingRow {
  display_name: string;
  season_total: number;
  weekly_scores: Record<string, number>;
}

/**
 * Cumulative season standings over all users, PRE-ORDERED by the backend
 * service (`(-season_total, display_name)`) — render rows in this exact order;
 * do not re-sort client-side.
 */
export interface SeasonStandingsResponse {
  season: number;
  standings: SeasonStandingRow[];
}

/** Fetch the season scoreboard (one row per player) for a season. */
export function getStandings(season: number): Promise<SeasonStandingsResponse> {
  return api<SeasonStandingsResponse>(
    `/api/results/standings?season=${season}`,
  );
}
