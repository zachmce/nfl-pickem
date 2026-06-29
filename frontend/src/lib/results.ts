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
import type { PickType } from "./picks";

/**
 * One graded pick within a user's week (mirrors backend
 * WeekResultPickRead in backend/app/schemas/results.py). `outcome` is the
 * GradeOutcome string value: WIN | LOSS | PUSH | INELIGIBLE | UNGRADEABLE
 * (UNGRADEABLE is the neutral "not yet scored" state for not-yet-FINAL games).
 * `pick_type` reuses the PickType union from ./picks (do NOT redefine it).
 */
export interface WeekResultPickRead {
  game_id: number;
  pick_type: PickType;
  is_mortal_lock: boolean;
  outcome: string;
  points: number;
  /**
   * The free-text prediction for a MISC pick (mirrors backend
   * WeekResultPickRead.misc_text). Present ONLY on a REVEALED entry — the server
   * already redacts other users' MISC text until the tied game locks, so an
   * unrevealed/other-user MISC simply arrives with this null/absent (or the whole
   * entry omitted). Null/absent for every non-MISC pick.
   */
  misc_text?: string | null;
}

/**
 * One user's graded picks + weekly score for a {season, week} (mirrors backend
 * UserWeekResult). `user_id` is deliberately ABSENT — display_name only.
 *
 * NOTE: OTHER users' picks on not-yet-locked games are OMITTED server-side
 * (the leak gate), so `picks` may contain fewer entries than the user actually
 * made; `weekly_score` is always whole.
 */
export interface UserWeekResult {
  display_name: string;
  weekly_score: number;
  picks: WeekResultPickRead[];
  /**
   * Discord snowflake id for this user as a STRING (mirrors the backend
   * UserWeekResult additions; a 64-bit id loses precision as a JSON number).
   * Null when the user has no Discord identity; paired with
   * `discord_avatar_hash` to build the avatar CDN URL (else initials fallback).
   */
  discord_id: string | null;
  /**
   * Discord avatar hash (mirrors the backend UserWeekResult addition). Null when
   * the user has no custom avatar; the `<Avatar>` component falls back to
   * initials when this (or `discord_id`) is null.
   */
  discord_avatar_hash: string | null;
}

/**
 * Per-week graded results across all users (mirrors backend
 * WeekResultsResponse). `results` is PRE-ORDERED by the backend
 * (`(-weekly_score, display_name)`) — render in this order, do NOT re-sort.
 */
export interface WeekResultsResponse {
  season: number;
  week: number;
  results: UserWeekResult[];
}

/** Fetch the per-week graded results (all users) for a {season, week}. */
export function getWeekResults(
  season: number,
  week: number,
): Promise<WeekResultsResponse> {
  return api<WeekResultsResponse>(
    `/api/results/week?season=${season}&week=${week}`,
  );
}

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
  /**
   * Discord snowflake id for this user as a STRING (mirrors the backend
   * SeasonStandingRow additions; a 64-bit id loses precision as a JSON number).
   * Null when the user has no Discord identity; paired with
   * `discord_avatar_hash` to build the avatar CDN URL (else initials fallback).
   */
  discord_id: string | null;
  /**
   * Discord avatar hash (mirrors the backend SeasonStandingRow addition). Null
   * when the user has no custom avatar; the `<Avatar>` component falls back to
   * initials when this (or `discord_id`) is null.
   */
  discord_avatar_hash: string | null;
}

/**
 * Cumulative season standings over all users, PRE-ORDERED by the backend
 * service (`(-season_total, display_name)`) — render rows in this exact order;
 * do not re-sort client-side.
 */
export interface SeasonStandingsResponse {
  season: number;
  standings: SeasonStandingRow[];
  /**
   * True ONLY when every game in the season is FINAL (mirrors the backend
   * SeasonStandingsResponse.season_complete). The Standings page uses this to
   * award 1st/2nd/3rd medals; it is false for an in-progress season and for a
   * season with zero games.
   */
  season_complete: boolean;
}

/** Fetch the season scoreboard (one row per player) for a season. */
export function getStandings(season: number): Promise<SeasonStandingsResponse> {
  return api<SeasonStandingsResponse>(
    `/api/results/standings?season=${season}`,
  );
}
