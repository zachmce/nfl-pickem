/**
 * Typed calendar API layer for the month-grid Calendar page.
 *
 * Mirrors the small, one-function-per-endpoint style of lib/picks.ts: the call
 * routes through the CSRF-aware `api<T>()` wrapper (credentials:include) — never
 * a raw fetch. Wire shapes match the backend exactly (see
 * backend/app/schemas/calendar.py).
 *
 * The endpoint is a pure date-range filter that returns each game's RAW UTC
 * `kickoff_at`; the CLIENT buckets games onto their US Eastern
 * (America/New_York) calendar day (see pages/CalendarPage.tsx).
 */
import { api } from "./api";

/** Grading/lifecycle state of a game (mirrors backend GameStatus). */
export type GameStatus = "SCHEDULED" | "IN_PROGRESS" | "FINAL";

/** Public team reference identity (abbreviation only) for one side. */
export interface CalendarTeam {
  abbreviation: string;
}

/**
 * The current user's picks on one game, folded to one value (null = no pick):
 * PENDING = none graded yet; WIN / LOSS = every graded pick agrees; SPLIT =
 * graded picks disagree.
 */
export type CalendarPickOutcome = "PENDING" | "WIN" | "LOSS" | "SPLIT";

/**
 * One game's display-only calendar entry. `kickoff_at` is a RAW UTC ISO string
 * (or null); the client buckets it onto its ET calendar day. `home_score` /
 * `away_score` are only meaningful (rendered) when `status === "FINAL"`.
 */
export interface CalendarGame {
  game_id: number;
  kickoff_at: string | null;
  home_team: CalendarTeam;
  away_team: CalendarTeam;
  status: GameStatus;
  home_score: number | null;
  away_score: number | null;
  my_pick_result: CalendarPickOutcome | null;
}

/**
 * Chip classes for a game by the current user's pick outcome (issue #301):
 * a won pick is green, a lost pick is red, any other pick is bold, and a game
 * with no pick keeps the plain chip.
 */
export function pickChipClasses(outcome: CalendarPickOutcome | null): {
  chip: string;
  text: string;
} {
  switch (outcome) {
    case "WIN":
      return {
        chip: "border-success-fg bg-success-bg",
        text: "font-bold text-success-fg",
      };
    case "LOSS":
      return {
        chip: "border-danger-fg bg-danger-bg",
        text: "font-bold text-danger-fg",
      };
    case "PENDING":
    case "SPLIT":
      return { chip: "border-border bg-surface-raised", text: "font-bold text-fg" };
    default:
      return { chip: "border-border bg-surface-raised", text: "font-medium text-fg" };
  }
}

/** The season's games whose kickoff falls in [from_date, to_date]. */
export interface CalendarResponse {
  from_date: string;
  to_date: string;
  games: CalendarGame[];
}

/**
 * Fetch the games whose kickoff falls in the inclusive `[from, to]` window.
 * Both bounds are `YYYY-MM-DD` (the `to` day is inclusive server-side).
 */
export function getCalendar(
  from: string,
  to: string,
): Promise<CalendarResponse> {
  return api<CalendarResponse>(`/api/calendar?from=${from}&to=${to}`);
}
