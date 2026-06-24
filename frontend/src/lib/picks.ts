/**
 * Typed picks/slate API layer for the My Picks page.
 *
 * Mirrors the small, one-function-per-endpoint style of lib/currentWeek.ts and
 * lib/config.ts: every call routes through the CSRF-aware `api<T>()` wrapper
 * (credentials:include + X-CSRF-Token on unsafe methods) — never a raw fetch.
 *
 * Wire shapes match the backend exactly (see backend/app/schemas/slate.py and
 * backend/app/schemas/picks.py). NOTE: Decimal line fields (spread/total)
 * serialize to JSON as strings, so they are typed `string | null` here.
 */
import { api } from "./api";

/** The four base pick types (literal union mirroring backend PickType). */
export type PickType = "UNDERDOG_COVER" | "FAVORITE_COVER" | "OVER" | "UNDER";

/** Grading state of a persisted pick (mirrors backend PickResult). */
export type PickResult = "PENDING" | "WIN" | "LOSS";

/** Public team reference identity for one side of a matchup. */
export interface SlateTeam {
  team_id: number;
  abbreviation: string;
  display_name: string;
}

/**
 * One game's pickable slate entry: identity, line, lock + per-type eligibility.
 * `spread`/`total` are JSON strings (backend Decimal) — keep them as strings.
 */
export interface SlateGame {
  game_id: number;
  kickoff_at: string | null;
  home_team: SlateTeam;
  away_team: SlateTeam;
  spread: string | null;
  total: string | null;
  favorite_team_id: number | null;
  underdog_team_id: number | null;
  locked: boolean;
  eligibility: Record<PickType, boolean>;
}

/** The pickable slate for a {season, week}: one entry per game. */
export interface Slate {
  season: number;
  week: number;
  games: SlateGame[];
}

/** A persisted pick as returned to its owner (never exposes user_id). */
export interface PickRead {
  id: number;
  game_id: number;
  week_id: number;
  pick_type: PickType;
  is_mortal_lock: boolean;
  result: PickResult;
  points: number;
}

/** A single autosave item (one pick on one game). */
export interface PickItem {
  game_id: number;
  pick_type: PickType;
  is_mortal_lock: boolean;
}

/**
 * Stable string key for a pick *slot*: `(pick_type, is_mortal_lock)`. The four
 * base slots are is_mortal_lock=false (one per pick_type); the mortal lock is
 * its own slot (is_mortal_lock=true) and may duplicate a base pick_type. Used to
 * merge returned PickRead rows into the picks map.
 */
export function slotKey(pick_type: PickType, is_mortal_lock: boolean): string {
  return `${pick_type}|${is_mortal_lock}`;
}

/**
 * Stable key for an inline pick *error*, scoped to the specific GAME + slot the
 * user acted on — `(game_id, pick_type, is_mortal_lock)`. Unlike slotKey, this
 * includes game_id so a rejection (e.g. a contradiction) renders only on the
 * control that was clicked, not on every same-type button across cards.
 */
export function errorKey(
  game_id: number,
  pick_type: PickType,
  is_mortal_lock: boolean,
): string {
  return `${game_id}|${slotKey(pick_type, is_mortal_lock)}`;
}

/** Fetch the pickable slate (the OPTIONS the page renders) for a week. */
export function getSlate(season: number, week: number): Promise<Slate> {
  return api<Slate>(`/api/slate?season=${season}&week=${week}`);
}

/** Fetch the current user's authoritative roster for a week. */
export function getMyPicks(season: number, week: number): Promise<PickRead[]> {
  return api<PickRead[]>(`/api/picks?season=${season}&week=${week}`);
}

/**
 * Autosave a SINGLE pick item. The endpoint accepts an array, but the page
 * commits one pick at a time (each selection fires its own POST). Returns only
 * the AFFECTED rows (not the whole roster). Content-Type + X-CSRF-Token are
 * handled by api().
 */
export function submitPick(
  season: number,
  week: number,
  item: PickItem,
): Promise<PickRead[]> {
  return api<PickRead[]>("/api/picks", {
    method: "POST",
    body: JSON.stringify({ season, week, picks: [item] }),
  });
}
