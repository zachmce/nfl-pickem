/**
 * Typed admin user-management API layer for the Admin (player-management) page.
 *
 * Mirrors the small, one-function-per-endpoint style of lib/picks.ts: every call
 * routes through the CSRF-aware `api<T>()` wrapper (credentials:include +
 * X-CSRF-Token on unsafe methods) — never a raw fetch. The acting admin is
 * resolved server-side from the session, so these calls carry NO caller/actor
 * field; the target is the path param only (no IDOR surface).
 *
 * Wire shapes mirror the backend exactly (see backend/app/schemas/admin.py and
 * backend/app/api/admin.py). The four POST mutations and the list element are all
 * `AdminUserRead`; DELETE returns 204 (void). Note that the list endpoint wraps
 * its rows in an envelope `{ users: AdminUserRead[] }` (AdminUserListResponse) —
 * it is NOT a bare array, unlike /api/picks — so listUsers() unwraps `.users`.
 *
 * 4xx error envelope: the backend emits `{ error: { code, message, reason } }`
 * (last_admin / cannot_act_on_self / user_not_found / already_* / not_admin).
 * api() already unwraps that into `ApiError.message`, so callers just surface
 * `err.message`. Do NOT modify lib/api.ts and do NOT add a new dependency.
 */
import { api } from "./api";
import type { PickRead, PickResult, PickType } from "./picks";

/**
 * One user as seen by an admin (mirrors backend AdminUserRead field-for-field).
 * `created_at` is an ISO datetime serialized as a string on the wire;
 * `discord_id` is null for web-origin accounts (non-null => Discord origin).
 */
export interface AdminUser {
  id: number;
  display_name: string;
  discord_id: number | null;
  is_admin: boolean;
  is_active: boolean;
  created_at: string;
  pick_count: number;
}

/** List every user (admin only). Unwraps the `{ users: [...] }` envelope. */
export function listUsers(): Promise<AdminUser[]> {
  return api<{ users: AdminUser[] }>("/api/admin/users").then((r) => r.users);
}

/** Deactivate another user; returns the updated target. */
export function deactivateUser(id: number): Promise<AdminUser> {
  return api<AdminUser>(`/api/admin/users/${id}/deactivate`, { method: "POST" });
}

/** Reactivate a deactivated user; returns the updated target. */
export function reactivateUser(id: number): Promise<AdminUser> {
  return api<AdminUser>(`/api/admin/users/${id}/reactivate`, { method: "POST" });
}

/** Grant admin to a user; returns the updated target. */
export function grantAdmin(id: number): Promise<AdminUser> {
  return api<AdminUser>(`/api/admin/users/${id}/grant-admin`, { method: "POST" });
}

/** Revoke admin from another user; returns the updated target. */
export function revokeAdmin(id: number): Promise<AdminUser> {
  return api<AdminUser>(`/api/admin/users/${id}/revoke-admin`, { method: "POST" });
}

/** Delete another user (their picks cascade). Backend returns 204 -> void. */
export function deleteUser(id: number): Promise<void> {
  return api<void>(`/api/admin/users/${id}`, { method: "DELETE" });
}

// --------------------------------------------------------------------------- //
// Admin pick-override API (QT-1, backend/app/api/admin.py routes 177-238)
//
// Lets an admin read/set/clear ANY user's pick for a {season, week} — past or
// upcoming — bypassing the pick window/lock but KEEPING roster integrity (the
// backend still rejects duplicate base types / contradictions / >1 mortal lock /
// ineligible types). Acting on another user is the whole point (off-window
// convenience for a small group): the caller is the SESSION admin and the target
// is the path {user_id} — there is NO caller/actor field on the wire, so this
// introduces no IDOR surface; the routes are require_admin-gated server-side.
//
// season/week are QUERY params on every call (never body), mirroring the
// user-facing picks DELETE shape in lib/picks.ts. The 4xx envelope (409
// roster-conflict / 422 eligibility / 404 not-found) is already unwrapped into
// ApiError.message by api(), so callers just surface `err.message` inline.
//
// Wire types (PickType, PickRead) are reused from lib/picks.ts — PickRead is
// field-identical to backend schemas/picks.py PickRead — so the read contract
// lives in exactly one place. Do NOT redefine PickRead here.
// --------------------------------------------------------------------------- //

/**
 * Body for an admin set/add/change of one slot (mirrors backend
 * AdminPickSetRequest; ConfigDict extra="forbid" — send ONLY these three keys).
 * The slot's {season, week} are query params, not body fields.
 */
export interface AdminPickSet {
  game_id: number;
  pick_type: PickType;
  is_mortal_lock: boolean;
  /**
   * The free-text prediction on a RETROACTIVE admin create of a MISC pick
   * (mirrors backend AdminPickSetRequest.misc_text). Send ONLY when creating a
   * MISC pick; OMIT it otherwise (extra="forbid" tolerates an absent key, but a
   * non-MISC pick must never carry text).
   */
  misc_text?: string | null;
}

/**
 * List the TARGET user's roster for a {season, week} (admin only).
 * GET /api/admin/users/{userId}/picks?season&week — the response is a BARE
 * PickRead[] (response_model=list[PickRead]), NOT an envelope unlike listUsers().
 */
export function getUserPicks(
  userId: number,
  season: number,
  week: number,
): Promise<PickRead[]> {
  return api<PickRead[]>(
    `/api/admin/users/${userId}/picks?season=${season}&week=${week}`,
  );
}

/**
 * Set/add/change the TARGET user's slot (window/lock bypassed, roster kept).
 * PUT /api/admin/users/{userId}/picks?season&week with a JSON body of ONLY
 * {game_id, pick_type, is_mortal_lock}. Returns the single updated PickRead.
 * X-CSRF-Token + Content-Type are attached by api() on the unsafe method.
 */
export function setUserPick(
  userId: number,
  season: number,
  week: number,
  body: AdminPickSet,
): Promise<PickRead> {
  return api<PickRead>(
    `/api/admin/users/${userId}/picks?season=${season}&week=${week}`,
    { method: "PUT", body: JSON.stringify(body) },
  );
}

/**
 * Clear the TARGET user's {pick_type, lock} slot (window/lock bypassed).
 * DELETE /api/admin/users/{userId}/picks with season, week, pick_type, and
 * is_mortal_lock ALL as query params (mirrors clearPick in lib/picks.ts;
 * is_mortal_lock serializes as the literal true/false). Backend returns 204 ->
 * api() yields undefined -> void.
 */
export function clearUserPick(
  userId: number,
  season: number,
  week: number,
  slot: { pick_type: PickType; is_mortal_lock: boolean },
): Promise<void> {
  const query =
    `season=${season}&week=${week}` +
    `&pick_type=${slot.pick_type}&is_mortal_lock=${slot.is_mortal_lock}`;
  return api<void>(`/api/admin/users/${userId}/picks?${query}`, {
    method: "DELETE",
  });
}

/**
 * Body for an admin grade of a user's MISC pick (mirrors backend
 * AdminMiscGradeRequest; ConfigDict extra="forbid"). `result` MUST be WIN or
 * LOSS: grading must DECIDE the pick, so PENDING is rejected server-side
 * (`misc_grade_must_decide`) — the UI must never submit PENDING. `points` is a
 * plain admin-decided int and MAY be negative (no schema constraint).
 */
export interface AdminMiscGrade {
  result: PickResult;
  points: number;
}

/**
 * Grade the TARGET user's MISC pick correct/incorrect + set points.
 * PUT /api/admin/users/{userId}/picks/misc-grade?season&week with a JSON body of
 * {result, points}. Returns the updated PickRead (the graded pick). The grade is
 * authoritative for MISC (the recompute-on-read scoring engine passes the stored
 * result/points through verbatim), so the points appear on Weekly/Standings on
 * the next read. X-CSRF-Token + Content-Type are attached by api() on the PUT.
 */
export function gradeMisc(
  userId: number,
  season: number,
  week: number,
  body: AdminMiscGrade,
): Promise<PickRead> {
  return api<PickRead>(
    `/api/admin/users/${userId}/picks/misc-grade?season=${season}&week=${week}`,
    { method: "PUT", body: JSON.stringify(body) },
  );
}

// --------------------------------------------------------------------------- //
// Admin worker-trigger API (QT-3 — frontend half of the live-ingest-worker thread)
//
// Two admin-only POSTs that DISPATCH a Celery background task and return
// immediately with HTTP 202 + a task_id. The 202 means the work was
// *accepted/queued*, NOT that ingest/freeze has completed — there is no
// status/progress endpoint, so callers must render the task_id as a
// background-dispatch confirmation (never as a done state).
//
// Both route through the CSRF-aware api<T>() (X-CSRF-Token attached on the unsafe
// POST) — never a raw fetch — and carry NO actor field (the admin is resolved
// server-side from the session; routes are require_admin-gated: 401 anon /
// 403 non-admin). The 4xx/403 envelope is already unwrapped into ApiError.message
// by api(), so callers just surface `err.message`. Wire shapes mirror the frozen
// backend 202 bodies (backend/app/api/admin.py ingest-season / freeze-week)
// field-for-field. Do NOT modify lib/api.ts and do NOT add a new dependency.
// --------------------------------------------------------------------------- //

/**
 * The 202 body of POST /api/admin/ingest-season (mirrors the backend return
 * `{ task_id, season }` field-for-field). `task_id` is the dispatched Celery task
 * id — accepted/queued, NOT a completion signal.
 */
export interface IngestSeasonDispatch {
  task_id: string;
  season: number;
}

/**
 * The 202 body of POST /api/admin/freeze-week (mirrors the backend return
 * `{ task_id, season, week }` field-for-field). `task_id` is the dispatched
 * Celery task id — accepted/queued, NOT a completion signal.
 */
export interface FreezeWeekDispatch {
  task_id: string;
  season: number;
  week: number;
}

/**
 * Dispatch a season-ingest background task (admin only). POSTs JSON `{ season }`
 * and resolves the 202 `{ task_id, season }` — the work is queued, not done.
 * Rejects with ApiError (carrying .status + .message) on 4xx/403.
 */
export function ingestSeason(season: number): Promise<IngestSeasonDispatch> {
  return api<IngestSeasonDispatch>("/api/admin/ingest-season", {
    method: "POST",
    body: JSON.stringify({ season }),
  });
}

/**
 * Dispatch a week-line-freeze background task (admin only). POSTs JSON
 * `{ season, week }` and resolves the 202 `{ task_id, season, week }` — the work
 * is queued, not done. Rejects with ApiError (.status + .message) on 4xx/403.
 */
export function freezeWeek(
  season: number,
  week: number,
): Promise<FreezeWeekDispatch> {
  return api<FreezeWeekDispatch>("/api/admin/freeze-week", {
    method: "POST",
    body: JSON.stringify({ season, week }),
  });
}
