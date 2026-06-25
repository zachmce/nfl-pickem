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
