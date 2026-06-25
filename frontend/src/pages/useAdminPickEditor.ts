/**
 * State hook for the admin per-user pick-override editor (QT-2).
 *
 * Modeled on useMyPicks, but operates on a TARGET user_id and an ADMIN-chosen
 * {season, week} (NOT getCurrentWeek): the editor panel owns the selection and
 * passes userId/season/week in; the load effect re-runs whenever any of them
 * change.
 *
 * Load sequence (cancel-on-unmount like useMyPicks):
 *   Promise.all(getSlate(season, week), getUserPicks(userId, season, week)).
 *   The slate is NOT user-scoped, so the admin sees the same week's options,
 *   eligibility, lock state, and lines; the picks are the TARGET user's roster.
 *   Any failure lands in a graceful "error" status (never throws to the page).
 *
 * Mutations re-GET the target roster after every action (server truth — a base
 * change can evict the prior holder of that type), share an in-flight + saving
 * guard across set/clear, surface non-404 4xx inline (scoped to the acted
 * control), and treat a clear-404 as already-empty (silent resync, mirroring
 * useMyPicks D-01). The roster is NEVER mutated optimistically on failure.
 *
 * CRITICAL difference from useMyPicks: there is NO window/lock gating. This hook
 * has no `editable`/window concept and MUST NOT freeze controls by game.locked —
 * the admin override deliberately bypasses the pick lock (the whole point). The
 * ONLY disable is the per-slot in-flight `saving` guard. Roster integrity
 * (duplicate base type / contradiction / >1 mortal lock / eligibility) is
 * enforced SERVER-side and surfaces as inline 4xx; the UI does not pre-block it.
 *
 * No JSX, no state library — plain useState/useEffect/useCallback/useRef.
 */
import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError } from "../lib/api";
import {
  clearUserPick,
  getUserPicks,
  setUserPick,
  type AdminPickSet,
} from "../lib/admin";
import {
  errorKey,
  getSlate,
  slotKey,
  type PickRead,
  type SlateGame,
} from "../lib/picks";
import type { PicksBySlot } from "./useMyPicks";

export type Status = "loading" | "ok" | "error";

export interface UseAdminPickEditor {
  status: Status;
  /** The chosen week's slate (options/eligibility/lock/lines — not user-scoped). */
  slate: SlateGame[];
  /** The TARGET user's authoritative roster keyed by slotKey. */
  picks: PicksBySlot;
  /** In-flight save state, keyed by slotKey. */
  saving: Record<string, boolean>;
  /** Inline error keyed by errorKey(game_id, pick_type, is_mortal_lock) — scoped
   * to the specific control acted on, not the whole slot. */
  slotError: Record<string, string>;
  /** Set/add/change one slot for the target user (window/lock bypassed). */
  set: (item: AdminPickSet) => Promise<void>;
  /** Clear one slot for the target user; game_id only scopes the inline error. */
  clear: (slot: AdminPickSet) => Promise<void>;
}

/** Build the slotKey-indexed picks map from a flat PickRead list (last-write-wins). */
function indexPicks(rows: PickRead[]): PicksBySlot {
  const map: PicksBySlot = {};
  for (const row of rows) {
    map[slotKey(row.pick_type, row.is_mortal_lock)] = row;
  }
  return map;
}

export function useAdminPickEditor(
  userId: number,
  season: number,
  week: number,
): UseAdminPickEditor {
  const [status, setStatus] = useState<Status>("loading");
  const [slate, setSlate] = useState<SlateGame[]>([]);
  const [picks, setPicks] = useState<PicksBySlot>({});
  const [saving, setSaving] = useState<Record<string, boolean>>({});
  const [slotError, setSlotError] = useState<Record<string, string>>({});

  // Stable view of the target/week for the mutation callbacks (avoids stale
  // closures and re-creating set()/clear() on every load tick).
  const targetRef = useRef<{ userId: number; season: number; week: number }>({
    userId,
    season,
    week,
  });
  targetRef.current = { userId, season, week };

  // Synchronous in-flight guard so a double-fire within the same tick (before
  // the saving state flushes) is reliably ignored — shared by set() and clear().
  const inFlightRef = useRef<Set<string>>(new Set());

  // Re-load whenever the target user OR the chosen season/week changes.
  useEffect(() => {
    let cancelled = false;
    setStatus("loading");
    setSlotError({});

    Promise.all([getSlate(season, week), getUserPicks(userId, season, week)])
      .then(([slateData, userPicks]) => {
        if (cancelled) return;
        setSlate(slateData.games);
        setPicks(indexPicks(userPicks));
        setStatus("ok");
      })
      .catch(() => {
        if (cancelled) return;
        setStatus("error");
      });

    return () => {
      cancelled = true;
    };
  }, [userId, season, week]);

  const set = useCallback(async (item: AdminPickSet): Promise<void> => {
    const { userId, season, week } = targetRef.current;

    const key = slotKey(item.pick_type, item.is_mortal_lock);
    const errKey = errorKey(item.game_id, item.pick_type, item.is_mortal_lock);

    // Ignore a re-fire while this slot's save is already in flight.
    if (inFlightRef.current.has(key)) return;
    inFlightRef.current.add(key);

    setSaving((prev) => ({ ...prev, [key]: true }));
    setSlotError((prev) => {
      if (!(errKey in prev)) return prev;
      const next = { ...prev };
      delete next[errKey];
      return next;
    });

    try {
      // Merge the single returned row by its own slotKey, THEN re-GET the full
      // roster (a base change can evict the prior holder of that type) so the
      // editor reflects server truth — never an optimistic-only flip.
      const updated = await setUserPick(userId, season, week, item);
      setPicks((prev) => ({
        ...prev,
        [slotKey(updated.pick_type, updated.is_mortal_lock)]: updated,
      }));
      try {
        const fresh = await getUserPicks(userId, season, week);
        setPicks(indexPicks(fresh));
      } catch {
        // Resync failed — keep the merged row; no inline error on a successful set.
      }
      setSlotError((prev) => (Object.keys(prev).length ? {} : prev));
    } catch (err) {
      // Inline error, do NOT mutate the roster, then resync from server truth.
      const message =
        err instanceof ApiError
          ? err.message
          : err instanceof Error
            ? err.message
            : "Could not save this pick.";
      setSlotError((prev) => ({ ...prev, [errKey]: message }));
      try {
        const fresh = await getUserPicks(userId, season, week);
        setPicks(indexPicks(fresh));
      } catch {
        // Resync failed too — leave the existing map; the inline error stands.
      }
    } finally {
      inFlightRef.current.delete(key);
      setSaving((prev) => {
        const next = { ...prev };
        delete next[key];
        return next;
      });
    }
  }, []);

  const clear = useCallback(async (slot: AdminPickSet): Promise<void> => {
    const { userId, season, week } = targetRef.current;

    const key = slotKey(slot.pick_type, slot.is_mortal_lock);
    // game_id only scopes the inline error; the DELETE call ignores it.
    const errKey = errorKey(slot.game_id, slot.pick_type, slot.is_mortal_lock);

    // Share the in-flight guard with set() (same slotKey space) so a slot
    // mid-save or mid-clear is ignored by both actions.
    if (inFlightRef.current.has(key)) return;
    inFlightRef.current.add(key);

    setSaving((prev) => ({ ...prev, [key]: true }));
    setSlotError((prev) => {
      if (!(errKey in prev)) return prev;
      const next = { ...prev };
      delete next[errKey];
      return next;
    });

    try {
      // 204: slot cleared server-side — remove it from the map, then re-GET +
      // re-index to reflect full server truth.
      await clearUserPick(userId, season, week, {
        pick_type: slot.pick_type,
        is_mortal_lock: slot.is_mortal_lock,
      });
      setPicks((prev) => {
        if (!(key in prev)) return prev;
        const next = { ...prev };
        delete next[key];
        return next;
      });
      try {
        const fresh = await getUserPicks(userId, season, week);
        setPicks(indexPicks(fresh));
      } catch {
        // Resync failed — the optimistic removal stands; no inline error.
      }
      setSlotError((prev) => (Object.keys(prev).length ? {} : prev));
    } catch (err) {
      // D-01: a 404 means the slot is already empty server-side — not an error.
      // Any other failure surfaces inline. BOTH paths resync from server truth.
      if (!(err instanceof ApiError && err.status === 404)) {
        const message =
          err instanceof ApiError
            ? err.message
            : err instanceof Error
              ? err.message
              : "Could not clear this pick.";
        setSlotError((prev) => ({ ...prev, [errKey]: message }));
      }
      try {
        const fresh = await getUserPicks(userId, season, week);
        setPicks(indexPicks(fresh));
      } catch {
        // Resync failed too — leave the existing map; any inline error stands.
      }
    } finally {
      inFlightRef.current.delete(key);
      setSaving((prev) => {
        const next = { ...prev };
        delete next[key];
        return next;
      });
    }
  }, []);

  return { status, slate, picks, saving, slotError, set, clear };
}
