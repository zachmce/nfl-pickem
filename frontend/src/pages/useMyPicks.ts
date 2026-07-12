/**
 * State hook for the My Picks page: owns all data loading and autosave state.
 *
 * Load sequence (on mount, cancel-on-unmount):
 *   1. getCurrentWeek() -> season/week + window_state.
 *   2. Promise.all(getSlate, getMyPicks) for that week.
 * Any failure lands in a graceful "error" status (never throws to the page).
 *
 * Autosave (select()) implements the 3 REQUIRED guardrails (CONTEXT.md):
 *   1. Pessimistic per-control: mark selected ONLY after a 200 (merge the
 *      returned PickRead rows by their own slotKey); never optimistic.
 *   2. Disable while in flight: saving[key] is true for the call's duration;
 *      a re-entrant call on a slot already saving is ignored.
 *   3. On error: set slotError[key] = err.message, do NOT select, and re-GET
 *      /api/picks to resync the authoritative roster. saving[key] cleared in
 *      a finally.
 *
 * clear() (un-pick) shares the SAME guardrails: it reuses the inFlightRef/saving
 * slotKey guard, removes the slot from the picks map pessimistically only after a
 * 204, treats a 404 as already-empty (silent resync, no error), and surfaces any
 * other failure inline + resyncs — exactly mirroring select()'s error branch.
 *
 * No JSX, no state library — plain useState/useEffect/useCallback.
 */
import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError } from "../lib/api";
import {
  getCurrentWeek,
  type CurrentWeek,
} from "../lib/currentWeek";
import {
  clearPick,
  errorKey,
  getMyPicks,
  getSlate,
  slotKey,
  submitPick,
  type PickItem,
  type PickRead,
  type SlateGame,
} from "../lib/picks";

export type Status = "loading" | "ok" | "error";

/** picks map keyed by slotKey(pick_type, is_mortal_lock) -> PickRead. */
export type PicksBySlot = Record<string, PickRead>;

export interface UseMyPicks {
  status: Status;
  currentWeek: CurrentWeek | null;
  slate: SlateGame[];
  /** Computed week-level odds-freeze flag from the slate (gates the re-pick notice). */
  oddsFrozen: boolean;
  /** Authoritative roster keyed by slotKey. */
  picks: PicksBySlot;
  /** True only when the pick window is open (page also freezes per-game locks). */
  editable: boolean;
  /** In-flight save state, keyed by slotKey. */
  saving: Record<string, boolean>;
  /** Inline error message keyed by errorKey(game_id, pick_type, is_mortal_lock) —
   * scoped to the specific control the user clicked, not the whole slot. */
  slotError: Record<string, string>;
  /** Autosave one pick item with the 3 guardrails. */
  select: (item: PickItem) => Promise<void>;
  /** Clear (un-pick) one slot, sharing select()'s guardrails. game_id is used
   * only to scope the inline error; the DELETE call ignores it. */
  clear: (slot: PickItem) => Promise<void>;
}

/** Build the slotKey-indexed picks map from a flat PickRead list (last-write-wins). */
function indexPicks(rows: PickRead[]): PicksBySlot {
  const map: PicksBySlot = {};
  for (const row of rows) {
    map[slotKey(row.pick_type, row.is_mortal_lock)] = row;
  }
  return map;
}

export function useMyPicks(): UseMyPicks {
  const [status, setStatus] = useState<Status>("loading");
  const [currentWeek, setCurrentWeek] = useState<CurrentWeek | null>(null);
  const [slate, setSlate] = useState<SlateGame[]>([]);
  const [oddsFrozen, setOddsFrozen] = useState<boolean>(false);
  const [picks, setPicks] = useState<PicksBySlot>({});
  const [saving, setSaving] = useState<Record<string, boolean>>({});
  const [slotError, setSlotError] = useState<Record<string, string>>({});

  // Stable view of season/week for the autosave callback (avoids stale closures
  // / re-creating select() on every load tick).
  const weekRef = useRef<{ season: number; week: number } | null>(null);
  // Synchronous in-flight guard so a double-fire within the same tick (before
  // the saving state flushes) is reliably ignored.
  const inFlightRef = useRef<Set<string>>(new Set());

  // Mount-only load (deps []). Status already starts at "loading", so an
  // in-effect setStatus("loading") would be redundant (and trips
  // react-hooks/set-state-in-effect); the .then/.catch set "ok"/"error".
  useEffect(() => {
    let cancelled = false;

    getCurrentWeek()
      .then(async (cw) => {
        if (cancelled) return;
        setCurrentWeek(cw);
        weekRef.current = { season: cw.season, week: cw.week };
        const [slateData, myPicks] = await Promise.all([
          getSlate(cw.season, cw.week),
          getMyPicks(cw.season, cw.week),
        ]);
        if (cancelled) return;
        setSlate(slateData.games);
        setOddsFrozen(slateData.odds_frozen);
        setPicks(indexPicks(myPicks));
        setStatus("ok");
      })
      .catch(() => {
        if (cancelled) return;
        setStatus("error");
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const select = useCallback(async (item: PickItem): Promise<void> => {
    const week = weekRef.current;
    if (!week) return;

    const key = slotKey(item.pick_type, item.is_mortal_lock);
    // Errors are scoped to the specific GAME + slot acted on, so a rejection
    // renders only on the clicked control (not every same-type button).
    const errKey = errorKey(item.game_id, item.pick_type, item.is_mortal_lock);

    // Guardrail 2: ignore a re-fire while this slot's save is already in flight.
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
      // Guardrail 1: mark selected ONLY after a 200 — merge returned rows by
      // their own slotKey (a base pick may evict the prior holder of a type).
      const affected = await submitPick(week.season, week.week, item);
      setPicks((prev) => {
        const next = { ...prev };
        for (const row of affected) {
          next[slotKey(row.pick_type, row.is_mortal_lock)] = row;
        }
        return next;
      });
      // A successful save changes the roster, so any prior inline errors are now
      // stale feedback — clear them all rather than letting them linger.
      setSlotError((prev) => (Object.keys(prev).length ? {} : prev));
    } catch (err) {
      // Guardrail 3: inline error, do NOT select, then resync from server truth.
      const message =
        err instanceof ApiError
          ? err.message
          : err instanceof Error
            ? err.message
            : "Could not save your pick.";
      setSlotError((prev) => ({ ...prev, [errKey]: message }));
      try {
        const fresh = await getMyPicks(week.season, week.week);
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

  const clear = useCallback(async (slot: PickItem): Promise<void> => {
    const week = weekRef.current;
    if (!week) return;

    const key = slotKey(slot.pick_type, slot.is_mortal_lock);
    // Error is scoped to the specific GAME + slot acted on (game_id is ignored
    // by the DELETE call itself).
    const errKey = errorKey(slot.game_id, slot.pick_type, slot.is_mortal_lock);

    // Guardrail 2: share the in-flight guard with select() (same slotKey space),
    // so a slot mid-save or mid-clear is ignored by both actions.
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
      // 204: slot cleared server-side — remove it from the map pessimistically
      // (only after the server confirms), then clear stale inline errors.
      await clearPick(week.season, week.week, slot);
      setPicks((prev) => {
        if (!(key in prev)) return prev;
        const next = { ...prev };
        delete next[key];
        return next;
      });
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
              : "Could not clear your pick.";
        setSlotError((prev) => ({ ...prev, [errKey]: message }));
      }
      try {
        const fresh = await getMyPicks(week.season, week.week);
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

  const editable = currentWeek?.window_state === "open";

  return {
    status,
    currentWeek,
    slate,
    oddsFrozen,
    picks,
    editable,
    saving,
    slotError,
    select,
    clear,
  };
}
