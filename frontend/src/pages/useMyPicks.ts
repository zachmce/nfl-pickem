/**
 * State hook for the My Picks page: owns all data loading and autosave state.
 *
 * Load sequence (on mount, cancel-on-unmount like ContextBar.useCurrentWeek):
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
 * No JSX, no state library — plain useState/useEffect/useCallback.
 */
import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError } from "../lib/api";
import {
  getCurrentWeek,
  type CurrentWeek,
} from "../lib/currentWeek";
import {
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
  /** Authoritative roster keyed by slotKey. */
  picks: PicksBySlot;
  /** True only when the pick window is open (page also freezes per-game locks). */
  editable: boolean;
  /** In-flight save state, keyed by slotKey. */
  saving: Record<string, boolean>;
  /** Inline error message for a slot, keyed by slotKey. */
  slotError: Record<string, string>;
  /** Autosave one pick item with the 3 guardrails. */
  select: (item: PickItem) => Promise<void>;
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
  const [picks, setPicks] = useState<PicksBySlot>({});
  const [saving, setSaving] = useState<Record<string, boolean>>({});
  const [slotError, setSlotError] = useState<Record<string, string>>({});

  // Stable view of season/week for the autosave callback (avoids stale closures
  // / re-creating select() on every load tick).
  const weekRef = useRef<{ season: number; week: number } | null>(null);
  // Synchronous in-flight guard so a double-fire within the same tick (before
  // the saving state flushes) is reliably ignored.
  const inFlightRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    let cancelled = false;
    setStatus("loading");

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

    // Guardrail 2: ignore a re-fire while this slot's save is already in flight.
    if (inFlightRef.current.has(key)) return;
    inFlightRef.current.add(key);

    setSaving((prev) => ({ ...prev, [key]: true }));
    setSlotError((prev) => {
      if (!(key in prev)) return prev;
      const next = { ...prev };
      delete next[key];
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
    } catch (err) {
      // Guardrail 3: inline error, do NOT select, then resync from server truth.
      const message =
        err instanceof ApiError
          ? err.message
          : err instanceof Error
            ? err.message
            : "Could not save your pick.";
      setSlotError((prev) => ({ ...prev, [key]: message }));
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

  const editable = currentWeek?.window_state === "open";

  return {
    status,
    currentWeek,
    slate,
    picks,
    editable,
    saving,
    slotError,
    select,
  };
}
