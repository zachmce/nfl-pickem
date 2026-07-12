/**
 * State hook for the Weekly page: a pageable, per-week view of every user's
 * graded picks. Mirrors useStandings (plain useState/useEffect, cancel-on-
 * unmount, graceful error status — never throws to the page).
 *
 * Load sequence:
 *   1. On mount, getCurrentWeek() resolves the active `season` + default `week`
 *      (which also becomes `maxWeek` for v1 — there is no week-list endpoint, so
 *      nav clamps to 1..currentWeek; prev walks back to 1).
 *   2. For the active week, load getWeekResults(season, week) and
 *      getSlate(season, week) in PARALLEL (Promise.all). The slate provides the
 *      game-identity join (matchup / favorite / line) keyed by game_id.
 *   3. Changing the week (setWeek / prev / next) re-runs the parallel load for
 *      the new week, guarded by the same cancel-on-unmount flag.
 */
import { useCallback, useEffect, useState } from "react";

import { getCurrentWeek } from "../lib/currentWeek";
import { getSlate, type SlateGame } from "../lib/picks";
import { getWeekResults, type UserWeekResult } from "../lib/results";

export type Status = "loading" | "ok" | "error";

export interface UseWeekly {
  status: Status;
  /** The resolved season once loaded, else null. */
  season: number | null;
  /** The currently displayed week. */
  week: number;
  /** Upper bound for week navigation (the current week for v1). */
  maxWeek: number;
  /** Per-user results in server order (do NOT re-sort). */
  results: UserWeekResult[];
  /** Slate games for the displayed week, indexed by game_id (the join). */
  slateByGameId: Record<number, SlateGame>;
  /** Jump to a specific week (clamped 1..maxWeek by prev/next callers). */
  setWeek: (week: number) => void;
  /** Step back a week (clamped at 1). */
  prev: () => void;
  /** Step forward a week (clamped at maxWeek). */
  next: () => void;
}

export function useWeekly(): UseWeekly {
  const [status, setStatus] = useState<Status>("loading");
  const [season, setSeason] = useState<number | null>(null);
  const [week, setWeekState] = useState<number>(1);
  const [maxWeek, setMaxWeek] = useState<number>(1);
  const [results, setResults] = useState<UserWeekResult[]>([]);
  const [slateByGameId, setSlateByGameId] = useState<
    Record<number, SlateGame>
  >({});

  // Re-show the "loading" placeholder whenever the fetched key (season+week)
  // changes, WITHOUT calling setState inside the fetch effect. This is React's
  // endorsed "adjust state during render on a key change" pattern: the key is
  // computed only once season resolves, so season-resolution AND any subsequent
  // week change both re-enter "loading" — byte-for-byte the prior UX.
  const fetchKey = season === null ? null : `${season}:${week}`;
  const [loadedKey, setLoadedKey] = useState<string | null>(null);
  if (fetchKey !== null && fetchKey !== loadedKey) {
    setLoadedKey(fetchKey);
    setStatus("loading");
  }

  // Resolve season + default week once on mount. Status already starts at
  // "loading" (initial state), so no in-effect setStatus is needed here.
  useEffect(() => {
    let cancelled = false;

    getCurrentWeek()
      .then((cw) => {
        if (cancelled) return;
        setSeason(cw.season);
        setMaxWeek(cw.week);
        setWeekState(cw.week);
      })
      .catch(() => {
        if (cancelled) return;
        setStatus("error");
      });

    return () => {
      cancelled = true;
    };
  }, []);

  // Load results + slate (in parallel) for the active {season, week}.
  useEffect(() => {
    if (season === null) return;
    let cancelled = false;

    Promise.all([getWeekResults(season, week), getSlate(season, week)])
      .then(([weekResults, slate]) => {
        if (cancelled) return;
        const byId: Record<number, SlateGame> = {};
        for (const g of slate.games) byId[g.game_id] = g;
        setResults(weekResults.results);
        setSlateByGameId(byId);
        setStatus("ok");
      })
      .catch(() => {
        if (cancelled) return;
        setStatus("error");
      });

    return () => {
      cancelled = true;
    };
  }, [season, week]);

  const setWeek = useCallback(
    (next: number) => {
      const clamped = Math.max(1, Math.min(maxWeek, next));
      setWeekState(clamped);
    },
    [maxWeek],
  );

  const prev = useCallback(() => setWeek(week - 1), [setWeek, week]);
  const next = useCallback(() => setWeek(week + 1), [setWeek, week]);

  return {
    status,
    season,
    week,
    maxWeek,
    results,
    slateByGameId,
    setWeek,
    prev,
    next,
  };
}
