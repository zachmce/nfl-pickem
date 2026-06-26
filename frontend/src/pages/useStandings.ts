/**
 * State hook for the Standings page: owns the season-scoreboard data load.
 *
 * Load sequence (on mount, cancel-on-unmount like useMyPicks):
 *   1. getCurrentWeek() -> resolve the active season.
 *   2. getStandings(season) -> the pre-ordered season matrix.
 * Any failure lands in a graceful "error" status (never throws to the page).
 *
 * No JSX, no state library — plain useState/useEffect.
 */
import { useEffect, useState } from "react";

import { getCurrentWeek } from "../lib/currentWeek";
import { getStandings, type SeasonStandingRow } from "../lib/results";

export type Status = "loading" | "ok" | "error";

export interface UseStandings {
  status: Status;
  /** The resolved season once loaded, else null. */
  season: number | null;
  /** Rows in the server-returned order (do NOT re-sort). */
  standings: SeasonStandingRow[];
}

export function useStandings(): UseStandings {
  const [status, setStatus] = useState<Status>("loading");
  const [season, setSeason] = useState<number | null>(null);
  const [standings, setStandings] = useState<SeasonStandingRow[]>([]);

  useEffect(() => {
    let cancelled = false;
    setStatus("loading");

    getCurrentWeek()
      .then((cw) => getStandings(cw.season))
      .then((resp) => {
        if (cancelled) return;
        setSeason(resp.season);
        setStandings(resp.standings);
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

  return { status, season, standings };
}
