/**
 * Week context: fetches GET /api/current-week once for the whole shell and
 * exposes {data, status} to the header WeekChip and the sub-header ContextBar so
 * they share ONE source of truth and can never momentarily disagree.
 *
 * The fetch re-runs on route change (keyed on useLocation().pathname) and on
 * window focus / tab visibility, so the label reflects the real pick-window
 * state during normal SPA browsing — not just after a hard reload. Focus/
 * visibility refetches are skipped while a request is already in flight
 * (inFlightRef guard); nav refetches always fire. The last-good data stays
 * visible during a background refresh (status is NOT reset to "loading" on
 * nav/focus) to avoid a loading flash on every navigation.
 */
import { useEffect, useRef, useState, type ReactNode } from "react";
import { useLocation } from "react-router-dom";

import { getCurrentWeek, type CurrentWeek } from "../lib/currentWeek";
import { WeekContext, type WeekState } from "./week-context";

export function WeekProvider({ children }: { children: ReactNode }) {
  const [data, setData] = useState<CurrentWeek | null>(null);
  const [status, setStatus] = useState<WeekState["status"]>("loading");
  const inFlightRef = useRef(false);
  const { pathname } = useLocation();

  useEffect(() => {
    let cancelled = false;

    function fetchWeek(guardInFlight: boolean) {
      // Focus/visibility refetches skip while a request is already running; the
      // nav/mount fetch (guardInFlight=false) always fires.
      if (guardInFlight && inFlightRef.current) return;
      inFlightRef.current = true;
      getCurrentWeek()
        .then((d) => {
          if (cancelled) return;
          setData(d);
          setStatus("ok");
        })
        .catch(() => {
          if (cancelled) return;
          setStatus("error");
        })
        .finally(() => {
          inFlightRef.current = false;
        });
    }

    // Mount / navigation fetch — never suppressed by the in-flight guard.
    fetchWeek(false);

    function handleFocus() {
      fetchWeek(true);
    }
    function handleVisibility() {
      if (document.visibilityState === "visible") fetchWeek(true);
    }

    window.addEventListener("focus", handleFocus);
    document.addEventListener("visibilitychange", handleVisibility);

    return () => {
      cancelled = true;
      window.removeEventListener("focus", handleFocus);
      document.removeEventListener("visibilitychange", handleVisibility);
    };
  }, [pathname]);

  return (
    <WeekContext.Provider value={{ data, status }}>
      {children}
    </WeekContext.Provider>
  );
}
