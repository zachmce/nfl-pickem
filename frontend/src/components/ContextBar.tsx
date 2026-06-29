/**
 * Slim current-week context bar (and a compact WeekChip variant for the header).
 *
 * Both consume GET /api/current-week. They degrade gracefully: a loading
 * placeholder, and an "unavailable" placeholder on error (e.g. 404 when no games
 * are seeded, or a network error). They NEVER throw — the shell must survive a
 * failed context read.
 */
import { useEffect, useState } from "react";

import {
  getCurrentWeek,
  type CurrentWeek,
  type WindowState,
} from "../lib/currentWeek";
import { formatLocalDateTime } from "../lib/datetime";

type Status = "loading" | "ok" | "error";

const STATE_LABEL: Record<WindowState, string> = {
  not_yet_open: "not yet open",
  open: "open",
  locked: "locked — games underway",
  closed: "closed — week final",
};

function readableState(state: WindowState): string {
  return STATE_LABEL[state] ?? state;
}

/** Shared hook: fetch the current week once, tracking loading/ok/error. */
function useCurrentWeek() {
  const [data, setData] = useState<CurrentWeek | null>(null);
  const [status, setStatus] = useState<Status>("loading");

  useEffect(() => {
    let cancelled = false;
    getCurrentWeek()
      .then((d) => {
        if (cancelled) return;
        setData(d);
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

  return { data, status };
}

/**
 * Compact week-status chip for the header bar (stays visible at all widths,
 * including when the mobile nav is collapsed).
 */
export function WeekChip() {
  const { data, status } = useCurrentWeek();

  let text = "Week …";
  if (status === "error") text = "Week —";
  else if (status === "ok" && data) {
    text = `Wk ${data.week} · ${readableState(data.window_state)}`;
  }

  return (
    <span className="rounded-full bg-blue-50 px-2.5 py-0.5 text-xs font-medium text-blue-700">
      {text}
    </span>
  );
}

/** Full slim context bar rendered below the header. */
export default function ContextBar() {
  const { data, status } = useCurrentWeek();

  let content: string;
  if (status === "loading") content = "Loading week…";
  else if (status === "error" || !data) content = "Week status unavailable";
  else if (data.season_complete) {
    // Season-over: no "closes" clause (the close date is now in the past and
    // reads nonsensically). Decision #4 of the site-consistency pass.
    content = `Season complete · ${data.season} final`;
  } else {
    content = `Week ${data.week} · picks ${readableState(
      data.window_state,
    )} · closes ${formatLocalDateTime(data.window_closes_at)}`;
  }

  return (
    <div className="border-b border-gray-200 bg-gray-50 px-4 py-1.5 text-center text-xs text-gray-600">
      {content}
    </div>
  );
}
