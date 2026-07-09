/**
 * Slim current-week context bar (and a compact WeekChip variant for the header).
 *
 * Both consume GET /api/current-week. They degrade gracefully: a loading
 * placeholder, and an "unavailable" placeholder on error (e.g. 404 when no games
 * are seeded, or a network error). They NEVER throw — the shell must survive a
 * failed context read.
 */
import { type WindowState } from "../lib/currentWeek";
import { formatLocalDateTime } from "../lib/datetime";
import { ERROR_WEEK_STATUS, LOADING_WEEK_STATUS } from "../lib/strings";
import { useWeek } from "./useWeek";

const STATE_LABEL: Record<WindowState, string> = {
  not_yet_open: "not yet open",
  open: "open",
  locked: "locked — games underway",
  closed: "closed — week final",
};

function readableState(state: WindowState): string {
  return STATE_LABEL[state] ?? state;
}

/**
 * Compact week-status chip for the header bar (stays visible at all widths,
 * including when the mobile nav is collapsed).
 */
export function WeekChip() {
  const { data, status } = useWeek();

  let text = "Week …";
  if (status === "error") text = "Week —";
  else if (status === "ok" && data) {
    text = `Wk ${data.week} · ${readableState(data.window_state)}`;
  }

  return (
    <span className="rounded-full bg-accent-bg px-2.5 py-0.5 text-xs font-medium text-accent">
      {text}
    </span>
  );
}

/** Full slim context bar rendered below the header. */
export default function ContextBar() {
  const { data, status } = useWeek();

  let content: string;
  if (status === "loading") content = LOADING_WEEK_STATUS;
  else if (status === "error" || !data) content = ERROR_WEEK_STATUS;
  else if (data.season_complete) {
    // Season-over: no "closes" clause (the close date is now in the past and
    // reads nonsensically). Decision #4 of the site-consistency pass.
    content = `Season complete · ${data.season} final`;
  } else {
    // Two-state week-level odds indicator, appended after the "closes" clause on
    // the in-progress line ONLY (not season-complete, loading, error, or the
    // compact WeekChip). Lowercase matches the bar's dot-separated segment style;
    // "live" (not "open") avoids colliding with the pick-window "open" state.
    const linesLabel = data.odds_frozen ? "lines locked" : "lines live";
    // Show the "closes" clause only while the window is open — the close date is
    // the week's first kickoff, so it's premature before open and already in the
    // past once locked/closed, reading nonsensically (same reason the
    // season_complete branch drops it).
    const closesClause =
      data.window_state === "open"
        ? ` · closes ${formatLocalDateTime(data.window_closes_at)}`
        : "";
    content = `Week ${data.week} · picks ${readableState(
      data.window_state,
    )}${closesClause} · ${linesLabel}`;
  }

  return (
    <div className="border-b border-border bg-surface-raised px-4 py-1.5 text-center text-xs text-fg-muted">
      {content}
    </div>
  );
}
