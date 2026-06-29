/**
 * Shared local-timezone date+time formatting for kickoff/window times.
 *
 * One source of truth for "when does this happen" displays across the app
 * (ContextBar window-close, MyPicks/Admin kickoff, Calendar kickoff). It always
 * renders in the VIEWER'S local timezone and always labels that zone's
 * abbreviation, so a time never reads ambiguously (e.g. "Sat, Jun 27 · 4:13 PM
 * PST"). This is decision #3 of the site-consistency pass — locked.
 *
 * Uses the platform `Intl.DateTimeFormat` (no date library). It only formats
 * whatever `Date` it is handed; demo time-offset behavior lives upstream and is
 * not touched here. Date-only displays (e.g. AdminPage `created_at`) are NOT in
 * scope — this helper is for times.
 */

/** Fallback shown when the kickoff/window time is null/empty or unparseable. */
const TIME_TBD = "Time TBD";

/**
 * Format an ISO timestamp as a local-timezone date+time with a zone label,
 * e.g. "Sat, Jun 27 · 4:13 PM PST". Returns "Time TBD" when `iso` is
 * null/empty or yields an invalid Date — never echoes the raw input and never
 * throws. The viewer's locale is honored (locale is left undefined).
 */
export function formatLocalDateTime(iso: string | null): string {
  if (!iso) return TIME_TBD;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return TIME_TBD;

  const parts = new Intl.DateTimeFormat(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZoneName: "short",
  }).formatToParts(d);

  const get = (type: Intl.DateTimeFormatPartTypes): string =>
    parts.find((p) => p.type === type)?.value ?? "";

  const weekday = get("weekday");
  const month = get("month");
  const day = get("day");
  const hour = get("hour");
  const minute = get("minute");
  const dayPeriod = get("dayPeriod");
  const timeZoneName = get("timeZoneName");

  // Assemble explicitly so the middot separator is standardized regardless of
  // the viewer's locale punctuation.
  return `${weekday}, ${month} ${day} · ${hour}:${minute} ${dayPeriod} ${timeZoneName}`;
}
