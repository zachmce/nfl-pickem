/**
 * Calendar page — a pageable month-grid view of the whole NFL schedule.
 *
 * Renders INSIDE AppShell (which is inside RequireAuth), so this is CONTENT
 * ONLY: no shell, header, nav, or auth guard — exactly like StandingsPage.
 *
 * Behaviour:
 *   - A standard 6-row x 7-col month grid, defaulting to the REAL current month
 *     via `new Date()`. Prev/Next page by one month (rolling the year on
 *     Dec->Jan / Jan->Dec); a "Today" button jumps back to the current month.
 *   - The grid starts on the Sunday on/before the 1st of the month and renders
 *     42 cells, so leading/trailing days of the adjacent months appear greyed.
 *   - Each game is placed on its US Eastern (America/New_York) CALENDAR DAY
 *     (LOAD-BEARING): an 8:20pm-ET Thursday night game stored as 00:20 UTC the
 *     next day must show on Thursday, not Friday. We derive the ET day key with
 *     `Intl.DateTimeFormat(..., { timeZone: 'America/New_York' })` and bucket
 *     games into a Map keyed by that `YYYY-MM-DD` ET key. Each grid cell builds
 *     its own `YYYY-MM-DD` key the same way (plain local calendar day) so cell
 *     key === game ET key matches.
 *   - Each game renders a compact informational chip (NO click-through): the
 *     matchup `AWAY @ HOME` by abbreviation, the ET kickoff time, and — only
 *     when `status === "FINAL"` — the final score `AWAY n @ HOME n`.
 *   - Loading / error / empty states render clean gray messages (none throw),
 *     mirroring the existing pages. Public-schedule view: no picks, no user data.
 */
import { useEffect, useMemo, useState } from "react";

import type { CalendarGame, CalendarResponse } from "../lib/calendar";
import { getCalendar } from "../lib/calendar";
import { formatLocalDateTime } from "../lib/datetime";
import { EMPTY_CALENDAR, ERROR_CALENDAR, LOADING_CALENDAR } from "../lib/strings";

const ET_TIME_ZONE = "America/New_York";

const MONTH_LABELS = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];

const WEEKDAY_LABELS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

/** A plain local calendar day (year, 0-indexed month, day-of-month). */
interface DayCell {
  year: number;
  month: number;
  day: number;
  inMonth: boolean;
  /** `YYYY-MM-DD` key matching the ET game keys. */
  key: string;
}

/** Zero-pad a number to two digits. */
function pad2(n: number): string {
  return String(n).padStart(2, "0");
}

/** `YYYY-MM-DD` key for a plain (year, 0-indexed month, day) tuple. */
function dayKey(year: number, month: number, day: number): string {
  return `${year}-${pad2(month + 1)}-${pad2(day)}`;
}

/**
 * The 42-cell (6x7) grid for the given month: starts on the Sunday on/before
 * the 1st, runs six full weeks. Leading/trailing adjacent-month days are
 * flagged `inMonth: false`.
 */
function buildGrid(year: number, month: number): DayCell[] {
  const first = new Date(year, month, 1);
  const start = new Date(year, month, 1 - first.getDay()); // back up to Sunday
  const cells: DayCell[] = [];
  for (let i = 0; i < 42; i++) {
    const d = new Date(start.getFullYear(), start.getMonth(), start.getDate() + i);
    const y = d.getFullYear();
    const m = d.getMonth();
    const dd = d.getDate();
    cells.push({
      year: y,
      month: m,
      day: dd,
      inMonth: m === month && y === year,
      key: dayKey(y, m, dd),
    });
  }
  return cells;
}

/**
 * The US Eastern `YYYY-MM-DD` calendar-day key for a UTC kickoff ISO string.
 * `en-CA` yields `YYYY-MM-DD` directly under the ET time zone.
 */
function etDayKey(kickoffIso: string): string {
  const d = new Date(kickoffIso);
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: ET_TIME_ZONE,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(d);
}

/** Today's ET calendar-day key (for the "today" cell marker). */
function todayEtKey(): string {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: ET_TIME_ZONE,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(new Date());
}

type Status = "loading" | "error" | "ok";

export default function CalendarPage() {
  const now = new Date();
  const [view, setView] = useState({
    year: now.getFullYear(),
    month: now.getMonth(),
  });
  const [status, setStatus] = useState<Status>("loading");
  const [games, setGames] = useState<CalendarGame[]>([]);

  const grid = useMemo(
    () => buildGrid(view.year, view.month),
    [view.year, view.month],
  );

  // The fetch window is the full visible grid: first cell .. last cell.
  const fromDate = grid[0].key;
  const toDate = grid[grid.length - 1].key;

  // Re-show the "loading" placeholder whenever the visible range changes (month
  // navigation) WITHOUT calling setState inside the fetch effect. React's
  // endorsed "adjust state during render on key change" pattern: the initial
  // range seeds loadedRange (status already starts "loading"), and any month
  // change re-enters "loading" exactly as the in-effect setStatus did.
  const rangeKey = `${fromDate}:${toDate}`;
  const [loadedRange, setLoadedRange] = useState(rangeKey);
  if (rangeKey !== loadedRange) {
    setLoadedRange(rangeKey);
    setStatus("loading");
  }

  useEffect(() => {
    let cancelled = false;
    getCalendar(fromDate, toDate)
      .then((res: CalendarResponse) => {
        if (cancelled) return;
        setGames(res.games);
        setStatus("ok");
      })
      .catch(() => {
        if (cancelled) return;
        setGames([]);
        setStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, [fromDate, toDate]);

  // Bucket games onto their ET calendar day.
  const gamesByDay = useMemo(() => {
    const map = new Map<string, CalendarGame[]>();
    for (const g of games) {
      if (!g.kickoff_at) continue;
      const key = etDayKey(g.kickoff_at);
      const list = map.get(key);
      if (list) list.push(g);
      else map.set(key, [g]);
    }
    return map;
  }, [games]);

  const todayKey = todayEtKey();
  const isCurrentMonth =
    view.year === now.getFullYear() && view.month === now.getMonth();

  function shiftMonth(delta: number) {
    setView((v) => {
      const total = v.year * 12 + v.month + delta;
      return { year: Math.floor(total / 12), month: ((total % 12) + 12) % 12 };
    });
  }

  function goToday() {
    const d = new Date();
    setView({ year: d.getFullYear(), month: d.getMonth() });
  }

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold">Calendar</h1>
          <p className="mt-1 text-sm text-fg-muted">
            {MONTH_LABELS[view.month]} {view.year}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => shiftMonth(-1)}
            className="rounded border border-border bg-surface px-3 py-1 text-sm font-medium text-fg-muted hover:bg-surface-raised"
          >
            ← Prev
          </button>
          <button
            type="button"
            onClick={goToday}
            disabled={isCurrentMonth}
            className="rounded border border-border bg-surface px-3 py-1 text-sm font-medium text-fg-muted hover:bg-surface-raised disabled:cursor-default disabled:opacity-40"
          >
            Today
          </button>
          <button
            type="button"
            onClick={() => shiftMonth(1)}
            className="rounded border border-border bg-surface px-3 py-1 text-sm font-medium text-fg-muted hover:bg-surface-raised"
          >
            Next →
          </button>
        </div>
      </header>

      {status === "loading" && (
        <p className="text-fg-muted">{LOADING_CALENDAR}</p>
      )}
      {status === "error" && (
        <p className="text-fg-muted">{ERROR_CALENDAR}</p>
      )}
      {status === "ok" && gamesByDay.size === 0 && (
        <p className="text-fg-muted">{EMPTY_CALENDAR}</p>
      )}

      <div className="overflow-x-auto rounded-lg border border-border bg-surface">
        {/* Weekday header */}
        <div className="grid grid-cols-7 border-b border-border text-center text-xs font-semibold text-fg-muted">
          {WEEKDAY_LABELS.map((w) => (
            <div key={w} className="px-2 py-2">
              {w}
            </div>
          ))}
        </div>
        {/* 6x7 day grid */}
        <div className="grid grid-cols-7">
          {grid.map((cell) => {
            const dayGames = gamesByDay.get(cell.key) ?? [];
            const isToday = cell.key === todayKey;
            return (
              <div
                key={cell.key}
                className={[
                  "min-h-24 border-b border-r border-border p-1 align-top",
                  cell.inMonth ? "bg-surface" : "bg-surface-raised",
                ].join(" ")}
              >
                <div
                  className={[
                    "mb-1 text-right text-xs",
                    cell.inMonth ? "text-fg-muted" : "text-fg-muted",
                  ].join(" ")}
                >
                  <span
                    className={
                      isToday
                        ? "inline-flex h-5 w-5 items-center justify-center rounded-full bg-accent-solid font-semibold text-on-accent"
                        : ""
                    }
                  >
                    {cell.day}
                  </span>
                </div>
                <div className="space-y-1">
                  {dayGames.map((g) => {
                    const away = g.away_team.abbreviation;
                    const home = g.home_team.abbreviation;
                    const isFinal = g.status === "FINAL";
                    return (
                      <div
                        key={g.game_id}
                        className="rounded border border-border bg-surface-raised px-1 py-0.5 text-[11px] leading-tight"
                      >
                        {isFinal ? (
                          <div className="font-medium text-fg">
                            {away} {g.away_score ?? 0} @ {home}{" "}
                            {g.home_score ?? 0}
                          </div>
                        ) : (
                          <>
                            <div className="font-medium text-fg">
                              {away} @ {home}
                            </div>
                            {g.kickoff_at && (
                              <div className="text-fg-muted">
                                {formatLocalDateTime(g.kickoff_at)}
                              </div>
                            )}
                          </>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
