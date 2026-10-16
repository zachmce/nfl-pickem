/**
 * Misc grading panel (#319): every MISC pick for one week, graded in place.
 * Opens on the most recent complete week (resolved by the server). Each row saves
 * through the existing per-user misc-grade endpoint, so the audit row and the
 * misc.graded chat post are the same as a grade from the pick editor.
 */
import { useCallback, useEffect, useState } from "react";
import { ApiError } from "../lib/api";
import {
  gameLabel,
  getWeekMiscPicks,
  gradeCounts,
  gradeMisc,
  type AdminMiscPickRow,
  type AdminMiscPicks,
} from "../lib/admin";

const WEEKS = Array.from({ length: 18 }, (_, i) => i + 1);

function messageFor(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  return "Something went wrong. Please try again.";
}

function buttonClass(disabled: boolean): string {
  return [
    "rounded-md border px-3 py-1.5 text-sm font-medium transition-colors",
    "border-accent-solid bg-accent-solid text-on-accent hover:bg-accent-solid-hover",
    disabled ? "cursor-not-allowed opacity-50" : "",
  ].join(" ");
}

function MiscGradeRow({
  row,
  season,
  week,
  onGraded,
}: {
  row: AdminMiscPickRow;
  season: number;
  week: number;
  onGraded: (row: AdminMiscPickRow) => void;
}) {
  const [result, setResult] = useState<"WIN" | "LOSS" | null>(
    row.result === "PENDING" ? null : row.result,
  );
  const [points, setPoints] = useState<string>(String(row.points));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onSave = async () => {
    if (result === null) return;
    setSaving(true);
    setError(null);
    try {
      const pick = await gradeMisc(row.user_id, season, week, {
        result,
        points: Math.trunc(Number(points) || 0),
      });
      onGraded({ ...row, result: pick.result, points: pick.points });
      setPoints(String(pick.points));
    } catch (err: unknown) {
      setError(messageFor(err));
    } finally {
      setSaving(false);
    }
  };

  const choice = (value: "WIN" | "LOSS", label: string) => {
    const active = result === value;
    const on =
      value === "WIN"
        ? "border-success-solid bg-success-solid text-on-success"
        : "border-danger-solid bg-danger-solid text-on-danger";
    const off =
      value === "WIN"
        ? "border-border bg-surface text-fg-muted hover:border-success-fg"
        : "border-border bg-surface text-fg-muted hover:border-danger-fg";
    return (
      <button
        type="button"
        disabled={saving}
        aria-pressed={active}
        onClick={() => setResult(value)}
        className={[
          "rounded-md border px-2.5 py-1 text-xs font-medium transition-colors",
          active ? on : off,
          saving ? "cursor-not-allowed opacity-50" : "",
        ].join(" ")}
      >
        {label}
      </button>
    );
  };

  return (
    <li className="space-y-2 border-t border-border py-3" data-testid="misc-grade-row">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <span className="font-medium">{row.display_name}</span>
        <span className="text-xs text-fg-muted">{gameLabel(row)}</span>
      </div>
      <p className="text-sm">{row.misc_text ?? <em className="text-fg-muted">No text</em>}</p>
      <div className="flex flex-wrap items-center gap-2">
        {choice("WIN", "Correct")}
        {choice("LOSS", "Incorrect")}
        <label className="text-xs font-medium text-fg-muted">
          Points
          <input
            type="number"
            step={1}
            value={points}
            disabled={saving}
            onChange={(e) => setPoints(e.target.value)}
            className="ml-1 w-20 rounded-md border border-border px-2 py-1 text-sm"
          />
        </label>
        <button
          type="button"
          disabled={saving || result === null}
          onClick={() => void onSave()}
          className={buttonClass(saving || result === null)}
        >
          Save grade
        </button>
        <span className="text-xs">
          {row.result === "WIN" ? (
            <span className="text-success-fg">Correct · {row.points} pts</span>
          ) : row.result === "LOSS" ? (
            <span className="text-danger-fg">Incorrect · {row.points} pts</span>
          ) : (
            <span className="text-fg-muted">Ungraded</span>
          )}
        </span>
      </div>
      {error && <p className="text-xs text-danger-fg">{error}</p>}
    </li>
  );
}

export default function MiscGradingPanel() {
  const [season, setSeason] = useState<number>(new Date().getFullYear());
  const [week, setWeek] = useState<number>(1);
  const [data, setData] = useState<AdminMiscPicks | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [hideGraded, setHideGraded] = useState(false);

  const apply = useCallback((res: AdminMiscPicks) => {
    setData(res);
    setSeason(res.season);
    setWeek(res.week);
  }, []);

  useEffect(() => {
    let cancelled = false;
    getWeekMiscPicks()
      .then((res) => {
        if (!cancelled) apply(res);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(messageFor(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [apply]);

  const onLoad = async () => {
    setLoading(true);
    setError(null);
    try {
      apply(await getWeekMiscPicks({ season, week }));
    } catch (err: unknown) {
      setData(null);
      setError(messageFor(err));
    } finally {
      setLoading(false);
    }
  };

  const onGraded = useCallback((updated: AdminMiscPickRow) => {
    setData((prev) =>
      prev && {
        ...prev,
        picks: prev.picks.map((r) => (r.user_id === updated.user_id ? updated : r)),
      },
    );
  }, []);

  const rows = data?.picks ?? [];
  const counts = gradeCounts(rows);
  const visible = hideGraded ? rows.filter((r) => r.result === "PENDING") : rows;

  return (
    <section
      data-testid="misc-grading-panel"
      className="space-y-3 rounded-lg border border-border bg-surface p-4"
    >
      <div>
        <h2 className="text-lg font-bold">Misc Grading</h2>
        <p className="mt-0.5 text-sm text-fg-muted">
          Grade every misc pick for a week. Each saved grade posts to the pickem
          chat, the same as a grade from the pick editor.
        </p>
      </div>

      <div className="flex flex-wrap items-end gap-3 border-t border-border pt-3">
        <label className="text-sm">
          <span className="block text-xs font-medium text-fg-muted">Season</span>
          <input
            type="number"
            value={season}
            onChange={(e) => setSeason(Number(e.target.value))}
            className="mt-1 w-24 rounded-md border border-border px-2 py-1 text-sm"
          />
        </label>
        <label className="text-sm">
          <span className="block text-xs font-medium text-fg-muted">Week</span>
          <select
            value={week}
            onChange={(e) => setWeek(Number(e.target.value))}
            className="mt-1 block w-24 rounded-md border border-border px-2 py-1.5 text-sm"
          >
            {WEEKS.map((w) => (
              <option key={w} value={w}>
                Week {w}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          disabled={loading || !Number.isFinite(season)}
          onClick={() => void onLoad()}
          className={buttonClass(loading || !Number.isFinite(season))}
        >
          Load misc picks
        </button>
        {loading && <span className="text-xs text-fg-muted">Loading…</span>}
      </div>

      {error && <p className="text-sm text-danger-fg">{error}</p>}

      {data && (
        <div>
          <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
            <span data-testid="misc-grading-summary">
              Season {data.season} · Week {data.week}: {rows.length} picks ·{" "}
              {counts.graded} graded · {counts.ungraded} ungraded
            </span>
            <label className="flex items-center gap-1.5 text-xs text-fg-muted">
              <input
                type="checkbox"
                checked={hideGraded}
                onChange={(e) => setHideGraded(e.target.checked)}
              />
              Hide graded
            </label>
          </div>
          {rows.length === 0 ? (
            <p className="mt-2 text-sm text-fg-muted">No misc picks for this week.</p>
          ) : (
            <ul className="mt-2">
              {visible.map((row) => (
                <MiscGradeRow
                  key={`${data.season}-${data.week}-${row.user_id}`}
                  row={row}
                  season={data.season}
                  week={data.week}
                  onGraded={onGraded}
                />
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}
