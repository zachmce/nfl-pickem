/**
 * Standings page — the season SCOREBOARD (D1: the season matrix).
 *
 * Renders INSIDE AppShell (which is inside RequireAuth), so this is CONTENT
 * ONLY: no shell, header, nav, or auth guard — exactly like MyPicksPage. A
 * logged-in user sees one row per player in the server-returned order, with
 * each player's per-week scores (W1, W2, …) and their season Total, all sourced
 * from the single GET /api/results/standings response (no extra fetch).
 *
 * Behaviour:
 *   - Week COLUMNS are ALWAYS the full regular season W1..W18 (extended only if
 *     the data somehow carries a higher week). Weeks AFTER the current week
 *     render as "N/A" (not yet playable); weeks <= the current week show their
 *     score (0 included).
 *   - Column order is Rank, Total, Player, then the weekly W1..W18 columns — the
 *     summary reads left-to-right before the per-week detail.
 *   - Rank is 1-based standard COMPETITION ranking over the server order: tied
 *     season_total values share a rank, and the next distinct total resumes at
 *     index+1 (totals [50,31,31,30] -> ranks [1,2,2,4]).
 *   - The current user's row (display_name === useAuth().user?.display_name) is
 *     highlighted with the app's single blue accent. No match -> no highlight.
 *   - Loading / error / empty (zero rows) each render a clean gray message; none
 *     of them throw.
 */
import { useAuth } from "../auth/useAuth";
import type { SeasonStandingRow } from "../lib/results";
import { EMPTY_STANDINGS, ERROR_STANDINGS, LOADING_STANDINGS } from "../lib/strings";
import { useStandings } from "./useStandings";

/** NFL regular season length — the matrix always shows at least W1..W18. */
const REGULAR_SEASON_WEEKS = 18;

/**
 * The week columns to render: always 1..REGULAR_SEASON_WEEKS, extended to cover
 * any higher integer week present in the data (defensive — the backend never
 * emits weeks beyond 18 for the regular season).
 */
function weekColumns(rows: SeasonStandingRow[]): number[] {
  let last = REGULAR_SEASON_WEEKS;
  for (const row of rows) {
    for (const key of Object.keys(row.weekly_scores)) {
      const n = Number(key);
      if (Number.isInteger(n) && n > last) last = n;
    }
  }
  return Array.from({ length: last }, (_, i) => i + 1);
}

/**
 * Map a 1-based COMPETITION rank to its season-end medal: 1 -> 🥇, 2 -> 🥈,
 * 3 -> 🥉, any other rank -> "" (no medal). Because the rank comes from
 * competitionRanks(), tied players share a rank and therefore a medal — e.g. two
 * players tied at rank 1 both show 🥇 and the next distinct total lands at rank 3
 * (🥉) with no 🥈 emitted. Only shown when the season is complete.
 */
function rankMedal(rank: number): string {
  if (rank === 1) return "🥇";
  if (rank === 2) return "🥈";
  if (rank === 3) return "🥉";
  return "";
}

/**
 * Standard competition ranks ("1, 2, 2, 4") for the ALREADY-ORDERED rows, keyed
 * by season_total: row 0 is rank 1; each subsequent row keeps the previous rank
 * when its total ties the row above, otherwise its rank is its 1-based position.
 */
function competitionRanks(rows: SeasonStandingRow[]): number[] {
  const ranks: number[] = [];
  for (let i = 0; i < rows.length; i++) {
    if (i > 0 && rows[i].season_total === rows[i - 1].season_total) {
      ranks.push(ranks[i - 1]);
    } else {
      ranks.push(i + 1);
    }
  }
  return ranks;
}

export default function StandingsPage() {
  const { status, season, currentWeek, standings, seasonComplete } =
    useStandings();
  const { user } = useAuth();

  const seasonLabel =
    season !== null
      ? `${season} season${seasonComplete ? " · final" : ""}`
      : null;

  if (status === "loading") {
    return (
      <div>
        <h1 className="text-2xl font-bold">Standings</h1>
        <p className="mt-2 text-fg-muted">{LOADING_STANDINGS}</p>
      </div>
    );
  }

  if (status === "error") {
    return (
      <div>
        <h1 className="text-2xl font-bold">Standings</h1>
        <p className="mt-2 text-fg-muted">{ERROR_STANDINGS}</p>
      </div>
    );
  }

  const weeks = weekColumns(standings);
  // Weeks after this are "future" -> N/A. Null (shouldn't happen in "ok"
  // status) treated as 0 so every week reads N/A rather than a false 0.
  const liveThroughWeek = currentWeek ?? 0;

  // Zero-board: no players yet.
  if (standings.length === 0) {
    return (
      <div className="space-y-6">
        <header>
          <h1 className="text-2xl font-bold">Standings</h1>
          {seasonLabel && (
            <p className="mt-1 text-sm text-fg-muted">{seasonLabel}</p>
          )}
        </header>
        <p className="text-fg-muted">{EMPTY_STANDINGS}</p>
      </div>
    );
  }

  const ranks = competitionRanks(standings);

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold">Standings</h1>
        {seasonLabel && (
          <p className="mt-1 text-sm text-fg-muted">{seasonLabel}</p>
        )}
      </header>

      <div className="overflow-x-auto rounded-lg border border-border bg-surface">
        <table className="min-w-full border-collapse text-sm">
          <thead>
            <tr className="border-b border-border text-fg-muted">
              <th className="px-3 py-2 text-right font-semibold">Rank</th>
              <th className="px-3 py-2 text-right font-semibold">Total</th>
              <th className="px-3 py-2 text-left font-semibold">Player</th>
              {weeks.map((w) => (
                <th key={w} className="px-3 py-2 text-right font-semibold">
                  W{w}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {standings.map((row, i) => {
              const isMe =
                user?.display_name != null &&
                row.display_name === user.display_name;
              return (
                <tr
                  key={`${row.display_name}-${i}`}
                  className={[
                    "border-b border-border last:border-0",
                    isMe ? "bg-accent-bg" : "",
                  ].join(" ")}
                >
                  <td className="px-3 py-2 text-right tabular-nums text-fg-muted">
                    {seasonComplete && rankMedal(ranks[i]) !== "" && (
                      <span className="mr-1" aria-hidden="true">
                        {rankMedal(ranks[i])}
                      </span>
                    )}
                    {ranks[i]}
                  </td>
                  <td className="px-3 py-2 text-right font-semibold tabular-nums text-fg">
                    {row.season_total}
                  </td>
                  <td
                    className={[
                      "px-3 py-2 text-left",
                      isMe ? "font-semibold text-accent" : "text-fg",
                    ].join(" ")}
                  >
                    {row.display_name}
                  </td>
                  {weeks.map((w) => {
                    const future = w > liveThroughWeek;
                    return (
                      <td
                        key={w}
                        className={[
                          "px-3 py-2 text-right tabular-nums",
                          future ? "text-fg-muted" : "text-fg-muted",
                        ].join(" ")}
                      >
                        {future ? "N/A" : (row.weekly_scores[String(w)] ?? 0)}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
