/**
 * Weekly page — a pageable, per-week view of EVERY user's graded picks.
 *
 * Renders INSIDE AppShell (content only, like StandingsPage): no shell/header/
 * nav/guard. It defaults to the current week, paginates prev/next (clamped
 * 1..maxWeek), and renders each user as a grouped card: a header row with
 * display_name + weekly_score, then each pick joined to the slate (by game_id)
 * to resolve the matchup, the concrete side from pick_type, the outcome, points,
 * and a mortal-lock badge. The current user's group is highlighted via a
 * display_name match against useAuth().
 *
 * Privacy: OTHER users' picks on not-yet-locked games are omitted SERVER-SIDE
 * (the leak gate). When a non-self group has no visible picks we show a subtle
 * static "hidden until lock" hint rather than a blank — the count is never
 * computed or shown (the server already redacted the entries).
 */
import { useAuth } from "../auth/useAuth";
import type { PickType, SlateGame } from "../lib/picks";
import type { UserWeekResult, WeekResultPickRead } from "../lib/results";
import { useWeekly } from "./useWeekly";

/** Tailwind accent classes for each GradeOutcome string (UNGRADEABLE = neutral). */
const OUTCOME_STYLE: Record<string, string> = {
  WIN: "bg-green-50 text-green-700 ring-green-200",
  LOSS: "bg-red-50 text-red-700 ring-red-200",
  PUSH: "bg-gray-100 text-gray-600 ring-gray-200",
  INELIGIBLE: "bg-amber-50 text-amber-700 ring-amber-200",
  UNGRADEABLE: "bg-gray-50 text-gray-500 ring-gray-200",
};

/** Human label for each outcome (UNGRADEABLE reads as "pending"). */
const OUTCOME_LABEL: Record<string, string> = {
  WIN: "Win",
  LOSS: "Loss",
  PUSH: "Push",
  INELIGIBLE: "Ineligible",
  UNGRADEABLE: "Pending",
};

function outcomeStyle(outcome: string): string {
  return OUTCOME_STYLE[outcome] ?? OUTCOME_STYLE.UNGRADEABLE;
}

function outcomeLabel(outcome: string): string {
  return OUTCOME_LABEL[outcome] ?? outcome;
}

/** "AWAY @ HOME" using team abbreviations. */
function matchupLabel(game: SlateGame): string {
  return `${game.away_team.abbreviation} @ ${game.home_team.abbreviation}`;
}

/**
 * Resolve the concrete SIDE a pick represents, joining pick_type to the slate
 * game: a spread side names the favorite/underdog team + line; a total side is
 * "Over {total}" / "Under {total}". Falls back to the raw pick_type when the
 * slate lacks the needed line/team fields.
 */
function sideLabel(pickType: PickType, game: SlateGame): string {
  const teamName = (teamId: number | null): string | null => {
    if (teamId === null) return null;
    if (game.home_team.team_id === teamId) return game.home_team.abbreviation;
    if (game.away_team.team_id === teamId) return game.away_team.abbreviation;
    return null;
  };

  switch (pickType) {
    case "FAVORITE_COVER": {
      const name = teamName(game.favorite_team_id);
      const spread = game.spread != null ? ` -${game.spread}` : "";
      return name ? `${name}${spread}` : "Favorite";
    }
    case "UNDERDOG_COVER": {
      const name = teamName(game.underdog_team_id);
      const spread = game.spread != null ? ` +${game.spread}` : "";
      return name ? `${name}${spread}` : "Underdog";
    }
    case "OVER":
      return game.total != null ? `Over ${game.total}` : "Over";
    case "UNDER":
      return game.total != null ? `Under ${game.total}` : "Under";
    default:
      return pickType;
  }
}

function PickRow({
  pick,
  slateByGameId,
}: {
  pick: WeekResultPickRead;
  slateByGameId: Record<number, SlateGame>;
}) {
  const game = slateByGameId[pick.game_id];
  const matchup = game ? matchupLabel(game) : `Game #${pick.game_id}`;

  // MISC is a non-base type with NO resolved side and NO mortal lock: the primary
  // label is its free-text prediction (NOT routed through the spread/total side
  // resolution), the matchup subline is prefixed "Misc ·", and no mortal-lock
  // badge is ever shown. A revealed MISC always carries misc_text; if it's
  // absent (server-redacted/omitted edge), fall back to a neutral label.
  const isMisc = pick.pick_type === "MISC";
  const primary = isMisc
    ? (pick.misc_text ?? "Misc prediction")
    : game
      ? sideLabel(pick.pick_type, game)
      : pick.pick_type;
  const subline = isMisc ? `Misc · ${matchup}` : matchup;

  return (
    <li className="flex items-center justify-between gap-3 px-3 py-2 text-sm">
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-medium text-gray-800">{primary}</span>
          {!isMisc && pick.is_mortal_lock && (
            <span className="rounded-full bg-blue-100 px-2 py-0.5 text-xs font-semibold text-blue-700">
              Mortal Lock
            </span>
          )}
        </div>
        <div className="text-xs text-gray-500">{subline}</div>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        <span className="tabular-nums text-gray-600">{pick.points} pts</span>
        <span
          className={[
            "rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset",
            outcomeStyle(pick.outcome),
          ].join(" ")}
        >
          {outcomeLabel(pick.outcome)}
        </span>
      </div>
    </li>
  );
}

function UserCard({
  result,
  slateByGameId,
  isMe,
}: {
  result: UserWeekResult;
  slateByGameId: Record<number, SlateGame>;
  isMe: boolean;
}) {
  return (
    <section
      className={[
        "overflow-hidden rounded-lg border bg-white",
        isMe ? "border-blue-300 ring-1 ring-blue-200" : "border-gray-200",
      ].join(" ")}
    >
      <header
        className={[
          "flex items-center justify-between px-3 py-2",
          isMe ? "bg-blue-50" : "bg-gray-50",
        ].join(" ")}
      >
        <span
          className={[
            "font-semibold",
            isMe ? "text-blue-800" : "text-gray-800",
          ].join(" ")}
        >
          {result.display_name}
          {isMe && (
            <span className="ml-2 text-xs font-normal text-blue-600">(you)</span>
          )}
        </span>
        <span className="tabular-nums text-sm font-semibold text-gray-900">
          {result.weekly_score} pts
        </span>
      </header>

      {result.picks.length > 0 ? (
        <ul className="divide-y divide-gray-100">
          {result.picks.map((pick) => (
            <PickRow
              key={`${pick.game_id}-${pick.pick_type}-${pick.is_mortal_lock}`}
              pick={pick}
              slateByGameId={slateByGameId}
            />
          ))}
        </ul>
      ) : (
        <p className="px-3 py-3 text-sm text-gray-400">
          {isMe
            ? "No picks for this week."
            : "Picks hidden until each game locks."}
        </p>
      )}
    </section>
  );
}

export default function WeeklyPage() {
  const { status, season, week, maxWeek, results, slateByGameId, prev, next } =
    useWeekly();
  const { user } = useAuth();

  const seasonLabel = season !== null ? `${season} season` : null;

  if (status === "loading") {
    return (
      <div>
        <h1 className="text-2xl font-bold">Weekly</h1>
        <p className="mt-2 text-gray-500">Loading this week's picks…</p>
      </div>
    );
  }

  if (status === "error") {
    return (
      <div>
        <h1 className="text-2xl font-bold">Weekly</h1>
        <p className="mt-2 text-gray-600">
          Couldn't load the weekly results. Please try again later.
        </p>
      </div>
    );
  }

  const atFirst = week <= 1;
  const atLast = week >= maxWeek;

  return (
    <div className="space-y-6">
      <header className="space-y-3">
        <div>
          <h1 className="text-2xl font-bold">Weekly</h1>
          {seasonLabel && (
            <p className="mt-1 text-sm text-gray-500">{seasonLabel}</p>
          )}
        </div>

        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={prev}
            disabled={atFirst}
            className="rounded-md border border-gray-300 px-3 py-1 text-sm font-medium text-gray-700 enabled:hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-40"
          >
            ‹ Prev
          </button>
          <span className="rounded-full bg-blue-50 px-3 py-1 text-sm font-semibold text-blue-700">
            Week {week}
          </span>
          <button
            type="button"
            onClick={next}
            disabled={atLast}
            className="rounded-md border border-gray-300 px-3 py-1 text-sm font-medium text-gray-700 enabled:hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Next ›
          </button>
        </div>
      </header>

      {results.length === 0 ? (
        <p className="text-gray-500">
          No picks have been made for this week yet.
        </p>
      ) : (
        <div className="space-y-4">
          {results.map((result, i) => {
            const isMe =
              user?.display_name != null &&
              result.display_name === user.display_name;
            return (
              <UserCard
                key={`${result.display_name}-${i}`}
                result={result}
                slateByGameId={slateByGameId}
                isMe={isMe}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}
