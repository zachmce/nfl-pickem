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
 * Privacy: OTHER users' picks for the week are omitted SERVER-SIDE (the leak
 * gate) until the week's picks lock at the first kickoff; once the window closes
 * everyone's picks for the week — including later games — are shown. The gate is
 * week-level, so a non-self group is either fully hidden (window open) or fully
 * shown (window closed) — there is no mixed per-game partial reveal. When a
 * non-self group has no visible picks we show a subtle static "hidden until
 * lock" hint rather than a blank — the count is never computed or shown (the
 * server already redacted the entries).
 */
import { useAuth } from "../auth/useAuth";
import type { PickType, SlateGame } from "../lib/picks";
import type { UserWeekResult, WeekResultPickRead } from "../lib/results";
import { EMPTY_WEEKLY, ERROR_WEEKLY, LOADING_WEEKLY } from "../lib/strings";
import { useWeekly } from "./useWeekly";

/** Tailwind accent classes for each GradeOutcome string (UNGRADEABLE = neutral). */
const OUTCOME_STYLE: Record<string, string> = {
  WIN: "bg-success-bg text-success-fg ring-success-fg",
  LOSS: "bg-danger-bg text-danger-fg ring-danger-fg",
  PUSH: "bg-surface-raised text-fg-muted ring-border",
  INELIGIBLE: "bg-warning-bg text-warning-fg ring-warning-fg",
  UNGRADEABLE: "bg-surface-raised text-fg-muted ring-border",
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
    ? (pick.misc_text ?? "Misc pick")
    : game
      ? sideLabel(pick.pick_type, game)
      : pick.pick_type;
  const subline = isMisc ? `Misc · ${matchup}` : matchup;

  return (
    <li className="flex items-center justify-between gap-3 px-3 py-2 text-sm">
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-medium text-fg">{primary}</span>
          {!isMisc && pick.is_mortal_lock && (
            <span className="rounded-full bg-accent-bg px-2 py-0.5 text-xs font-semibold text-accent">
              Mortal Lock
            </span>
          )}
        </div>
        <div className="text-xs text-fg-muted">{subline}</div>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        <span className="tabular-nums text-fg-muted">{pick.points} pts</span>
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
        "overflow-hidden rounded-lg border bg-surface",
        isMe ? "border-accent ring-1 ring-accent" : "border-border",
      ].join(" ")}
    >
      <header
        className={[
          "flex items-center justify-between px-3 py-2",
          isMe ? "bg-accent-bg" : "bg-surface-raised",
        ].join(" ")}
      >
        <span
          className={[
            "font-semibold",
            isMe ? "text-accent" : "text-fg",
          ].join(" ")}
        >
          {result.display_name}
          {isMe && (
            <span className="ml-2 text-xs font-normal text-accent">(you)</span>
          )}
        </span>
        <span className="tabular-nums text-sm font-semibold text-fg">
          {result.weekly_score} pts
        </span>
      </header>

      {result.picks.length > 0 ? (
        <ul className="divide-y divide-border">
          {result.picks.map((pick) => (
            <PickRow
              key={`${pick.game_id}-${pick.pick_type}-${pick.is_mortal_lock}`}
              pick={pick}
              slateByGameId={slateByGameId}
            />
          ))}
        </ul>
      ) : (
        <p className="px-3 py-3 text-sm text-fg-muted">
          {isMe
            ? "No picks for this week."
            : "Picks hidden until the week's picks lock (first kickoff)."}
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
        <p className="mt-2 text-fg-muted">{LOADING_WEEKLY}</p>
      </div>
    );
  }

  if (status === "error") {
    return (
      <div>
        <h1 className="text-2xl font-bold">Weekly</h1>
        <p className="mt-2 text-fg-muted">{ERROR_WEEKLY}</p>
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
            <p className="mt-1 text-sm text-fg-muted">{seasonLabel}</p>
          )}
        </div>

        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={prev}
            disabled={atFirst}
            className="rounded-md border border-border px-3 py-1 text-sm font-medium text-fg-muted enabled:hover:bg-surface-raised disabled:cursor-not-allowed disabled:opacity-40"
          >
            ‹ Prev
          </button>
          <span className="rounded-full bg-accent-bg px-3 py-1 text-sm font-semibold text-accent">
            Week {week}
          </span>
          <button
            type="button"
            onClick={next}
            disabled={atLast}
            className="rounded-md border border-border px-3 py-1 text-sm font-medium text-fg-muted enabled:hover:bg-surface-raised disabled:cursor-not-allowed disabled:opacity-40"
          >
            Next ›
          </button>
        </div>
      </header>

      {results.length === 0 ? (
        <p className="text-fg-muted">{EMPTY_WEEKLY}</p>
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
