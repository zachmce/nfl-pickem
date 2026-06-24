/**
 * My Picks page — the first real interactive screen.
 *
 * Renders as the index route INSIDE AppShell (which is inside RequireAuth), so
 * this is CONTENT ONLY: no shell, header, nav, or auth guard. A logged-in user
 * views and sets/edits their weekly picks for the current week:
 *   - a compact roster tracker (4 base slots + the dedicated mortal lock),
 *   - one card per game showing only that game's ELIGIBLE bet options,
 *   - per-pick autosave with the 3 guardrails (enforced in useMyPicks),
 *   - window/per-game-locked read-only states, and inline 4xx errors.
 *
 * v1 is SET/CHANGE ONLY. The backend exposes only POST /api/picks (submit/
 * replace) and GET /api/picks (read) — there is NO un-pick / DELETE route, so a
 * base slot cannot be emptied and the mortal lock can be CHANGED but not
 * REMOVED once set.
 * TODO(deferred): "clear a pick / remove the mortal lock" needs a backend
 * un-pick endpoint (DELETE /api/picks). Out of scope for this frontend task.
 */
import { slotKey, type PickType, type SlateGame } from "../lib/picks";
import type { WindowState } from "../lib/currentWeek";
import { useMyPicks, type PicksBySlot } from "./useMyPicks";

const PICK_TYPE_LABEL: Record<PickType, string> = {
  UNDERDOG_COVER: "Underdog",
  FAVORITE_COVER: "Favorite",
  OVER: "Over",
  UNDER: "Under",
};

/** Order the base slots appear in the roster tracker. */
const BASE_SLOTS: PickType[] = [
  "UNDERDOG_COVER",
  "FAVORITE_COVER",
  "OVER",
  "UNDER",
];

const WINDOW_BANNER: Record<Exclude<WindowState, "open">, string> = {
  not_yet_open: "This week's pick window hasn't opened yet — picks are read-only.",
  locked: "This week is locked — picks can no longer be changed.",
  closed: "This week is closed — picks can no longer be changed.",
};

/** Format an ISO kickoff like ContextBar.friendlyTime; tolerate null/invalid. */
function friendlyKickoff(iso: string | null): string {
  if (!iso) return "Time TBD";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "Time TBD";
  return d.toLocaleString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

/** Resolve a team_id on a game to its abbreviation (or "—" if unknown). */
function abbrevFor(game: SlateGame, teamId: number | null): string {
  if (teamId === null) return "—";
  if (game.home_team.team_id === teamId) return game.home_team.abbreviation;
  if (game.away_team.team_id === teamId) return game.away_team.abbreviation;
  return "—";
}

/** A short human description of the persisted line for a game card header. */
function lineSummary(game: SlateGame): string {
  const parts: string[] = [];
  if (game.spread !== null && game.favorite_team_id !== null) {
    const fav = abbrevFor(game, game.favorite_team_id);
    const dog = abbrevFor(game, game.underdog_team_id);
    parts.push(`${fav} ${game.spread} over ${dog}`);
  }
  if (game.total !== null) {
    parts.push(`O/U ${game.total}`);
  }
  return parts.length ? parts.join(" · ") : "Line unavailable";
}

/** Which game (abbrevs) a filled slot is on, for the roster tracker. */
function slotGameLabel(
  picks: PicksBySlot,
  slate: SlateGame[],
  key: string,
): string | null {
  const pick = picks[key];
  if (!pick) return null;
  const game = slate.find((g) => g.game_id === pick.game_id);
  if (!game) return PICK_TYPE_LABEL[pick.pick_type];
  return `${game.away_team.abbreviation} @ ${game.home_team.abbreviation}`;
}

export default function MyPicksPage() {
  const {
    status,
    currentWeek,
    slate,
    picks,
    editable,
    saving,
    slotError,
    select,
  } = useMyPicks();

  if (status === "loading") {
    return (
      <div>
        <h1 className="text-2xl font-bold">My Picks</h1>
        <p className="mt-2 text-gray-500">Loading your picks…</p>
      </div>
    );
  }

  if (status === "error" || !currentWeek) {
    return (
      <div>
        <h1 className="text-2xl font-bold">My Picks</h1>
        <p className="mt-2 text-gray-600">
          Couldn't load this week's picks. Please try again later.
        </p>
      </div>
    );
  }

  const windowState = currentWeek.window_state;

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold">My Picks</h1>
        <p className="mt-1 text-sm text-gray-500">
          Week {currentWeek.week} · {currentWeek.season} season
        </p>
      </header>

      {!editable && windowState !== "open" && (
        <div className="rounded-md border border-amber-300 bg-amber-50 px-4 py-2 text-sm text-amber-800">
          {WINDOW_BANNER[windowState]}
        </div>
      )}

      <RosterTracker picks={picks} slate={slate} />

      {slate.length === 0 ? (
        <p className="text-gray-500">No games are scheduled for this week.</p>
      ) : (
        <div className="space-y-4">
          {slate.map((game) => (
            <GameCard
              key={game.game_id}
              game={game}
              picks={picks}
              editable={editable}
              saving={saving}
              slotError={slotError}
              onSelect={select}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/** Compact at-a-glance summary of the 5 slots (4 base + the mortal lock). */
function RosterTracker({
  picks,
  slate,
}: {
  picks: PicksBySlot;
  slate: SlateGame[];
}) {
  const mortalKey = (() => {
    // The mortal lock is whichever pick_type has is_mortal_lock=true.
    for (const pt of BASE_SLOTS) {
      const k = slotKey(pt, true);
      if (picks[k]) return k;
    }
    return null;
  })();

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      <h2 className="text-sm font-semibold text-gray-700">Your roster</h2>
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-5">
        {BASE_SLOTS.map((pt) => {
          const key = slotKey(pt, false);
          const onGame = slotGameLabel(picks, slate, key);
          const filled = Boolean(picks[key]);
          return (
            <SlotChip
              key={key}
              label={PICK_TYPE_LABEL[pt]}
              filled={filled}
              detail={onGame}
            />
          );
        })}
        <SlotChip
          label="Mortal Lock"
          filled={Boolean(mortalKey)}
          detail={mortalKey ? slotGameLabel(picks, slate, mortalKey) : null}
          accent
        />
      </div>
    </div>
  );
}

function SlotChip({
  label,
  filled,
  detail,
  accent,
}: {
  label: string;
  filled: boolean;
  detail: string | null;
  accent?: boolean;
}) {
  return (
    <div
      className={[
        "rounded-md border px-2.5 py-2 text-center",
        filled
          ? accent
            ? "border-blue-300 bg-blue-50"
            : "border-green-300 bg-green-50"
          : "border-dashed border-gray-300 bg-gray-50",
      ].join(" ")}
    >
      <div className="text-xs font-medium text-gray-700">{label}</div>
      <div
        className={[
          "mt-0.5 text-xs",
          filled ? "text-gray-600" : "text-gray-400",
        ].join(" ")}
      >
        {filled ? (detail ?? "filled") : "empty"}
      </div>
    </div>
  );
}

/** One slate game: header (matchup/kickoff/line) + eligible bet options. */
function GameCard({
  game,
  picks,
  editable,
  saving,
  slotError,
  onSelect,
}: {
  game: SlateGame;
  picks: PicksBySlot;
  editable: boolean;
  saving: Record<string, boolean>;
  slotError: Record<string, string>;
  onSelect: (item: {
    game_id: number;
    pick_type: PickType;
    is_mortal_lock: boolean;
  }) => void;
}) {
  // Per-game frozen when the week isn't open OR this game is individually locked.
  const frozen = !editable || game.locked;

  const eligibleTypes = BASE_SLOTS.filter((pt) => game.eligibility[pt]);

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <div className="text-base font-semibold">
            {game.away_team.display_name} @ {game.home_team.display_name}
          </div>
          <div className="mt-0.5 text-xs text-gray-500">
            {friendlyKickoff(game.kickoff_at)} · {lineSummary(game)}
          </div>
        </div>
        {game.locked && (
          <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-500">
            locked
          </span>
        )}
      </div>

      {eligibleTypes.length === 0 ? (
        <p className="mt-3 text-sm text-gray-400">
          No bet options are eligible for this game.
        </p>
      ) : (
        <div className="mt-3 space-y-2">
          {eligibleTypes.map((pt) => (
            <BetOption
              key={pt}
              game={game}
              pickType={pt}
              picks={picks}
              frozen={frozen}
              saving={saving}
              slotError={slotError}
              onSelect={onSelect}
            />
          ))}
        </div>
      )}

      <p className="mt-3 text-[11px] leading-snug text-gray-400">
        Mortal lock can be changed but not removed in v1.
      </p>
    </div>
  );
}

/** A single eligible pick type on a game: base-pick button + mortal-lock toggle. */
function BetOption({
  game,
  pickType,
  picks,
  frozen,
  saving,
  slotError,
  onSelect,
}: {
  game: SlateGame;
  pickType: PickType;
  picks: PicksBySlot;
  frozen: boolean;
  saving: Record<string, boolean>;
  slotError: Record<string, string>;
  onSelect: (item: {
    game_id: number;
    pick_type: PickType;
    is_mortal_lock: boolean;
  }) => void;
}) {
  const baseKey = slotKey(pickType, false);
  const lockKey = slotKey(pickType, true);

  // A base option is "selected" when the base slot for this pick_type points at
  // THIS game. The mortal lock is "selected" when the mortal-lock slot's
  // (game_id, pick_type) matches this game + type.
  const baseSelected = picks[baseKey]?.game_id === game.game_id;
  const lockSelected = picks[lockKey]?.game_id === game.game_id;

  const baseSaving = Boolean(saving[baseKey]);
  const lockSaving = Boolean(saving[lockKey]);

  const baseError = slotError[baseKey];
  const lockError = slotError[lockKey];

  const baseDisabled = frozen || baseSaving;
  const lockDisabled = frozen || lockSaving;

  return (
    <div>
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          disabled={baseDisabled}
          onClick={() =>
            onSelect({
              game_id: game.game_id,
              pick_type: pickType,
              is_mortal_lock: false,
            })
          }
          className={[
            "rounded-md border px-3 py-1.5 text-sm font-medium transition-colors",
            baseSelected
              ? "border-blue-600 bg-blue-600 text-white"
              : "border-gray-300 bg-white text-gray-700 hover:border-blue-400",
            baseDisabled ? "cursor-not-allowed opacity-50" : "",
          ].join(" ")}
        >
          {PICK_TYPE_LABEL[pickType]}
        </button>

        <button
          type="button"
          disabled={lockDisabled}
          onClick={() =>
            onSelect({
              game_id: game.game_id,
              pick_type: pickType,
              is_mortal_lock: true,
            })
          }
          title="Designate this as your mortal lock"
          className={[
            "rounded-md border px-2.5 py-1.5 text-xs font-medium transition-colors",
            lockSelected
              ? "border-blue-600 bg-blue-50 text-blue-700"
              : "border-gray-300 bg-white text-gray-500 hover:border-blue-400",
            lockDisabled ? "cursor-not-allowed opacity-50" : "",
          ].join(" ")}
        >
          {lockSelected ? "★ Mortal lock" : "☆ Mortal lock"}
        </button>

        {(baseSaving || lockSaving) && (
          <span className="text-xs text-gray-400">Saving…</span>
        )}
      </div>

      {baseError && (
        <p className="mt-1 text-xs text-red-600">{baseError}</p>
      )}
      {lockError && lockError !== baseError && (
        <p className="mt-1 text-xs text-red-600">{lockError}</p>
      )}
    </div>
  );
}
