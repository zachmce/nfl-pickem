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
 * Picks can be SET, CHANGED, AND CLEARED (un-picked): each control exposes a
 * "Clear" affordance — wired to clear() — shown only when that slot is filled on
 * this game AND the slot is editable (window open + game not locked). The mortal
 * lock can now be removed, not just changed. Clearing is pessimistic (the slot
 * empties only after the server confirms via DELETE /api/picks).
 */
import { errorKey, slotKey, type PickType, type SlateGame } from "../lib/picks";
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

/**
 * The user's held picks whose `pick_type` is now INELIGIBLE on its game.
 *
 * INELIGIBLE-ONLY (side-flip / favorite-flip is deferred — there is no model
 * field for the picked side): a held pick is flagged iff its game is present on
 * the slate AND `game.eligibility[pick.pick_type] === false`. A pick whose game
 * is missing from the slate is NOT flagged. Returns one entry per affected pick
 * with the matchup label + human pick-type label for the notice copy.
 */
function ineligibleHeldPicks(
  picks: PicksBySlot,
  slate: SlateGame[],
): { matchup: string; typeLabel: string }[] {
  const out: { matchup: string; typeLabel: string }[] = [];
  for (const pick of Object.values(picks)) {
    const game = slate.find((g) => g.game_id === pick.game_id);
    if (!game) continue;
    if (game.eligibility[pick.pick_type] === false) {
      out.push({
        matchup: `${game.away_team.abbreviation} @ ${game.home_team.abbreviation}`,
        typeLabel: PICK_TYPE_LABEL[pick.pick_type],
      });
    }
  }
  return out;
}

export default function MyPicksPage() {
  const {
    status,
    currentWeek,
    slate,
    oddsFrozen,
    picks,
    editable,
    saving,
    slotError,
    select,
    clear,
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

  // After the odds freeze, surface any held pick whose type is now ineligible on
  // its game so the user can re-pick before lock (INELIGIBLE-ONLY; gated on the
  // week-level freeze flag — no notice when the week isn't frozen).
  const rePickFlags = oddsFrozen ? ineligibleHeldPicks(picks, slate) : [];

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

      {rePickFlags.length > 0 && (
        <div className="rounded-md border border-amber-300 bg-amber-50 px-4 py-2 text-sm text-amber-800">
          <p className="font-medium">
            The odds for this week have frozen and one or more of your picks is
            no longer eligible. You can re-pick before the week locks.
          </p>
          <ul className="mt-1 list-disc pl-5">
            {rePickFlags.map((f, i) => (
              <li key={`${f.matchup}-${f.typeLabel}-${i}`}>
                {f.typeLabel} on {f.matchup} is no longer eligible.
              </li>
            ))}
          </ul>
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
              onClear={clear}
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
  onClear,
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
  onClear: (item: {
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
              onClear={onClear}
            />
          ))}
        </div>
      )}
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
  onClear,
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
  onClear: (item: {
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

  // Errors are game-scoped so a rejection shows only on THIS game's control,
  // not on every same-type button across cards.
  const baseError = slotError[errorKey(game.game_id, pickType, false)];
  const lockError = slotError[errorKey(game.game_id, pickType, true)];

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

        {baseSelected && !frozen && (
          <button
            type="button"
            disabled={baseDisabled}
            onClick={() =>
              onClear({
                game_id: game.game_id,
                pick_type: pickType,
                is_mortal_lock: false,
              })
            }
            title="Clear this pick"
            className={[
              "rounded-md border px-2 py-1 text-xs font-medium transition-colors",
              "border-red-300 bg-white text-red-600 hover:border-red-500",
              baseDisabled ? "cursor-not-allowed opacity-50" : "",
            ].join(" ")}
          >
            Clear
          </button>
        )}

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

        {lockSelected && !frozen && (
          <button
            type="button"
            disabled={lockDisabled}
            onClick={() =>
              onClear({
                game_id: game.game_id,
                pick_type: pickType,
                is_mortal_lock: true,
              })
            }
            title="Remove your mortal lock"
            className={[
              "rounded-md border px-2 py-1 text-xs font-medium transition-colors",
              "border-red-300 bg-white text-red-600 hover:border-red-500",
              lockDisabled ? "cursor-not-allowed opacity-50" : "",
            ].join(" ")}
          >
            ✕ Remove lock
          </button>
        )}

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
