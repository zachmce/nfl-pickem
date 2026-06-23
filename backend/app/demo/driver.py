"""Demo season-walkthrough driver — build-order step #5, the CAPSTONE.

This is the integration harness that ties the whole pick'em season lifecycle
together and makes it ASSERTABLE: scoreboard source + poller + pick submission +
bots + oracle, run as a single weeks-1-13 walkthrough of the 2025 season against
the REAL app surfaces (weeks 1-13 are the odds-bearing extent of the fixture).

For each week N (default 1..13) the driver:

1. positions :class:`~app.scoreboard.demo.Demo2025Source` via
   :func:`app.demo.offset.compute_offset` so week N's pick window is OPEN (week N
   future, week N-1 ended) and runs :func:`app.services.refresh.refresh_games` so
   windows are stamped and prior weeks reflect FINAL;
2. submits each bot's preordained week-N picks through the REAL
   :func:`app.services.pick_submission.submit_picks` (never hand-inserting
   ``Pick`` rows) — so the picks pass the same window/lock/validation a real user
   would;
3. re-positions the source so all of week N is FINAL and runs ``refresh_games``
   again to finalize the games with their real scores;
4. computes ACTUAL standings from the PERSISTED DB rows (not by re-reading
   ``BOT_PICKS``) and, when ``assert_oracle`` is set, asserts they equal
   :func:`app.demo.oracle.compute_standings` for weeks 1..N — the integration
   proof.

Why this is the proof: the oracle is the precomputed-expected side; the driver is
the actual side built entirely from real persisted rows that flowed through the
real submit/poll/score stack. ``actual == expected`` proves the surrounding
integration is wired correctly ("tests pass but prod breaks" made executable).

Design — offline-testable core (mirrors the sibling services):

* Operates on a **passed-in** ``Session`` plus an injected ``source_factory``
  (default ``lambda offset: Demo2025Source(offset)``), so the test drives the
  exact same core against in-memory SQLite. The thin CLI shell
  (:mod:`app.demo.run`) wires a real demo DB + the default factory.
* Imports NO ``app.db``, NO ``app.config``, and NO network layer — fully
  offline. ``now`` is always the real ``datetime.now(timezone.utc)`` (no virtual
  clock); positioning is the ONLY mechanism, via the offset helper.
* Submits picks ONLY through ``submit_picks``. If ``submit_picks`` (or
  ``refresh_games``) raises during the walkthrough — a closed window, a locked
  game, a validation failure — that is a REAL integration bug: it PROPAGATES (it
  is never caught-and-skipped), surfacing rather than silently working around the
  services.
* Does NOT modify ``pick_submission``, ``refresh``, the seeds, ``oracle``, or the
  pure services — it only imports them.

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use the venv interpreter ``.venv/bin/python``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable, Mapping, Sequence

from sqlmodel import Session, select

from app.demo.offset import DemoPhase, compute_offset, load_fixture_kickoffs
from app.demo.oracle import (
    BotSeasonResult,
    Standings,
    compute_standings,
    games_by_pk_index,
)
from app.models import Game, GameStatus, Pick, User, Week
from app.schemas.picks import PickItem
from app.scoreboard.demo import Demo2025Source
from app.seeds.bots import BOT_ACCOUNTS, seed_bots
from app.seeds.data.bot_picks_2025 import BOT_PICKS
from app.seeds.fixture_2025 import import_fixture_2025
from app.seeds.teams import seed_teams
from app.services.pick_submission import submit_picks
from app.services.refresh import refresh_games
from app.services.scoring import score_week

# The bot display_names that participate in the walkthrough (the keys shared by
# BOT_ACCOUNTS and BOT_PICKS). Computed once, not re-derived per call.
BOT_NAMES: frozenset[str] = frozenset(name for name, _pw in BOT_ACCOUNTS)

# Default per-week set for the walkthrough — the full odds-bearing season
# (weeks 1-13; weeks 14-18 carry no odds in the fixture, so the demo season runs
# out of gradeable data at week 13).
DEFAULT_WEEKS: tuple[int, ...] = tuple(range(1, 14))

# Source factory type: maps an offset timedelta to a ScoreboardSource.
SourceFactory = Callable[[object], object]


class WalkthroughAssertionError(AssertionError):
    """Raised when DB-sourced actual standings diverge from the oracle.

    Carries the structured ``(week, actual, expected)`` so a mismatch is
    diagnosable rather than a bare ``AssertionError``.
    """

    def __init__(
        self,
        *,
        week: int,
        actual: Standings,
        expected: Standings,
    ) -> None:
        self.week = week
        self.actual = actual
        self.expected = expected
        super().__init__(
            f"walkthrough mismatch at week {week}: "
            f"DB-sourced standings != oracle.\n"
            f"  actual:   {actual}\n"
            f"  expected: {expected}"
        )


@dataclass(frozen=True)
class WalkthroughResult:
    """Structured result of a walkthrough run, consumed by both CLI modes.

    ``snapshots`` maps each completed week N -> the cumulative DB-sourced
    :class:`~app.demo.oracle.Standings` after weeks 1..N. ``passed`` is True iff
    every snapshot matched the oracle (it is False only when ``assert_oracle`` is
    disabled and a mismatch was recorded; with ``assert_oracle`` a mismatch
    raises before the result is returned).
    """

    snapshots: dict[int, Standings] = field(default_factory=dict)
    passed: bool = True


def _as_aware(dt: datetime | None) -> datetime | None:
    """Re-attach UTC to a naive datetime read back from the store.

    ``DateTime(timezone=True)`` round-trips NAIVE on SQLite (Postgres preserves
    tz). Re-declared locally (mirrors ``refresh._as_aware`` /
    ``pick_submission._as_aware``) rather than importing a private helper. The
    normalized copy is never persisted, leaving production-on-Postgres
    unaffected.
    """
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def setup(session: Session) -> None:
    """Seed demo state, then reset every game to a fresh pre-game state.

    Runs the EXISTING seeds in order — :func:`seed_teams` ->
    :func:`import_fixture_2025` -> :func:`seed_bots` — then resets all ``Game``
    rows to SCHEDULED with no scores (mirroring ``test_refresh_games`` setUp) so
    ``refresh_games`` has non-FINAL weeks to poll, simulating the schedule being
    ingested before kickoff. Idempotent/repeatable: a second ``setup`` leaves the
    same row counts (the seeds upsert on natural keys; the reset is a no-op the
    second time once games are re-set to SCHEDULED). Commits the reset.
    """
    seed_teams(session)
    import_fixture_2025(session)
    seed_bots(session)

    for game in session.exec(select(Game)).all():
        game.status = GameStatus.SCHEDULED
        game.home_score = None
        game.away_score = None
        session.add(game)
    session.commit()


def _bot_users(session: Session) -> dict[str, User]:
    """Map each walkthrough bot's ``display_name`` -> its persisted ``User``."""
    return {
        u.display_name: u
        for u in session.exec(select(User)).all()
        if u.display_name in BOT_NAMES
    }


def _event_id_to_game_id(games: Iterable[Game]) -> dict[int, int]:
    """Build ``{espn_event_id: Game.id}`` for resolving BOT_PICKS to PKs."""
    return {g.espn_event_id: g.id for g in games}


def _submit_week_picks(
    session: Session, *, season: int, week: int, bots: Mapping[str, User]
) -> None:
    """Submit every bot's week-N preordained picks through ``submit_picks``.

    Resolves each :class:`~app.seeds.data.bot_picks_2025.BotPick`'s
    ``espn_event_id`` to the seeded ``Game.id`` and submits via the REAL
    submission service (window/lock/validation enforced). Commits after each
    bot. Any rejection propagates — it is a real integration bug, never
    swallowed.
    """
    week_games = list(
        session.exec(
            select(Game).where(Game.season == season, Game.week == week)
        ).all()
    )
    event_to_id = _event_id_to_game_id(week_games)

    for name, user in bots.items():
        records = BOT_PICKS.get(name, {}).get(week)
        if not records:
            continue
        items = [
            PickItem(
                game_id=event_to_id[bp.espn_event_id],
                pick_type=bp.pick_type,
                is_mortal_lock=bp.is_mortal_lock,
            )
            for bp in records
        ]
        assert user.id is not None  # a persisted bot always has an id
        submit_picks(
            session,
            user_id=user.id,
            season=season,
            week=week,
            items=items,
        )
        session.commit()


def compute_db_standings(
    session: Session, *, season: int, weeks: Sequence[int]
) -> Standings:
    """Build ACTUAL standings from PERSISTED DB rows (the integration side).

    Loads the walkthrough bots, their persisted ``Pick`` rows for ``weeks``, and
    the (now-final) ``Game`` rows, normalizes kickoff tz with the ``_as_aware``
    pattern, and scores each bot/week via the SAME
    :func:`app.services.scoring.score_week` the oracle uses — producing the SAME
    :class:`~app.demo.oracle.Standings` shape (ordered by ``(-season_total,
    display_name)``) so it compares ``==`` with
    :func:`app.demo.oracle.compute_standings`.

    Crucially this reads REAL persisted picks (NOT ``BOT_PICKS``) — that is what
    makes the comparison an integration proof rather than a tautology.
    """
    games = list(session.exec(select(Game).where(Game.season == season)).all())
    for g in games:
        g.kickoff_at = _as_aware(g.kickoff_at)  # in-memory copy only; not committed
    games_by_pk = games_by_pk_index(games)

    # week number -> week_id for the requested weeks, scoped to the season.
    week_rows = {
        w.week: w.id
        for w in session.exec(select(Week).where(Week.season == season)).all()
        if w.week in set(weeks)
    }

    bots = _bot_users(session)

    results: list[BotSeasonResult] = []
    for name, user in bots.items():
        weekly_scores: dict[int, int] = {}
        for week in weeks:
            week_id = week_rows.get(week)
            if week_id is None:
                continue
            picks = list(
                session.exec(
                    select(Pick).where(
                        Pick.user_id == user.id, Pick.week_id == week_id
                    )
                ).all()
            )
            if not picks:
                # No persisted picks for this bot/week -> contributes no week
                # entry, matching the oracle (which only emits weeks present in
                # BOT_PICKS for that bot).
                continue
            weekly_scores[week] = score_week(games_by_pk, picks)
        results.append(
            BotSeasonResult(
                display_name=name,
                weekly_scores=weekly_scores,
                season_total=sum(weekly_scores.values()),
            )
        )

    results.sort(key=lambda r: (-r.season_total, r.display_name))
    return Standings(results=tuple(results))


def _oracle_for_weeks(games: list[Game], weeks: Sequence[int]) -> Standings:
    """Oracle standings restricted to ``weeks`` (the expected side)."""
    restricted = {
        name: {w: recs for w, recs in by_week.items() if w in set(weeks)}
        for name, by_week in BOT_PICKS.items()
    }
    return compute_standings(restricted, games)


def run_walkthrough(
    session: Session,
    *,
    weeks: Sequence[int] = DEFAULT_WEEKS,
    source_factory: SourceFactory = lambda offset: Demo2025Source(offset),  # type: ignore[arg-type]
    assert_oracle: bool = True,
) -> WalkthroughResult:
    """Run the season walkthrough over ``weeks`` and (optionally) assert == oracle.

    Assumes :func:`setup` has populated the session. For each week N in order:
    position WINDOW_OPEN, ``refresh_games``, submit every bot's picks via
    ``submit_picks``, position ALL_WEEK_FINAL, ``refresh_games`` to finalize,
    then compute the cumulative DB-sourced standings for weeks 1..N and compare
    with the oracle. With ``assert_oracle`` a mismatch raises
    :class:`WalkthroughAssertionError` (the runnable proof's non-zero exit);
    without it the mismatch is recorded as ``passed=False``.

    Returns a :class:`WalkthroughResult` carrying the per-week cumulative
    snapshots so both CLI modes consume the same core.
    """
    weeks_kickoffs = load_fixture_kickoffs()
    season = _season(session)

    completed: list[int] = []
    snapshots: dict[int, Standings] = {}
    passed = True

    for week in weeks:
        bots = _bot_users(session)

        # (1) Window open for week N: stamp windows, prior weeks reflect FINAL.
        open_offset = compute_offset(
            datetime.now(timezone.utc),
            target_week=week,
            phase=DemoPhase.WINDOW_OPEN_FOR_WEEK,
            weeks_kickoffs=weeks_kickoffs,
        )
        refresh_games(session, source_factory(open_offset))  # type: ignore[arg-type]
        session.commit()

        # (2) Submit each bot's picks through the REAL submission service.
        _submit_week_picks(session, season=season, week=week, bots=bots)

        # (3) All week N final: finalize the games with their real scores.
        final_offset = compute_offset(
            datetime.now(timezone.utc),
            target_week=week,
            phase=DemoPhase.ALL_WEEK_FINAL,
            weeks_kickoffs=weeks_kickoffs,
        )
        refresh_games(session, source_factory(final_offset))  # type: ignore[arg-type]
        session.commit()

        # (4) Compare cumulative DB-sourced standings (weeks 1..N) to the oracle.
        completed.append(week)
        actual = compute_db_standings(session, season=season, weeks=completed)
        all_games = list(
            session.exec(select(Game).where(Game.season == season)).all()
        )
        for g in all_games:
            g.kickoff_at = _as_aware(g.kickoff_at)
        expected = _oracle_for_weeks(all_games, completed)

        snapshots[week] = actual
        if actual != expected:
            passed = False
            if assert_oracle:
                raise WalkthroughAssertionError(
                    week=week, actual=actual, expected=expected
                )

    return WalkthroughResult(snapshots=snapshots, passed=passed)


def _season(session: Session) -> int:
    """Resolve the single season present in the seeded Game rows."""
    seasons = {g.season for g in session.exec(select(Game)).all()}
    if len(seasons) != 1:
        raise ValueError(
            f"expected exactly one season in the demo DB, found {sorted(seasons)}"
        )
    return next(iter(seasons))
