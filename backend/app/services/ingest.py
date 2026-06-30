"""Source-agnostic season-bootstrap ingest — the manual "ingest season XXXX now".

This is the schedule-CREATE path the pollers do NOT cover. ``refresh_games`` /
``reconcile_odds_games`` only UPDATE :class:`~app.models.Game` rows that already
exist (matched by ``espn_event_id``) in weeks that already exist — they never
CREATE a Week/Game. :func:`ingest_season` pulls every regular-season week of a
season from an injected :class:`~app.scoreboard.port.ScoreboardSource` and CREATES
the Week + Game skeleton (teams resolved, status, scores, kickoff) plus the
current odds snapshot AT INGEST TIME, for a season that does not yet exist in the
DB.

It is modeled directly on the two proven precedents:

* :func:`app.seeds.fixture_2025.import_fixture_2025` — the existing CREATE path
  (Week upsert on ``(season, week)`` + flush so the surrogate PK exists for
  ``Game.week_id``; Game upsert on the unique ``espn_event_id``; positive-magnitude
  Decimal spread; favorite/underdog FK resolution; "commit once at the end"). We
  REUSE its ``_build_team_map`` / ``TeamsNotSeededError`` rather than re-declaring
  them, so the team-resolution contract can never drift.
* :func:`app.services.refresh.refresh_games` — the per-week fetch + guard
  structure (one ``fetch_week`` per week, a :class:`ScoreboardFetchError` on one
  week is recorded and never aborts the others) and the None-never-nulls rule.

Source-agnostic invariant (mirrors ``refresh.py`` / ``odds.py``): this module
imports ONLY :mod:`app.models`, the scoreboard port/types, and
:mod:`app.seeds.fixture_2025` (for the shared team-map helpers). It imports NO
:mod:`app.config`, NO ESPN adapter, and has NO ``IS_DEMO_DATA`` branch — the gated
production source resolution lives only in the thin Celery task wrapper
(:func:`app.tasks.ingest_season_task`). Importing this module is side-effect-free
(it opens no DB engine; the wrapper/CLI opens the session).

Commit policy: like ``import_fixture_2025`` (and unlike the pollers, which leave
the commit to their wrapper), :func:`ingest_season` commits ONCE at the end —
this is a one-shot manual action, so owning its commit keeps the worker-callable
contract simple.

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use the venv interpreter ``.venv/bin/python``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal

from sqlmodel import Session, select

from app.models import Game, Week
from app.scoreboard.port import ScoreboardFetchError, ScoreboardSource
from app.scoreboard.types import ScoreboardGame, ScoreboardOdds
from app.seeds.fixture_2025 import TeamsNotSeededError, _build_team_map

# Default regular-season weeks (1..18). The NFL regular season is 18 weeks; the
# source returns seasontype=2 per the ESPN adapter's URL template.
DEFAULT_WEEKS: range = range(1, 19)


@dataclass(frozen=True)
class IngestResult:
    """Immutable summary of one :func:`ingest_season` run (value object).

    Mirrors :class:`app.seeds.fixture_2025.ImportResult` /
    :class:`app.services.refresh.RefreshResult` styling:

    * ``weeks_present`` — count of ``Week`` rows for the season after the run
      (post-run total, like ``ImportResult.week_count``).
    * ``games_present`` — count of ``Game`` rows for the season after the run.
    * ``weeks_created`` — count of NEW ``Week`` rows this run created.
    * ``games_created`` — count of NEW ``Game`` rows this run created.
    * ``failed_weeks`` — the ``(season, week)`` pairs whose ``fetch_week`` raised
      :class:`~app.scoreboard.port.ScoreboardFetchError` (recorded, never fatal).
    """

    weeks_present: int = 0
    games_present: int = 0
    weeks_created: int = 0
    games_created: int = 0
    failed_weeks: tuple[tuple[int, int], ...] = field(default_factory=tuple)


def _resolve_team(team_map: dict[int, int], espn_team_id: str | None) -> int:
    """Resolve a source (string) espn team id to a seeded ``team.id`` PK.

    Raises :class:`~app.seeds.fixture_2025.TeamsNotSeededError` (naming the missing
    id) on an unseeded/unknown team rather than inserting an orphan FK — the same
    contract ``fixture_2025._resolve_team`` enforces, kept strict here because the
    CREATE path must never write a Game with a dangling team FK.
    """
    if espn_team_id is None:
        raise TeamsNotSeededError(
            "Source game is missing an espn_team_id; cannot resolve a team FK. "
            "Run `python -m app.seeds.teams` and verify the source team ids."
        )
    espn_team_id_int = int(espn_team_id)
    team_id = team_map.get(espn_team_id_int)
    if team_id is None:
        raise TeamsNotSeededError(
            f"No seeded Team for espn_team_id={espn_team_id_int}. "
            "Run `python -m app.seeds.teams` before ingesting the season."
        )
    return team_id


def _apply_scores(row: Game, src: ScoreboardGame) -> None:
    """Copy status/kickoff/scores from the source onto ``row`` (None-never-nulls).

    A source side whose score is ``None`` never nulls a present score (mirrors
    ``refresh._reconcile_game``); status and (when present) kickoff are always
    copied. On a freshly-created row every value is simply set.
    """
    row.status = src.status
    if src.kickoff_at is not None:
        row.kickoff_at = src.kickoff_at
    if src.home.score is not None:
        row.home_score = src.home.score
    if src.away.score is not None:
        row.away_score = src.away.score


def _apply_odds(
    row: Game,
    odds: ScoreboardOdds | None,
    team_map: dict[int, int],
    *,
    now: datetime,
) -> None:
    """Snapshot the source odds onto ``row`` at ``now`` (or leave odds None).

    Normalization mirrors :func:`app.seeds.fixture_2025.import_fixture_2025` /
    :func:`app.services.odds.reconcile_odds` EXACTLY: positive-magnitude Decimal
    spread (``Decimal(str(abs(...)))``), ``Decimal(str(total))``, favorite/underdog
    int FKs resolved from the source's STRING ids. The chosen provider's NAME
    (``odds.provider``) AND ID (``odds.provider_id``) are BOTH persisted — both
    handed in from the SAME selected odds item (the drift-proof DraftKings
    selection already happened in the adapter); the service never re-selects or
    hardcodes an id. ``odds_captured_at`` is stamped to ``now``.

    When ``odds`` is ``None`` every odds field is left ``None`` and no
    captured-at is stamped. ``odds_frozen`` is NOT touched here (QT-2's job —
    leave its model default).
    """
    if odds is None:
        return

    if odds.spread is not None:
        row.spread = Decimal(str(abs(odds.spread)))
    if odds.total is not None:
        row.total = Decimal(str(odds.total))
    if odds.favorite_team_id is not None:
        row.favorite_team_id = _resolve_team(team_map, odds.favorite_team_id)
    if odds.underdog_team_id is not None:
        row.underdog_team_id = _resolve_team(team_map, odds.underdog_team_id)
    row.odds_provider = odds.provider
    row.odds_provider_id = odds.provider_id
    row.odds_captured_at = now


def ingest_season(
    session: Session,
    source: ScoreboardSource,
    season: int,
    *,
    weeks: Iterable[int] = DEFAULT_WEEKS,
    now: datetime | None = None,
) -> IngestResult:
    """Idempotently CREATE the Week + Game skeleton for ``season`` from ``source``.

    Pulls each week in ``weeks`` with one ``source.fetch_week(season, week)``,
    CREATES one :class:`~app.models.Week` per week the source returns games for and
    one :class:`~app.models.Game` per event, resolving home/away (and
    favorite/underdog) FKs by ``espn_team_id`` against the seeded ``Team`` table.
    Each game gets the current odds snapshot (chosen provider NAME + ID,
    positive-magnitude spread, total, ``odds_captured_at = now``). Commits once at
    the end.

    Idempotent: Week is upserted on ``(season, week)``, Game on the unique
    ``espn_event_id`` — a re-run creates no duplicate rows and a ``None`` source
    field never nulls an already-present value. A per-week
    :class:`~app.scoreboard.port.ScoreboardFetchError` is recorded in
    ``failed_weeks`` and never aborts the other weeks.

    :param session: an open SQLModel session the caller owns. This function
        commits once at the end (matching ``import_fixture_2025``).
    :param source: the injected :class:`~app.scoreboard.port.ScoreboardSource`
        (production passes the ESPN adapter, resolved in the task wrapper; tests
        pass a synthetic fake).
    :param season: the NFL season year to ingest (e.g. ``2026``).
    :param weeks: the regular-season weeks to pull (default ``range(1, 19)``).
    :param now: tz-aware UTC instant used as the ``odds_captured_at`` stamp;
        defaults to ``datetime.now(timezone.utc)``.
    :raises TeamsNotSeededError: if a source game references an unseeded team.
    :returns: an :class:`IngestResult` summarizing the run.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    team_map = _build_team_map(session)

    weeks_created = 0
    games_created = 0
    failed_weeks: list[tuple[int, int]] = []

    for week in weeks:
        try:
            fetched = source.fetch_week(season, week)
        except ScoreboardFetchError:
            failed_weeks.append((season, week))
            continue

        if not fetched:
            # A week the source genuinely has no games for creates no rows but
            # does not abort the others.
            continue

        for src in fetched:
            if src.espn_event_id is None:
                # Defensive: an event with no id cannot be upserted by its
                # natural key — skip it rather than create an unkeyed row.
                continue
            espn_event_id = int(src.espn_event_id)

            # 1. Upsert the Week on (season, week); flush so its surrogate PK
            #    exists for Game.week_id (exactly as import_fixture_2025 does).
            week_row = session.exec(
                select(Week).where(Week.season == season, Week.week == week)
            ).first()
            if week_row is None:
                week_row = Week(season=season, week=week)
                session.add(week_row)
                session.flush()
                weeks_created += 1

            # 2. Upsert the Game on its unique espn_event_id.
            game = session.exec(select(Game).where(Game.espn_event_id == espn_event_id)).first()
            if game is None:
                game = Game(espn_event_id=espn_event_id)
                session.add(game)
                games_created += 1

            game.week_id = week_row.id
            game.season = season
            game.week = week
            game.home_team_id = _resolve_team(team_map, src.home.espn_team_id)
            game.away_team_id = _resolve_team(team_map, src.away.espn_team_id)
            _apply_scores(game, src)
            _apply_odds(game, src.odds, team_map, now=now)

    session.commit()

    weeks_present = len(session.exec(select(Week).where(Week.season == season)).all())
    games_present = len(session.exec(select(Game).where(Game.season == season)).all())
    return IngestResult(
        weeks_present=weeks_present,
        games_present=games_present,
        weeks_created=weeks_created,
        games_created=games_created,
        failed_weeks=tuple(failed_weeks),
    )
