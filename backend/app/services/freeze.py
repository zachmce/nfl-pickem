"""Source-agnostic manual line-freeze — "retrieve and freeze week N lines NOW".

This is the on-demand counterpart to the COMPUTED odds-freeze clock that ships
from the odds poller (:mod:`app.services.odds`). Where ``is_odds_frozen`` derives
freeze from ``min(noon-ET-Wed, pick_lock)`` against the real clock,
:func:`freeze_week` lets an admin LOCK a week's line immediately — before the
ephemeral DraftKings line vanishes — by (1) re-snapshotting the current odds onto
the week's :class:`~app.models.Game` rows and (2) flipping the week's
:class:`~app.models.Week` ``lines_frozen`` OVERRIDE flag.

Why ``Week.lines_frozen`` (not ``Game.odds_frozen``, not a timestamp): the
override is exactly the flag :func:`app.services.odds.is_odds_frozen` reads FIRST
(``if week_row.lines_frozen: return True``) before consulting the computed
``freeze_at`` clock. Setting it therefore makes the week frozen regardless of the
cadence — the "manual freeze beats the clock" requirement. ``Game.odds_frozen`` is
a per-game default the pollers never flip and ``is_odds_frozen`` never reads, so
it is NOT the override and we leave it at its model default. No new column / no
migration.

Why ``ingest._apply_odds`` (not ``odds.reconcile_odds``): the QT-1 snapshot path
:func:`app.services.ingest._apply_odds` persists BOTH the chosen provider NAME and
ID from the same selected odds item; the poll-path ``reconcile_odds`` writes only
the name. The manual freeze must carry the same drift-proof provider name+id, so
it REUSES the QT-1 snapshot helper rather than re-implementing normalization.

Order of operations: snapshot the odds FIRST (while still writable), THEN flip
``lines_frozen``. If the flag were set first, the odds write path's freeze refusal
would block the re-snapshot.

Source-agnostic invariant (mirrors ``ingest.py`` / ``odds.py``): this module
imports ONLY :mod:`app.models`, the scoreboard port/types, and the shared
snapshot/team-map helpers (:func:`app.services.ingest._apply_odds`,
:func:`app.seeds.fixture_2025._build_team_map`). It imports NO settings module, NO
ESPN adapter, and has NO demo-mode branch — the gated production source
resolution lives only in the thin Celery task wrapper
(:func:`app.tasks.freeze_week_task`). Importing this module is side-effect-free.

Commit policy: like ``ingest_season`` (and ``import_fixture_2025``),
:func:`freeze_week` commits ONCE at the end — this is a one-shot manual action, so
owning its commit keeps the worker-callable contract simple.

Fetch-failure contract: if the per-week ``fetch_week`` raises
:class:`~app.scoreboard.port.ScoreboardFetchError`, the week is NOT locked
(``lines_frozen`` stays False) and ``FreezeResult.failed`` is set — a fetch
failure must never lock a week against stale/missing data.

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use the venv interpreter ``.venv/bin/python``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlmodel import Session, select

from app.models import Game, Week
from app.scoreboard.port import ScoreboardFetchError, ScoreboardSource
from app.seeds.fixture_2025 import _build_team_map
from app.services.ingest import _apply_odds


@dataclass(frozen=True)
class FreezeResult:
    """Immutable summary of one :func:`freeze_week` run (value object).

    Mirrors :class:`app.services.ingest.IngestResult` /
    :class:`app.services.odds.OddsResult` styling:

    * ``season`` / ``week`` — the week that was targeted.
    * ``games_updated`` — count of matched ``Game`` rows whose odds snapshot was
      (re)written this run.
    * ``already_frozen`` — True when the week was ALREADY ``lines_frozen`` before
      this run (an idempotent re-freeze).
    * ``failed`` — True when the per-week ``fetch_week`` raised
      :class:`~app.scoreboard.port.ScoreboardFetchError`; the week is then NOT
      locked (no freeze against stale/missing data).
    """

    season: int
    week: int
    games_updated: int = 0
    already_frozen: bool = False
    failed: bool = False


def _snapshot_changed(before: tuple, after: tuple) -> bool:
    """Whether the snapshot-relevant odds tuple changed between before/after."""
    return before != after


def _odds_signature(row: Game) -> tuple:
    """A comparable signature of the odds-bearing fields on ``row``.

    Used to count ``games_updated`` (rows whose snapshot actually changed), so an
    idempotent re-freeze reports zero new writes.
    """
    return (
        row.spread,
        row.total,
        row.favorite_team_id,
        row.underdog_team_id,
        row.odds_provider,
        row.odds_provider_id,
        row.odds_captured_at,
    )


def freeze_week(
    session: Session,
    source: ScoreboardSource,
    season: int,
    week: int,
    *,
    now: datetime | None = None,
) -> FreezeResult:
    """Re-snapshot ``(season, week)``'s current odds, then LOCK the week.

    Fetches the one week from the injected ``source``, snapshots the current odds
    onto each matched :class:`~app.models.Game` row by REUSING the QT-1 snapshot
    helper :func:`app.services.ingest._apply_odds` (provider NAME + ID,
    positive-magnitude spread, total, ``odds_captured_at = now``), then sets the
    owning :class:`~app.models.Week`'s ``lines_frozen = True`` so
    :func:`app.services.odds.is_odds_frozen` returns True regardless of the
    computed ``freeze_at`` clock. Commits once at the end.

    Idempotent: a second run snapshots the (unchanged) line again as a harmless
    no-op, leaves ``lines_frozen`` True, and reports ``already_frozen=True`` with
    ``games_updated=0``.

    :param session: an open SQLModel session the caller owns. This function
        commits once at the end (matching ``ingest_season``).
    :param source: the injected :class:`~app.scoreboard.port.ScoreboardSource`
        (production passes the ESPN adapter, resolved in the task wrapper; tests
        pass a synthetic fake).
    :param season: the NFL season year (e.g. ``2026``).
    :param week: the regular-season week to freeze (``1``..``18``).
    :param now: tz-aware UTC instant used as the ``odds_captured_at`` stamp;
        defaults to ``datetime.now(timezone.utc)``.
    :raises ValueError: with a leading ``week_not_found`` stable code if no
        :class:`~app.models.Week` exists for ``(season, week)`` — you cannot
        freeze a week that has not been ingested yet.
    :returns: a :class:`FreezeResult` summarizing the run.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    week_row = session.exec(
        select(Week).where(Week.season == season, Week.week == week)
    ).first()
    if week_row is None:
        raise ValueError(
            f"week_not_found: no Week row for (season={season}, week={week}); "
            "ingest the season before freezing its lines."
        )

    already_frozen = bool(week_row.lines_frozen)
    if already_frozen:
        # The week's line is ALREADY locked: the odds write path would refuse a
        # re-snapshot anyway (belt-and-suspenders), so a re-freeze is a true
        # no-op. Report it without touching any row or re-fetching the source.
        return FreezeResult(
            season=season,
            week=week,
            games_updated=0,
            already_frozen=True,
            failed=False,
        )

    week_games = list(
        session.exec(
            select(Game).where(Game.season == season, Game.week == week)
        ).all()
    )

    team_map = _build_team_map(session)

    try:
        fetched = source.fetch_week(season, week)
    except ScoreboardFetchError:
        # Do NOT lock a week against stale/missing data — record the failure and
        # leave lines_frozen untouched.
        return FreezeResult(
            season=season,
            week=week,
            games_updated=0,
            already_frozen=already_frozen,
            failed=True,
        )

    # Index the fetched games by int espn_event_id (same shape ingest/odds use).
    src_by_id: dict[int, object] = {}
    for sg in fetched:
        if sg.espn_event_id is None:
            continue
        src_by_id[int(sg.espn_event_id)] = sg

    games_updated = 0
    for row in week_games:
        sg = src_by_id.get(row.espn_event_id)
        if sg is None:
            continue
        before = _odds_signature(row)
        # Reuse the QT-1 snapshot path: persists provider NAME + ID.
        _apply_odds(row, sg.odds, team_map, now=now)
        after = _odds_signature(row)
        if _snapshot_changed(before, after):
            session.add(row)
            games_updated += 1

    # Snapshot done -> NOW flip the override flag (after the writable snapshot).
    week_row.lines_frozen = True
    session.add(week_row)

    session.commit()

    return FreezeResult(
        season=season,
        week=week,
        games_updated=games_updated,
        already_frozen=already_frozen,
        failed=False,
    )


__all__ = ["FreezeResult", "freeze_week"]
