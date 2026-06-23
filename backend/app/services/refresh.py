"""Source-agnostic game reconciliation — the ``refresh_games`` poller core.

This is the first DB-writing piece of the pick'em domain and build-order step #2
of the season walkthrough. It consumes the :class:`~app.scoreboard.port.ScoreboardSource`
port and idempotently reconciles live results into existing :class:`~app.models.Game`
rows, then stamps the persisted pick window onto :class:`~app.models.Week` rows.

Design — service-layer, DB-focused, source-injected (mirrors the sibling pure
services' value-object style while still owning the DB writes):

* The business logic lives here and operates on a **passed-in** ``Session``. The
  Celery task in :mod:`app.tasks` is a thin wrapper that opens the session and
  resolves the production default source. This module NEVER constructs an ESPN
  adapter and NEVER imports the network layer or :mod:`app.config` — it is
  source-agnostic and unit-testable offline against
  :class:`~app.scoreboard.demo.Demo2025Source`.
* ``source`` is a required :class:`~app.scoreboard.port.ScoreboardSource` at this
  layer (no config dependency). The config-resolved production default lives only
  in the task wrapper.
* ``now`` is injected (defaulting to the real UTC clock) and is used ONLY for the
  window-stamping decisions, mirroring the established ``pick_window`` /
  ``derive_status`` "inject now, default real" pattern. The status/score
  reconciliation itself never reads the clock — it copies whatever the source
  reports (the source already derived status from the real clock + its offset).

Reconciliation rules (idempotent, match by ``espn_event_id``):

* Only weeks with at least one non-FINAL game are polled; a week whose rows are
  all FINAL is skipped entirely (no fetch).
* For each polled week the returned games are matched to existing rows by integer
  ``espn_event_id``; only fields that actually differ are written so an unchanged
  re-run produces zero dirty rows. A source game that withholds a score (``None``)
  never nulls a present score, but its status transition is still applied. Source
  home/away scores map onto the row's existing home/away sides (the row's
  assignment is authoritative; the poller does not re-match teams).

Window stamping (reuses :func:`app.services.pick_window.compute_window`):

* Every week's ``window_closes_at`` is stamped to its earliest kickoff.
* Week N+1's ``window_opens_at`` is stamped (= the kickoff-time approximation
  from :func:`compute_window`) only once week N is fully FINAL. Week 1 has no
  predecessor so its open boundary stays ``None``.
* All stamping is idempotent (write only when the stored value differs).

Failure contract: each week's fetch is guarded; a
:class:`~app.scoreboard.port.ScoreboardFetchError` on one week is recorded in the
result and never aborts or corrupts the other weeks.

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use the venv interpreter ``.venv/bin/python``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlmodel import Session, select

from app.models import Game, GameStatus, Week
from app.scoreboard.port import ScoreboardFetchError, ScoreboardSource
from app.scoreboard.types import ScoreboardGame
from app.services.pick_window import compute_window


@dataclass(frozen=True)
class RefreshResult:
    """Immutable summary of one ``refresh_games`` run (value object).

    Mirrors the sibling services' frozen-dataclass result style
    (``scoring.GradeResult``):

    * ``weeks_polled`` — count of weeks that needed a fetch (non-FINAL rows) and
      were fetched without error.
    * ``games_updated`` — count of Game rows whose status/scores changed.
    * ``windows_stamped`` — count of Week window field-writes (opens_at/closes_at).
    * ``failed_weeks`` — the ``(season, week)`` pairs whose fetch raised
      :class:`~app.scoreboard.port.ScoreboardFetchError`.
    """

    weeks_polled: int = 0
    games_updated: int = 0
    windows_stamped: int = 0
    failed_weeks: tuple[tuple[int, int], ...] = field(default_factory=tuple)


def _as_aware(dt: datetime | None) -> datetime | None:
    """Re-attach UTC to a naive datetime read back from the store.

    ``DateTime(timezone=True)`` round-trips NAIVE on SQLite (Postgres preserves
    tz). The window math compares against tz-aware values, so this normalizes a
    naive value to UTC for the comparison ONLY — the normalized copy is never
    persisted back, leaving production-on-Postgres unaffected.
    """
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _normalized_games(games: list[Game]) -> list[Game]:
    """Return shallow copies of ``games`` with tz-aware kickoffs for window math.

    Avoids mutating (and thus dirtying/persisting) the live rows: each copy
    carries the same data with ``kickoff_at`` re-attached to UTC when naive, so
    :func:`compute_window` can be called safely on store-read rows.
    """
    out: list[Game] = []
    for g in games:
        out.append(
            Game(
                espn_event_id=g.espn_event_id,
                week_id=g.week_id,
                season=g.season,
                week=g.week,
                home_team_id=g.home_team_id,
                away_team_id=g.away_team_id,
                kickoff_at=_as_aware(g.kickoff_at),
                status=g.status,
            )
        )
    return out


def _scores_by_event_id(games: list[ScoreboardGame]) -> dict[int, ScoreboardGame]:
    """Index the source's games by integer ``espn_event_id`` (skip Nones)."""
    indexed: dict[int, ScoreboardGame] = {}
    for sg in games:
        if sg.espn_event_id is None:
            continue
        indexed[int(sg.espn_event_id)] = sg
    return indexed


def _reconcile_game(row: Game, src: ScoreboardGame) -> bool:
    """Apply the source's status/scores onto ``row``, writing only what changed.

    Returns ``True`` iff any field was actually written. A source side whose
    score is ``None`` (a live game withholding its score) never overwrites a
    present score, but the status transition is still applied.
    """
    changed = False

    if row.status != src.status:
        row.status = src.status
        changed = True

    if src.home.score is not None and row.home_score != src.home.score:
        row.home_score = src.home.score
        changed = True

    if src.away.score is not None and row.away_score != src.away.score:
        row.away_score = src.away.score
        changed = True

    return changed


def refresh_games(
    session: Session,
    source: ScoreboardSource,
    *,
    now: datetime | None = None,
) -> RefreshResult:
    """Reconcile non-final games and stamp pick windows for the polled season.

    :param session: an open SQLModel ``Session`` the caller owns (the task
        wrapper opens ``task_session()``; tests pass an in-memory session). This
        function adds/updates rows but does NOT commit — the caller commits.
    :param source: the injected :class:`~app.scoreboard.port.ScoreboardSource`.
        Production passes the ESPN adapter (resolved in the task wrapper); tests
        pass :class:`~app.scoreboard.demo.Demo2025Source`.
    :param now: tz-aware UTC instant used only for window-stamping decisions;
        defaults to ``datetime.now(timezone.utc)``.
    :returns: a :class:`RefreshResult` summarizing the run.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    all_games = list(session.exec(select(Game)).all())
    if not all_games:
        return RefreshResult()

    # Group rows by (season, week) once.
    by_week: dict[tuple[int, int], list[Game]] = {}
    for g in all_games:
        by_week.setdefault((g.season, g.week), []).append(g)

    # Weeks that need polling: at least one non-FINAL row. Fully-FINAL weeks are
    # skipped entirely (no fetch) — the idempotency + don't-refetch guarantee.
    needy = sorted(
        key
        for key, rows in by_week.items()
        if any(r.status != GameStatus.FINAL for r in rows)
    )

    games_updated = 0
    weeks_polled = 0
    failed_weeks: list[tuple[int, int]] = []

    for season, week in needy:
        try:
            fetched = source.fetch_week(season, week)
        except ScoreboardFetchError:
            failed_weeks.append((season, week))
            continue

        weeks_polled += 1
        src_by_id = _scores_by_event_id(fetched)
        for row in by_week[(season, week)]:
            src = src_by_id.get(row.espn_event_id)
            if src is None:
                continue
            if _reconcile_game(row, src):
                session.add(row)
                games_updated += 1

    # --- Window stamping (after reconciliation, over every week in the DB) ---
    seasons = {season for season, _ in by_week}
    windows_stamped = 0

    for season in seasons:
        season_weeks = sorted(w for s, w in by_week if s == season)
        weeks_by_number = {w: by_week[(season, w)] for w in season_weeks}

        # Persisted Week rows, indexed by week number.
        week_rows = {
            wr.week: wr
            for wr in session.exec(
                select(Week).where(Week.season == season)
            ).all()
        }

        for idx, wk in enumerate(season_weeks):
            week_row = week_rows.get(wk)
            if week_row is None:
                continue

            this_games = weeks_by_number[wk]
            prev_games = (
                weeks_by_number[season_weeks[idx - 1]] if idx > 0 else None
            )

            # closes_at: always derivable from this week's kickoffs.
            try:
                window = compute_window(
                    _normalized_games(this_games),
                    _normalized_games(prev_games) if prev_games else None,
                )
            except ValueError:
                # No kickoff to close on — nothing to stamp for this week.
                continue

            if week_row.window_closes_at != window.close_at:
                # Compare tz-normalized to avoid a spurious rewrite on SQLite.
                if _as_aware(week_row.window_closes_at) != window.close_at:
                    week_row.window_closes_at = window.close_at
                    session.add(week_row)
                    windows_stamped += 1

            # opens_at for THIS week is stamped only once the PREVIOUS week is
            # fully FINAL. Week 1 (idx 0) has no predecessor -> stays None.
            if prev_games is not None and all(
                r.status == GameStatus.FINAL for r in prev_games
            ):
                open_at = window.open_at
                if open_at is not None and _as_aware(
                    week_row.window_opens_at
                ) != open_at:
                    week_row.window_opens_at = open_at
                    session.add(week_row)
                    windows_stamped += 1

    return RefreshResult(
        weeks_polled=weeks_polled,
        games_updated=games_updated,
        windows_stamped=windows_stamped,
        failed_weeks=tuple(failed_weeks),
    )
