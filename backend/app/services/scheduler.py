"""The scheduler seam — first-class polling jobs + a shared per-week fetch.

This module factors the app's scheduled *polling* concerns into a clean,
source-agnostic seam so multiple sibling jobs can coexist: today the single
``refresh-games-poller`` (scores) lives here as one :class:`PollingJob`; the
odds poller (a different cadence + a different stop condition) will join the
``POLLING_JOBS`` registry next as a sibling, WITHOUT touching the scores job,
``services/refresh.py``, or the scores beat entry.

Design — this is a PURE REFACTOR seam, not new behavior:

* A :class:`PollingJob` is an immutable value object describing ONE scheduled
  polling concern: a stable ``name``, the Celery ``beat_name`` key, the
  ``schedule_seconds`` cadence, the registered ``task_name`` the beat dispatches,
  a ``needy`` predicate (which weeks to poll), and a ``reconcile`` callable (apply
  one fetched week's data). ``app.celery_app`` builds its ``beat_schedule`` from
  this registry; ``app.tasks`` dispatches a job's ``reconcile``.
* :data:`POLLING_JOBS` is the ordered registry. In THIS task it holds exactly one
  job — :data:`SCORES_JOB` — whose ``reconcile`` delegates to the existing
  :func:`app.services.refresh.refresh_games` core so the reconcile + window
  stamping behavior is byte-for-byte unchanged.
* :func:`fetch_needy_weeks` is the shared per-week fetch: it issues exactly ONE
  ``source.fetch_week`` per needy week and returns the successes keyed by
  ``(season, week)`` plus the failed keys. A future odds reconciler reuses this
  so scores + odds can share ONE network call per week instead of fetching twice.

SOURCE-AGNOSTIC INVARIANT (asserted by a guard test): this module imports ONLY
from ``app.models``, ``app.scoreboard`` port/types (NOT the ESPN adapter), and
``app.services.*``. It NEVER imports the settings module, the ESPN adapter, or
the network layer, and it contains NO demo-mode gate — the gated source
resolution stays in the thin task wrapper (:mod:`app.tasks`), exactly as it does
today. (The guard test scans this file for those forbidden tokens, so they are
deliberately not spelled out literally anywhere below.)

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use the venv interpreter ``.venv/bin/python``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from sqlmodel import Session

from app.models import Game
from app.scoreboard.port import ScoreboardFetchError, ScoreboardSource
from app.scoreboard.types import ScoreboardGame
from app.services.odds import OddsResult, odds_needy_weeks, reconcile_odds_games
from app.services.refresh import (
    RefreshResult,
    group_games_by_week,
    needy_weeks,
    refresh_games,
)

# A grouped-rows mapping: (season, week) -> the Game rows for that week.
GroupedRows = dict[tuple[int, int], list[Game]]
WeekKey = tuple[int, int]


@dataclass(frozen=True)
class PollingJob:
    """One scheduled polling concern as a first-class, source-agnostic unit.

    Fields:

    * ``name`` — a stable identifier for the job (logging / future registry keys).
    * ``beat_name`` — the Celery ``beat_schedule`` key (the beat entry name).
    * ``schedule_seconds`` — the beat cadence, in seconds (float).
    * ``task_name`` — the registered Celery task string the beat dispatches.
    * ``needy`` — predicate selecting which weeks to poll (the job's "when to
      poll"). The scores job's needy is ``(grouped_rows) -> list[(season,
      week)]``; the odds job's needy additionally consults the persisted ``Week``
      rows + ``now`` for its freeze stop condition, so the field is typed loosely
      (``Callable[..., list[WeekKey]]``) to admit both shapes rather than forcing
      one job's signature on the other.
    * ``reconcile`` — callable ``(session, source, *, now) -> <result>`` that
      applies the fetched data and returns a summary (the job's "what a tick
      does"). The scores job delegates this to ``refresh_games`` (a
      ``RefreshResult``); the odds job delegates to the odds service (its own
      ``OddsResult``). The return type is loose (``Any``) so a sibling job's
      result value object need not subclass ``RefreshResult``.

    A sibling odds job slots into :data:`POLLING_JOBS` by supplying its OWN
    ``needy``/``reconcile``/cadence — without disturbing this one.
    """

    name: str
    beat_name: str
    schedule_seconds: float
    task_name: str
    needy: Callable[..., list[WeekKey]] = field(compare=False)
    reconcile: Callable[..., Any] = field(compare=False)


def fetch_needy_weeks(
    source: ScoreboardSource,
    weeks: list[WeekKey],
) -> tuple[dict[WeekKey, list[ScoreboardGame]], set[WeekKey]]:
    """Fetch each needy week ONCE, returning successes keyed by week + failures.

    Issues exactly one :meth:`ScoreboardSource.fetch_week` call per ``(season,
    week)`` in ``weeks`` (preserving order). A week whose fetch raises
    :class:`~app.scoreboard.port.ScoreboardFetchError` is recorded in the failed
    set — NOT raised — mirroring the existing per-week failure contract, and is
    excluded from the returned successes; the other weeks are unaffected.

    This is the shared seam: a future odds reconciler can consume the SAME fetched
    games (the port's ``fetch_week`` already returns odds in the same call) so
    scores + odds share one network call per week rather than fetching twice.

    :returns: ``(fetched, failed)`` where ``fetched`` maps ``(season, week)`` to
        the source's games for that week and ``failed`` is the set of week keys
        whose fetch raised.
    """
    fetched: dict[WeekKey, list[ScoreboardGame]] = {}
    failed: set[WeekKey] = set()
    for season, week in weeks:
        try:
            fetched[(season, week)] = source.fetch_week(season, week)
        except ScoreboardFetchError:
            failed.add((season, week))
    return fetched, failed


def _scores_needy(by_week: GroupedRows) -> list[WeekKey]:
    """The scores job's needy predicate: weeks with a non-FINAL row.

    Routed through :func:`app.services.refresh.needy_weeks` (the single home of
    the non-FINAL selection) so the scheduler's scores job and the reconcile core
    can never drift. Stop condition: a week stops being needy once all its games
    are FINAL.
    """
    return needy_weeks(by_week)


def _scores_reconcile(
    session: Session,
    source: ScoreboardSource,
    *,
    now: datetime | None = None,
) -> RefreshResult:
    """The scores job's reconcile: delegate to the unchanged ``refresh_games`` core.

    Keeping the delegation here (rather than re-expressing the reconcile) means
    the reconcile + window-stamping behavior — and ``refresh_games``' signature
    and return value — stay byte-for-byte what they are today. ``group_games_by_week``
    is re-exported via this module's imports so a future caller can pre-group rows
    and feed :func:`fetch_needy_weeks`; the scores reconcile keeps using the core.
    """
    return refresh_games(session, source, now=now)


# The scores cadence has ONE home: app.celery_app.REFRESH_GAMES_INTERVAL_SECONDS;
# the celery beat is built FROM this registry there (see app/celery_app.py), so
# the 60.0s below is the same value that beat entry will emit.
SCORES_JOB = PollingJob(
    name="scores",
    beat_name="refresh-games-poller",
    schedule_seconds=60.0,
    task_name="app.tasks.refresh_games",
    needy=_scores_needy,
    reconcile=_scores_reconcile,
)


def _odds_needy(by_week: GroupedRows, **kwargs: Any) -> list[WeekKey]:
    """The odds job's needy predicate: not-yet-frozen weeks with games.

    Delegates to :func:`app.services.odds.odds_needy_weeks` (an ``app.services.*``
    import the source-agnostic guard allows). The odds stop condition is FREEZE
    (not FINAL), so this consults the persisted ``Week`` rows + ``now`` passed
    through ``kwargs`` (``week_rows_by_key=`` and ``now=``). The odds reconcile
    entry computes the same predicate internally, so this is the registry-facing
    view for callers that pre-group rows.
    """
    return odds_needy_weeks(by_week, kwargs["week_rows_by_key"], now=kwargs["now"])


def _odds_reconcile(
    session: Session,
    source: ScoreboardSource,
    *,
    now: datetime | None = None,
) -> OddsResult:
    """The odds job's reconcile: delegate to the odds service's poll entry.

    Mirrors ``_scores_reconcile`` (which delegates to ``refresh_games``): this
    delegates to :func:`app.services.odds.reconcile_odds_games`, which selects all
    games, computes the odds-active weeks (not-yet-frozen), fetches each needy
    week once (reusing the shared per-week fetch seam), and reconciles the line.
    Returns its own :class:`~app.services.odds.OddsResult`.
    """
    return reconcile_odds_games(session, source, now=now)


# The odds cadence has ONE home: app.celery_app.REFRESH_ODDS_INTERVAL_SECONDS;
# the celery beat is built FROM this registry there, so the 300.0s below is the
# same value that beat entry emits (a test asserts the two stay equal, mirroring
# the scores-cadence single-home invariant). It is deliberately SLOWER than the
# scores 60.0s: lines crawl on minutes-to-hours and provider quotas are finite.
ODDS_JOB = PollingJob(
    name="odds",
    beat_name="refresh-odds-poller",
    schedule_seconds=300.0,
    task_name="app.tasks.refresh_odds",
    needy=_odds_needy,
    reconcile=_odds_reconcile,
)

# The ordered registry of scheduled polling jobs. The scores job stays FIRST and
# unchanged; the odds job is the sibling 2nd entry (its own cadence/needy/stop).
POLLING_JOBS: tuple[PollingJob, ...] = (SCORES_JOB, ODDS_JOB)


__all__ = [
    "PollingJob",
    "POLLING_JOBS",
    "SCORES_JOB",
    "ODDS_JOB",
    "fetch_needy_weeks",
    "group_games_by_week",
]
