"""Source-agnostic odds reconciliation + the computed odds-freeze predicate.

This is the odds-poll core — a sibling to :mod:`app.services.refresh` (the scores
poller). Where ``refresh`` reconciles status/scores/kickoff and stops at FINAL,
this reconciles a game's betting line and stops at *freeze*. The design mirrors
``refresh.py`` exactly:

* The business logic lives here and operates on a **passed-in** ``Session``. The
  Celery task in :mod:`app.tasks` is a thin wrapper that opens the session and
  resolves the production default source. This module NEVER constructs an ESPN
  adapter and NEVER imports the network layer or the settings module — it is
  source-agnostic and unit-testable offline against a synthetic source. It also
  contains NO demo-mode gate; gated source resolution stays in the task wrapper
  (the same source-agnostic invariant the scheduler seam asserts).
* ``now`` is injected (defaulting to the real UTC clock) and used only for the
  freeze decision (``is_odds_frozen``) and the captured-at stamp.

Freeze is a **computed predicate**, not a flag a beat flips (mirroring how the
pick window is computed from kickoffs, see :mod:`app.services.pick_window`)::

    freeze_at(week)        = min( noon-ET-Wednesday-of-the-week, pick_lock )
    is_odds_frozen(week)   = week.lines_frozen  OR  now >= freeze_at(week)

* ``pick_lock`` is the same boundary the pick window uses:
  ``compute_window(...).close_at`` (the week's earliest kickoff). Reusing it means
  freeze can never land *after* picks lock for valid schedules.
* ``noon-ET-Wednesday`` is computed via :class:`zoneinfo.ZoneInfo` (handles
  EST/EDT) as 12:00 America/New_York on the Wednesday on or before the week's
  earliest kickoff, converted to UTC.
* A fail-loud guard raises a labeled :class:`ValueError` if the computed
  ``freeze_at`` would ever exceed ``pick_lock``. ``min()`` makes this dormant for
  valid schedules; the guard catches a future reschedule/early-game quirk instead
  of silently freezing a line after picks already locked.

Normalization (source -> stored) mirrors :mod:`app.seeds.fixture_2025` EXACTLY so
the slate read path keeps seeing a consistent representation:

* ``spread`` is stored as a POSITIVE ``Decimal`` magnitude
  (``Decimal(str(abs(source_spread)))`` — fixture line 221), with direction
  carried by ``favorite_team_id`` / ``underdog_team_id``.
* ``total`` is ``Decimal(str(source_total))``.
* ``favorite_team_id`` / ``underdog_team_id`` are resolved from the source's
  STRING espn team ids to int ``team.id`` FKs via a ``{espn_team_id: team.id}``
  map (mirroring ``fixture_2025._build_team_map`` / ``_resolve_team``). An
  unresolvable id is SKIPPED in the poll path (not crashed) so a bad source id
  never aborts a tick.
* A ``None`` odds block (or ``None`` field) NEVER nulls a present value.

Reconciliation is idempotent: only changed fields are written (Decimals compared
by value), so an unchanged re-run produces zero dirty rows.

The reconciler enforces freeze at the WRITE path — :func:`reconcile_odds` refuses
to write a frozen game even when asked directly (belt-and-suspenders beyond the
needy filter).

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use the venv interpreter ``.venv/bin/python``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

from sqlmodel import Session, select

from app.models import Game, Week
from app.scoreboard.types import ScoreboardGame, ScoreboardOdds
from app.services.pick_window import compute_window
from app.services.refresh import group_games_by_week

# The line-freeze timezone + default freeze instant: noon (12:00) ET Wednesday.
_FREEZE_TZ = ZoneInfo("America/New_York")
_FREEZE_HOUR = 12  # noon ET
_WEDNESDAY = 2  # Monday=0 ... Wednesday=2 (datetime.weekday())

WeekKey = tuple[int, int]
GroupedRows = dict[WeekKey, list[Game]]


@dataclass(frozen=True)
class OddsResult:
    """Immutable summary of one odds-poll run (value object).

    Mirrors :class:`app.services.refresh.RefreshResult`'s frozen-dataclass style:

    * ``weeks_polled`` — count of needy weeks whose fetch succeeded and were
      reconciled.
    * ``games_updated`` — count of Game rows whose line changed.
    * ``frozen_weeks`` — the ``(season, week)`` pairs that were frozen at ``now``
      (skipped/refused for writes).
    * ``failed_weeks`` — the ``(season, week)`` pairs whose fetch raised.
    """

    weeks_polled: int = 0
    games_updated: int = 0
    frozen_weeks: tuple[WeekKey, ...] = field(default_factory=tuple)
    failed_weeks: tuple[WeekKey, ...] = field(default_factory=tuple)


def _as_aware(dt: datetime | None) -> datetime | None:
    """Re-attach UTC to a naive datetime read back from the store.

    ``DateTime(timezone=True)`` round-trips NAIVE on SQLite (Postgres preserves
    tz). The freeze math compares against tz-aware values, so this normalizes a
    naive value to UTC for the comparison ONLY — mirrors ``refresh._as_aware``.
    """
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _normalized_games(games: list[Game]) -> list[Game]:
    """Shallow copies with tz-aware kickoffs so ``compute_window`` is safe.

    Avoids mutating (and thus dirtying) the live rows — mirrors
    ``refresh._normalized_games`` but only carries the fields the window math and
    grouping need.
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


def _noon_et_wednesday_on_or_before(kickoff: datetime) -> datetime:
    """Return noon ET on the Wednesday on or before ``kickoff`` (as UTC).

    Computed in America/New_York (via :class:`zoneinfo.ZoneInfo`, which handles
    EST/EDT) then converted to UTC. ``kickoff`` must be tz-aware.
    """
    local = kickoff.astimezone(_FREEZE_TZ)
    # Days since the most recent Wednesday (0 if local day is Wednesday).
    days_since_wed = (local.weekday() - _WEDNESDAY) % 7
    wed_date = (local - timedelta(days=days_since_wed)).date()
    noon_local = datetime.combine(wed_date, time(hour=_FREEZE_HOUR), tzinfo=_FREEZE_TZ)
    return noon_local.astimezone(timezone.utc)


def _guard_freeze_at_le_pick_lock(*, freeze: datetime, pick_lock: datetime) -> datetime:
    """Fail-loud guard: ``freeze`` must be <= ``pick_lock``.

    ``min(..., pick_lock)`` makes this dormant for valid schedules; the guard
    catches a future reschedule/early-game quirk where the math would otherwise
    freeze a line AFTER picks already locked. Raises a deliberate, labeled
    :class:`ValueError` (not a bare ``assert`` that ``-O`` could strip).
    """
    if freeze > pick_lock:
        raise ValueError(
            "computed odds freeze_at "
            f"({freeze.isoformat()}) would be AFTER pick_lock "
            f"({pick_lock.isoformat()}); refusing to freeze a line after picks "
            "lock"
        )
    return freeze


def freeze_at(
    week_games: list[Game],
    prev_week_games: list[Game] | None = None,
) -> datetime:
    """Compute the week's odds-freeze instant: ``min(noon-ET-Wed, pick_lock)``.

    ``pick_lock`` is :func:`app.services.pick_window.compute_window`'s ``close_at``
    (the week's earliest kickoff) — the SAME boundary the pick window uses, so
    freeze can never land after picks lock for a valid schedule. ``noon-ET-Wed``
    is 12:00 America/New_York on the Wednesday on or before that earliest kickoff.

    Inputs are tz-normalized (store-read naive kickoffs are re-attached to UTC for
    the math, never persisted). Raises :class:`ValueError` via ``compute_window``
    when the week has no kickoff to reason about, and via the fail-loud guard if
    the result would ever exceed ``pick_lock``.
    """
    window = compute_window(
        _normalized_games(week_games),
        _normalized_games(prev_week_games) if prev_week_games else None,
    )
    pick_lock = window.close_at
    noon_wed = _noon_et_wednesday_on_or_before(pick_lock)
    return _guard_freeze_at_le_pick_lock(freeze=min(noon_wed, pick_lock), pick_lock=pick_lock)


def is_odds_frozen(
    week_row: Week,
    week_games: list[Game],
    prev_week_games: list[Game] | None = None,
    *,
    now: datetime | None = None,
) -> bool:
    """Whether the week's odds are frozen at ``now``.

    ``week_row.lines_frozen`` is the explicit admin/early-freeze override; the
    computed predicate (``now >= freeze_at(...)``) is otherwise the source of
    truth (per the locked design note — prefer the computed predicate over
    flipping a stored ``odds_frozen`` flag). ``now`` defaults to the real UTC
    clock.
    """
    if week_row.lines_frozen:
        return True
    if now is None:
        now = datetime.now(timezone.utc)
    return now >= freeze_at(week_games, prev_week_games)


def _resolve_team_id(team_map: dict[int, int], espn_team_id_str: str | None) -> int | None:
    """Resolve a source (string) espn team id to an int ``team.id`` FK.

    Mirrors ``fixture_2025._resolve_team`` but is POLL-SAFE: an unresolvable id
    returns ``None`` (skip the FK write) rather than raising, so a bad source id
    never aborts a tick. A ``None``/non-numeric input also returns ``None``.
    """
    if espn_team_id_str is None:
        return None
    try:
        espn_team_id = int(espn_team_id_str)
    except TypeError, ValueError:
        return None
    return team_map.get(espn_team_id)


def reconcile_odds(
    row: Game,
    src_odds: ScoreboardOdds | None,
    team_map: dict[int, int],
    *,
    frozen: bool,
    now: datetime,
) -> bool:
    """Apply ``src_odds`` onto ``row``, writing only changed fields. Freeze-safe.

    WRITE-PATH enforcement: if ``frozen`` is ``True`` this refuses to write
    anything and returns ``False`` (belt-and-suspenders beyond the needy filter).

    Normalization mirrors :mod:`app.seeds.fixture_2025` EXACTLY:

    * ``spread`` -> ``Decimal(str(abs(src_odds.spread)))`` (positive magnitude),
    * ``total`` -> ``Decimal(str(src_odds.total))``,
    * ``favorite_team_id`` / ``underdog_team_id`` -> int ``team.id`` resolved from
      the source's STRING espn ids (unresolvable id is skipped, not crashed),
    * ``odds_provider`` -> ``src_odds.provider``,
    * ``odds_captured_at`` -> ``now`` (stamped only when a field actually changed).

    A ``None`` ``src_odds`` (or ``None`` field) NEVER nulls a present value.
    Decimals are compared by value, so an unchanged re-apply is a no-op. Returns
    ``True`` iff any field was written.
    """
    if frozen or src_odds is None:
        return False

    changed = False

    if src_odds.spread is not None:
        new_spread = Decimal(str(abs(src_odds.spread)))
        if row.spread is None or row.spread != new_spread:
            row.spread = new_spread
            changed = True

    if src_odds.total is not None:
        new_total = Decimal(str(src_odds.total))
        if row.total is None or row.total != new_total:
            row.total = new_total
            changed = True

    fav_id = _resolve_team_id(team_map, src_odds.favorite_team_id)
    if fav_id is not None and row.favorite_team_id != fav_id:
        row.favorite_team_id = fav_id
        changed = True

    dog_id = _resolve_team_id(team_map, src_odds.underdog_team_id)
    if dog_id is not None and row.underdog_team_id != dog_id:
        row.underdog_team_id = dog_id
        changed = True

    if src_odds.provider is not None and row.odds_provider != src_odds.provider:
        row.odds_provider = src_odds.provider
        changed = True

    if changed:
        row.odds_captured_at = now

    return changed


def _odds_by_event_id(games: list[ScoreboardGame]) -> dict[int, ScoreboardOdds | None]:
    """Index the source's games' odds blocks by integer ``espn_event_id``.

    Mirrors ``refresh._scores_by_event_id`` but yields the odds block (which may
    be ``None``) so the reconciler can decide per row.
    """
    indexed: dict[int, ScoreboardOdds | None] = {}
    for sg in games:
        if sg.espn_event_id is None:
            continue
        indexed[int(sg.espn_event_id)] = sg.odds
    return indexed


def reconcile_week_odds(
    session: Session,
    fetched_games: list[ScoreboardGame],
    week_games: list[Game],
    *,
    week_row: Week,
    now: datetime,
    team_map: dict[int, int],
    prev_week_games: list[Game] | None = None,
) -> OddsResult:
    """Reconcile one week's odds: compute freeze once, write per matched row.

    Indexes the fetched games by int ``espn_event_id`` (reusing the
    ``_scores_by_event_id`` pattern), computes ``frozen = is_odds_frozen(...)``
    once for the week, and calls :func:`reconcile_odds` per matched row,
    ``session.add``-ing changed rows. Does NOT commit — the caller commits.
    """
    frozen = is_odds_frozen(week_row, week_games, prev_week_games, now=now)
    key = (week_row.season, week_row.week)
    if frozen:
        # The week is frozen: nothing is written (the write-path refusal in
        # reconcile_odds also enforces this per-row).
        return OddsResult(weeks_polled=1, frozen_weeks=(key,))

    src_by_id = _odds_by_event_id(fetched_games)
    games_updated = 0
    for row in week_games:
        src_odds = src_by_id.get(row.espn_event_id)
        if src_odds is None and row.espn_event_id not in src_by_id:
            # No matching fetched game for this row -> skip.
            continue
        if reconcile_odds(row, src_odds, team_map, frozen=False, now=now):
            session.add(row)
            games_updated += 1

    return OddsResult(weeks_polled=1, games_updated=games_updated)


def odds_needy_weeks(
    by_week: GroupedRows,
    week_rows_by_key: dict[int, Week] | dict[WeekKey, Week],
    *,
    now: datetime,
) -> list[WeekKey]:
    """The odds-active predicate: weeks NOT yet frozen (and that have games).

    A week is odds-active iff it has games AND is not yet frozen
    (:func:`is_odds_frozen` False). The stop condition is FREEZE — a frozen week
    is excluded (the efficiency skip; the write path also refuses it). Grouping is
    routed through ``services.refresh.group_games_by_week`` by the callers so the
    needy view never drifts from the scores view.

    ``week_rows_by_key`` may be keyed by week number (single-season convenience,
    as the tests pass) or by ``(season, week)`` — both are accepted.
    """
    needy: list[WeekKey] = []
    for (season, week), rows in by_week.items():
        if not rows:
            continue
        week_row = week_rows_by_key.get((season, week))
        if week_row is None:
            week_row = week_rows_by_key.get(week)  # week-number keyed fallback
        if week_row is None:
            # No persisted Week row to read lines_frozen from -> not pollable.
            continue
        if is_odds_frozen(week_row, rows, now=now):
            continue
        needy.append((season, week))
    return sorted(needy)


def reconcile_odds_games(
    session: Session,
    source,
    *,
    now: datetime | None = None,
    team_map: dict[int, int] | None = None,
) -> OddsResult:
    """Reconcile odds for every odds-active week. Mirrors ``refresh_games``.

    Selects all Game rows, groups them by week, computes the odds-needy weeks
    (not-yet-frozen weeks with games), fetches each needy week (one
    ``source.fetch_week`` per week — the same per-week fetch seam the scores job
    shares), and reconciles the line onto the matched rows. Adds/updates rows but
    does NOT commit — the caller commits.

    :param source: any object exposing ``fetch_week(season, week)`` returning the
        port's :class:`~app.scoreboard.types.ScoreboardGame` list (the injected
        :class:`~app.scoreboard.port.ScoreboardSource`).
    :param now: tz-aware UTC instant for the freeze decision + captured-at stamp;
        defaults to ``datetime.now(timezone.utc)``.
    :param team_map: optional ``{espn_team_id: team.id}`` map; built from the
        ``Team`` table when omitted.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    all_games = list(session.exec(select(Game)).all())
    if not all_games:
        return OddsResult()

    if team_map is None:
        from app.models import Team

        team_map = {t.espn_team_id: t.id for t in session.exec(select(Team)).all()}

    by_week = group_games_by_week(all_games)

    seasons = {season for season, _ in by_week}
    week_rows_by_key: dict[WeekKey, Week] = {}
    for season in seasons:
        for wr in session.exec(select(Week).where(Week.season == season)).all():
            week_rows_by_key[(wr.season, wr.week)] = wr

    needy = odds_needy_weeks(by_week, week_rows_by_key, now=now)

    weeks_polled = 0
    games_updated = 0
    frozen_weeks: list[WeekKey] = []
    failed_weeks: list[WeekKey] = []

    from app.scoreboard.port import ScoreboardFetchError

    for season, week in needy:
        try:
            fetched = source.fetch_week(season, week)
        except ScoreboardFetchError:
            failed_weeks.append((season, week))
            continue

        weeks_polled += 1
        week_row = week_rows_by_key[(season, week)]
        result = reconcile_week_odds(
            session,
            fetched,
            by_week[(season, week)],
            week_row=week_row,
            now=now,
            team_map=team_map,
        )
        games_updated += result.games_updated
        frozen_weeks.extend(result.frozen_weeks)

    return OddsResult(
        weeks_polled=weeks_polled,
        games_updated=games_updated,
        frozen_weeks=tuple(frozen_weeks),
        failed_weeks=tuple(failed_weeks),
    )


__all__ = [
    "OddsResult",
    "freeze_at",
    "is_odds_frozen",
    "reconcile_odds",
    "reconcile_week_odds",
    "odds_needy_weeks",
    "reconcile_odds_games",
]
