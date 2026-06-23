"""User-agnostic, DB-sourced season standings + per-week graded results.

This is the read-side service behind the season-scoreboard HTTP view
(build-order #6): the human-facing counterpart to the pick-submission API. It
computes, from PERSISTED ``Pick`` rows, (1) cumulative season standings over
**all** users and (2) per-week graded results for a ``{season, week}``.

Why a new service instead of reusing :func:`app.demo.driver.compute_db_standings`
-----------------------------------------------------------------------------

``driver.compute_db_standings`` is deliberately **bot-scoped** — it loads only
the walkthrough bots (``_bot_users``) — because it is the *actual* side of the
demo capstone's ``actual == expected`` proof against
:func:`app.demo.oracle.compute_standings`. The real scoreboard, by contrast,
must show ALL users. Rather than mutate the demo path (which would risk the
integration proof), this module is an **additive extraction**: it computes
all-users standings while reusing the exact same building blocks —
:func:`app.services.scoring.score_week` / :func:`~app.services.scoring.grade_pick`
for the math and :class:`app.demo.oracle.Standings` /
:class:`~app.demo.oracle.BotSeasonResult` for the shape and the
``(-season_total, display_name)`` ordering. The import is one-directional (this
service imports the demo module; the demo never imports this service), so there
is no cycle and the demo proof is structurally unaffected.

Purity / side effects
----------------------

This is a **read** service: it reads the passed-in session and writes nothing —
no ``add``, no ``commit``. As in the sibling services it re-attaches UTC to the
naive ``kickoff_at`` that ``DateTime(timezone=True)`` round-trips on SQLite, but
only on in-memory ``Game`` copies that are never persisted (mirrors
``driver._as_aware`` / ``test_picks_api._aware``); production-on-Postgres is
unaffected.

> Note: on this machine there is no bare ``python`` on ``PATH``; use the venv
> interpreter ``.venv/bin/python`` for any commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlmodel import Session, select

from app.demo.oracle import (
    BotSeasonResult,
    Standings,
    games_by_pk_index,
)
from app.models import Game, Pick, PickType, User, Week
from app.services.scoring import GradeOutcome, grade_pick, score_week


@dataclass(frozen=True)
class WeekResultPick:
    """One graded pick within a user's week, ready for HTTP shaping.

    ``outcome`` is the :class:`~app.services.scoring.GradeOutcome` string value
    and ``points`` is the points it earned (per the scoring table). These come
    straight from :func:`app.services.scoring.grade_pick` — never re-derived.
    """

    game_id: int
    pick_type: PickType
    is_mortal_lock: bool
    outcome: str
    points: int


@dataclass(frozen=True)
class UserWeekResult:
    """One user's graded picks + weekly score for a single ``{season, week}``.

    ``weekly_score`` equals the sum of ``picks``' points and equals what
    :func:`app.services.scoring.score_week` returns for the same picks — the
    score is taken from ``score_week`` directly, the per-pick points from
    ``grade_pick``, so they are consistent by construction. ``user_id`` is
    deliberately ABSENT (display_name only) matching the read-side privacy
    posture of ``PickRead``.
    """

    display_name: str
    weekly_score: int
    picks: tuple[WeekResultPick, ...]


def _as_aware(dt: datetime | None) -> datetime | None:
    """Re-attach UTC to a naive datetime read back from the store.

    ``DateTime(timezone=True)`` round-trips NAIVE on SQLite (Postgres preserves
    tz). Re-declared locally (mirrors ``driver._as_aware`` /
    ``pick_submission._as_aware``) rather than importing a private helper. The
    normalized copy is never persisted, leaving production-on-Postgres
    unaffected.
    """
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _season_games_by_pk(session: Session, *, season: int) -> dict[int, Game]:
    """Load the season's games once and build the PK index ``score_week`` wants.

    Normalizes each in-memory ``Game``'s ``kickoff_at`` to UTC (sqlite tz
    round-trip); the copies are never committed.
    """
    games = list(session.exec(select(Game).where(Game.season == season)).all())
    for g in games:
        g.kickoff_at = _as_aware(g.kickoff_at)  # in-memory copy only; not committed
    return games_by_pk_index(games)


def _season_week_ids(session: Session, *, season: int) -> dict[int, int]:
    """Map ``week number -> Week.id`` for the season (one query, one place)."""
    return {
        w.week: w.id
        for w in session.exec(select(Week).where(Week.season == season)).all()
        if w.id is not None
    }


def season_standings(session: Session, *, season: int) -> Standings:
    """Cumulative season standings over ALL users with picks in ``season``.

    Derives the user set from the distinct ``user_id`` of ``Pick`` rows in the
    season (same convention as ``driver.compute_db_standings`` derives its set
    from the bot users — only users who actually picked appear). For each such
    user, scores every week they picked via
    :func:`app.services.scoring.score_week` (a week with no picks contributes no
    entry, matching the oracle's "only weeks present" convention) and builds a
    :class:`~app.demo.oracle.BotSeasonResult`. Returns a
    :class:`~app.demo.oracle.Standings` ordered by ``(-season_total,
    display_name)``.

    Pure read: opens no transaction state of its own, writes nothing.
    """
    games_by_pk = _season_games_by_pk(session, season=season)
    week_id_to_number = {
        week_id: number
        for number, week_id in _season_week_ids(session, season=season).items()
    }
    season_week_ids = set(week_id_to_number)

    # Every persisted pick in the season, in one query, grouped per user/week.
    # picks_by_user[user_id][week_number] -> list[Pick]
    picks_by_user: dict[int, dict[int, list[Pick]]] = {}
    all_picks = session.exec(select(Pick).where(Pick.week_id.in_(season_week_ids))).all()
    for pick in all_picks:
        week_number = week_id_to_number.get(pick.week_id)
        if week_number is None:  # pick outside this season's weeks
            continue
        picks_by_user.setdefault(pick.user_id, {}).setdefault(week_number, []).append(
            pick
        )

    display_names = _display_names_for(session, user_ids=picks_by_user.keys())

    results: list[BotSeasonResult] = []
    for user_id, weeks in picks_by_user.items():
        weekly_scores: dict[int, int] = {
            week_number: score_week(games_by_pk, week_picks)
            for week_number, week_picks in weeks.items()
        }
        results.append(
            BotSeasonResult(
                display_name=display_names[user_id],
                weekly_scores=weekly_scores,
                season_total=sum(weekly_scores.values()),
            )
        )

    results.sort(key=lambda r: (-r.season_total, r.display_name))
    return Standings(results=tuple(results))


def week_results(
    session: Session, *, season: int, week: int
) -> list[UserWeekResult]:
    """Per-user graded results for a single ``{season, week}`` over ALL users.

    Resolves the ``Week`` row, loads that week's persisted picks grouped by
    user, grades each pick via :func:`app.services.scoring.grade_pick` (game
    resolved from the season's normalized PK index), and reports the user's
    weekly score via :func:`app.services.scoring.score_week` (so the score is
    never re-derived from the per-pick points). Users with no picks that week
    are omitted. Ordered by ``(-weekly_score, display_name)``.

    An empty season/week simply yields an empty list — a 404-style miss is NOT
    raised here (this is a pure read).
    """
    week_id = _season_week_ids(session, season=season).get(week)
    if week_id is None:
        return []

    games_by_pk = _season_games_by_pk(session, season=season)

    picks_by_user: dict[int, list[Pick]] = {}
    for pick in session.exec(select(Pick).where(Pick.week_id == week_id)).all():
        picks_by_user.setdefault(pick.user_id, []).append(pick)

    display_names = _display_names_for(session, user_ids=picks_by_user.keys())

    results: list[UserWeekResult] = []
    for user_id, picks in picks_by_user.items():
        graded: list[WeekResultPick] = []
        for pick in picks:
            decision = grade_pick(games_by_pk[pick.game_id], pick)
            graded.append(
                WeekResultPick(
                    game_id=pick.game_id,
                    pick_type=pick.pick_type,
                    is_mortal_lock=pick.is_mortal_lock,
                    outcome=GradeOutcome(decision.outcome).value,
                    points=decision.points,
                )
            )
        results.append(
            UserWeekResult(
                display_name=display_names[user_id],
                # Take the weekly score from the scorer directly (== sum of the
                # graded points above, by construction) so nothing is re-derived.
                weekly_score=score_week(games_by_pk, picks),
                picks=tuple(graded),
            )
        )

    results.sort(key=lambda r: (-r.weekly_score, r.display_name))
    return results


def _display_names_for(session: Session, *, user_ids) -> dict[int, str]:
    """Look up ``{User.id: display_name}`` for the given user ids (one query)."""
    ids = set(user_ids)
    if not ids:
        return {}
    return {
        u.id: u.display_name
        for u in session.exec(select(User).where(User.id.in_(ids))).all()
        if u.id is not None
    }
