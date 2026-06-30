"""Authenticated read-only CALENDAR endpoint (date-range games).

A thin GET-only router mirroring :mod:`app.api.slate` / :mod:`app.api.results`:
it authenticates the caller, reads every :class:`~app.models.Game` whose
``kickoff_at`` falls inside a requested ``[from, to]`` date range, resolves the
referenced teams (no N+1), and shapes a DISPLAY-ONLY response (matchup
abbreviations, raw UTC kickoff, status, home/away score). It is the data source
for the month-grid Calendar page.

Discretion (LOCKED in CONTEXT.md): the SERVER is a pure date-range filter — it
returns the RAW UTC ``kickoff_at`` and does zero timezone math. The CLIENT
buckets each game onto its US Eastern (``America/New_York``) calendar day with
``Intl.DateTimeFormat`` (no date library added). This keeps the endpoint simple
and the day-placement logic in one place on the client.

Security — deliberate shared-read posture
------------------------------------------

The endpoint requires authentication (``get_current_user`` -> 401 envelope when
unauthenticated) but is intentionally **NOT user-scoped**: the schedule is the
same for every member (the season's public games — no per-user data). This is a
reviewed, deliberate choice (same posture as :mod:`app.api.slate` /
:mod:`app.api.results`), not an IDOR oversight. The response carries no
``user_id``; the only identity surfaced is public Team reference data.

Demo correctness — no ``IS_DEMO_DATA`` branch
---------------------------------------------

This is a pure date-range read of the PERSISTED kickoffs — "real now" is
irrelevant to the query (the client picks the default month from the real
clock, but the endpoint just answers whatever ``[from, to]`` it is asked for).
In demo mode the seed rigidly time-shifts kickoffs onto real-near-now dates, so
the same range query is automatically demo-correct. This module therefore does
NOT import :mod:`app.config`, ``settings.is_demo_data``, or any :mod:`app.demo`
module.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select

from app.api.deps import get_current_user
from app.db import get_session
from app.exceptions import ValidationError
from app.models import Game, Team, User
from app.schemas.calendar import CalendarGame, CalendarResponse, CalendarTeam

router = APIRouter(prefix="/api/calendar", tags=["calendar"])


def _as_aware(dt: datetime | None) -> datetime | None:
    """Re-attach UTC to a naive datetime read back from the store.

    ``DateTime(timezone=True)`` round-trips NAIVE on SQLite (Postgres preserves
    tz). Ordering compares against tz-aware values, so this normalizes a naive
    value to UTC for the comparison ONLY — the normalized copy is never persisted
    back (mirrors :func:`app.api.slate._as_aware`).
    """
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_day(value: str, *, field: str) -> datetime:
    """Parse a ``YYYY-MM-DD`` string into a tz-aware UTC midnight instant.

    Raises the project's enveloped :class:`~app.exceptions.ValidationError` (422)
    on a malformed string so a bad query param is always a clean 4xx, never a
    500 (mirrors how the other routers surface bad input).
    """
    try:
        d = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValidationError(
            f"Invalid date for '{field}': expected YYYY-MM-DD.",
            fields={field: ["Expected a YYYY-MM-DD date."]},
        ) from exc
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


@router.get("", response_model=CalendarResponse)
def read_calendar(
    from_: str = Query(..., alias="from"),
    to: str = Query(...),
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> CalendarResponse:
    """The season's games whose kickoff falls in the ``[from, to]`` window.

    Shared read: authenticated but NOT user-scoped (see module docstring).
    ``from``/``to`` are ``YYYY-MM-DD`` (the ``from`` query param is aliased
    because ``from`` is a Python keyword). The upper bound is INCLUSIVE of the
    whole ``to`` day — the filter uses an exclusive ``end = to + 1 day`` so a
    game at any time on the ``to`` date is returned. Games with a NULL kickoff
    are naturally excluded by the range comparison. An empty window yields an
    empty ``games`` list (a pure read, never a 404). Malformed dates -> 422.
    """
    start = _parse_day(from_, field="from")
    to_start = _parse_day(to, field="to")
    end = to_start + timedelta(days=1)  # exclusive: covers the whole `to` day

    games = list(
        session.exec(select(Game).where(Game.kickoff_at >= start, Game.kickoff_at < end)).all()
    )

    # Resolve referenced teams in one query (no N+1), keyed by id.
    team_ids = {g.home_team_id for g in games} | {g.away_team_id for g in games}
    teams_by_id: dict[int, Team] = {}
    if team_ids:
        for t in session.exec(select(Team).where(Team.id.in_(team_ids))).all():
            assert t.id is not None
            teams_by_id[t.id] = t

    def _team(team_id: int) -> CalendarTeam:
        return CalendarTeam(abbreviation=teams_by_id[team_id].abbreviation)

    # Stable order: kickoff then game_id (null kickoffs sort last — though the
    # range filter already excludes them).
    _MAX_KO = datetime.max.replace(tzinfo=timezone.utc)
    games.sort(key=lambda g: (_as_aware(g.kickoff_at) or _MAX_KO, g.id or 0))

    calendar_games: list[CalendarGame] = []
    for g in games:
        assert g.id is not None
        calendar_games.append(
            CalendarGame(
                game_id=g.id,
                kickoff_at=g.kickoff_at,
                home_team=_team(g.home_team_id),
                away_team=_team(g.away_team_id),
                # Pass the GameStatus enum member straight through — never compare
                # against a raw status string (that errors on the PG enum).
                status=g.status,
                home_score=g.home_score,
                away_score=g.away_score,
            )
        )

    return CalendarResponse.from_games(from_date=from_, to_date=to, games=calendar_games)
