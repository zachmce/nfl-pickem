"""Authenticated results / season-scoreboard READ endpoints.

A thin GET-only router mirroring :mod:`app.api.picks`: it
authenticates the caller, calls :mod:`app.services.standings`, and shapes the
response. No scoring or ordering logic lives here — the service produces the
``score_week`` / ``grade_pick`` numbers and the ``(-season_total, display_name)``
ordering; the router only translates HTTP <-> service.

Security — deliberate shared-read posture
------------------------------------------

Both endpoints require authentication (``get_current_user`` -> 401 envelope when
unauthenticated) but are intentionally **NOT user-scoped**: any authenticated
member reads ALL users' graded results and standings. This is a reviewed,
deliberate choice — a season scoreboard is inherently shared among members — and
is therefore NOT an IDOR oversight. Contrast :mod:`app.api.picks`, which is
strictly self-scoped (a user can only read/write their own picks). The read side
still does not leak the internal ``user_id`` surrogate: responses carry
``display_name`` only.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlmodel import Session

from app.api.deps import get_current_user
from app.db import get_session
from app.models import User
from app.schemas.results import SeasonStandingsResponse, WeekResultsResponse
from app.services.standings import (
    season_is_complete,
    season_standings,
    week_results,
)

router = APIRouter(prefix="/api/results", tags=["results"])


@router.get("/week", response_model=WeekResultsResponse)
def read_week(
    season: int,
    week: int,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> WeekResultsResponse:
    """Per-week graded results for ``{season, week}`` across ALL users.

    Shared read: authenticated but NOT user-scoped (see module docstring) — any
    member sees every user's graded picks and weekly score. An empty/unknown
    week yields an empty ``results`` list (a pure read, never a 404).
    """
    results = week_results(session, season=season, week=week, caller_user_id=user.id)
    return WeekResultsResponse.from_results(season=season, week=week, results=results)


@router.get("/standings", response_model=SeasonStandingsResponse)
def read_standings(
    season: int,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> SeasonStandingsResponse:
    """Cumulative season standings over ALL users, ordered by the service.

    Shared read: authenticated but NOT user-scoped (see module docstring).
    Ordering is ``(-season_total, display_name)`` as produced by
    :func:`app.services.standings.season_standings`.
    """
    standings, identities = season_standings(session, season=season)
    complete = season_is_complete(session, season=season)
    return SeasonStandingsResponse.from_standings(
        season=season,
        standings=standings,
        season_complete=complete,
        identities=identities,
    )
