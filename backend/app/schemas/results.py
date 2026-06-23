"""Pydantic response schemas for the results / season-scoreboard read API.

Mirrors the Pydantic v2 ``BaseModel`` + ``ConfigDict(extra="forbid")`` style of
:mod:`app.schemas.picks`. Each schema is built explicitly from the
:mod:`app.services.standings` service objects via a ``from_*`` classmethod
(mirroring :meth:`app.schemas.picks.PickRead.from_orm_pick`) rather than coupling
to ORM rows.

Privacy posture: like ``PickRead``, these expose ``display_name`` only — never
``user_id``. The scoreboard is shared-among-members but still does not leak the
internal user surrogate id.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from app.demo.oracle import Standings
from app.models import PickType
from app.services.standings import UserWeekResult as UserWeekResultService
from app.services.standings import WeekResultPick


class WeekResultPickRead(BaseModel):
    """One graded pick within a user's week."""

    model_config = ConfigDict(extra="forbid")

    game_id: int
    pick_type: PickType
    is_mortal_lock: bool
    outcome: str  # the GradeOutcome string value
    points: int

    @classmethod
    def from_service(cls, pick: WeekResultPick) -> "WeekResultPickRead":
        """Build from a service :class:`~app.services.standings.WeekResultPick`."""
        return cls(
            game_id=pick.game_id,
            pick_type=pick.pick_type,
            is_mortal_lock=pick.is_mortal_lock,
            outcome=pick.outcome,
            points=pick.points,
        )


class UserWeekResult(BaseModel):
    """One user's graded picks + weekly score for a ``{season, week}``.

    ``user_id`` is deliberately absent (display_name only).
    """

    model_config = ConfigDict(extra="forbid")

    display_name: str
    weekly_score: int
    picks: list[WeekResultPickRead]

    @classmethod
    def from_service(cls, result: UserWeekResultService) -> "UserWeekResult":
        """Build from a service :class:`~app.services.standings.UserWeekResult`."""
        return cls(
            display_name=result.display_name,
            weekly_score=result.weekly_score,
            picks=[WeekResultPickRead.from_service(p) for p in result.picks],
        )


class WeekResultsResponse(BaseModel):
    """Per-week graded results for a ``{season, week}`` across all users."""

    model_config = ConfigDict(extra="forbid")

    season: int
    week: int
    results: list[UserWeekResult]

    @classmethod
    def from_results(
        cls, *, season: int, week: int, results: list[UserWeekResultService]
    ) -> "WeekResultsResponse":
        """Shape the service per-user results into the HTTP response."""
        return cls(
            season=season,
            week=week,
            results=[UserWeekResult.from_service(r) for r in results],
        )


class SeasonStandingRow(BaseModel):
    """One user's cumulative season standing row.

    ``weekly_scores`` maps week number -> that week's integer score.
    """

    model_config = ConfigDict(extra="forbid")

    display_name: str
    season_total: int
    weekly_scores: dict[int, int]


class SeasonStandingsResponse(BaseModel):
    """Cumulative season standings over all users, pre-ordered by the service."""

    model_config = ConfigDict(extra="forbid")

    season: int
    standings: list[SeasonStandingRow]

    @classmethod
    def from_standings(
        cls, *, season: int, standings: Standings
    ) -> "SeasonStandingsResponse":
        """Shape the service :class:`~app.demo.oracle.Standings` into the response.

        Preserves the service's ``(-season_total, display_name)`` ordering (the
        rows are emitted in the order the service produced them).
        """
        return cls(
            season=season,
            standings=[
                SeasonStandingRow(
                    display_name=row.display_name,
                    season_total=row.season_total,
                    weekly_scores=dict(row.weekly_scores),
                )
                for row in standings.results
            ],
        )
