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
from app.schemas.types import DiscordId
from app.services.standings import UserIdentity
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
    # MISC free-text prediction; only ever present on a revealed entry (the
    # service privacy gate hides an other-user pick on a not-yet-locked game).
    misc_text: str | None = None

    @classmethod
    def from_service(cls, pick: WeekResultPick) -> "WeekResultPickRead":
        """Build from a service :class:`~app.services.standings.WeekResultPick`."""
        return cls(
            game_id=pick.game_id,
            pick_type=pick.pick_type,
            is_mortal_lock=pick.is_mortal_lock,
            outcome=pick.outcome,
            points=pick.points,
            misc_text=pick.misc_text,
        )


class UserWeekResult(BaseModel):
    """One user's graded picks + weekly score for a ``{season, week}``.

    ``user_id`` is deliberately absent (display_name only).
    """

    model_config = ConfigDict(extra="forbid")

    display_name: str
    weekly_score: int
    picks: list[WeekResultPickRead]
    # Avatar identity (mirrors the service UserWeekResult). Both None when the
    # user has no Discord avatar; the frontend builds the CDN URL from these and
    # falls back to initials otherwise. user_id stays deliberately absent.
    discord_id: DiscordId = None
    discord_avatar_hash: str | None = None

    @classmethod
    def from_service(cls, result: UserWeekResultService) -> "UserWeekResult":
        """Build from a service :class:`~app.services.standings.UserWeekResult`."""
        return cls(
            display_name=result.display_name,
            weekly_score=result.weekly_score,
            picks=[WeekResultPickRead.from_service(p) for p in result.picks],
            discord_id=result.discord_id,
            discord_avatar_hash=result.discord_avatar_hash,
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
    # Avatar identity, joined by the unique display_name from the standings
    # service's identity map. Both None when the user has no Discord avatar; the
    # frontend builds the CDN URL from these and falls back to initials.
    discord_id: DiscordId = None
    discord_avatar_hash: str | None = None


class SeasonStandingsResponse(BaseModel):
    """Cumulative season standings over all users, pre-ordered by the service."""

    model_config = ConfigDict(extra="forbid")

    season: int
    standings: list[SeasonStandingRow]
    # True ONLY when the season has games and EVERY game is FINAL — the
    # season-end state that the frontend uses to award 1st/2nd/3rd medals. False
    # for an in-progress season and for a season with zero games.
    season_complete: bool

    @classmethod
    def from_standings(
        cls,
        *,
        season: int,
        standings: Standings,
        season_complete: bool,
        identities: dict[str, UserIdentity] | None = None,
    ) -> "SeasonStandingsResponse":
        """Shape the service :class:`~app.demo.oracle.Standings` into the response.

        Preserves the service's ``(-season_total, display_name)`` ordering (the
        rows are emitted in the order the service produced them). ``season_complete``
        is the route-computed season-end flag (see
        :func:`app.services.standings.season_is_complete`).

        ``identities`` maps each row's unique ``display_name`` to its
        :class:`~app.services.standings.UserIdentity` (``discord_id`` +
        ``discord_avatar_hash``). A missing/absent entry yields ``None`` for both
        avatar fields — never raises (defensive: a row without an identity simply
        renders initials downstream).
        """
        identities = identities or {}
        return cls(
            season=season,
            season_complete=season_complete,
            standings=[
                SeasonStandingRow(
                    display_name=row.display_name,
                    season_total=row.season_total,
                    weekly_scores=dict(row.weekly_scores),
                    discord_id=(
                        identities[row.display_name].discord_id
                        if row.display_name in identities
                        else None
                    ),
                    discord_avatar_hash=(
                        identities[row.display_name].discord_avatar_hash
                        if row.display_name in identities
                        else None
                    ),
                )
                for row in standings.results
            ],
        )
