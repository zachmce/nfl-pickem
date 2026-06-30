"""Pydantic response schemas for the pickable-slate read API.

Mirrors the Pydantic v2 ``BaseModel`` + ``ConfigDict(extra="forbid")`` style of
:mod:`app.schemas.results` and :mod:`app.schemas.current_week`. ``SlateResponse``
is built explicitly from already-shaped row objects via a ``from_*`` classmethod
(mirroring :meth:`app.schemas.results.WeekResultsResponse.from_results`) rather
than coupling to ORM rows — the router shapes each game (resolving team identity,
the per-game ``locked`` bool and per-``PickType`` eligibility) and hands the
schema the finished values.

Privacy posture: the slate is OPTIONS only (no picks, no per-user data). It
surfaces public game/line/team reference fields and carries no ``user_id`` — the
shared-read posture mirrors :mod:`app.schemas.results`.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from app.models import PickType


class SlateTeam(BaseModel):
    """Public team reference identity for one side of a matchup."""

    model_config = ConfigDict(extra="forbid")

    team_id: int
    abbreviation: str
    display_name: str


class SlateGame(BaseModel):
    """One game's pickable slate entry: identity, line, lock + eligibility.

    ``eligibility`` maps every :class:`~app.models.PickType` to whether that pick
    type is a legal *choice* on this game (computed by the shared
    :func:`app.services.pick_validation.is_pick_type_eligible`). ``locked`` is the
    per-game lock derived from real now vs the persisted kickoff. The line fields
    (``spread`` / ``total`` / ``favorite_team_id`` / ``underdog_team_id``) and
    ``kickoff_at`` preserve their persisted nulls.
    """

    model_config = ConfigDict(extra="forbid")

    game_id: int
    kickoff_at: datetime | None
    home_team: SlateTeam
    away_team: SlateTeam
    spread: Decimal | None
    total: Decimal | None
    favorite_team_id: int | None
    underdog_team_id: int | None
    locked: bool
    eligibility: dict[PickType, bool]


class SlateResponse(BaseModel):
    """The pickable slate for a ``{season, week}``: one entry per game.

    ``user_id`` is deliberately absent — the slate is OPTIONS only, shared among
    all members (see :mod:`app.api.slate`).

    ``odds_frozen`` is the COMPUTED week-level freeze predicate result — the
    output of :func:`app.services.odds.is_odds_frozen` against the real clock. It
    is distinct from the ``Week.lines_frozen`` admin-override INPUT column (one of
    the predicate's inputs) and from the per-game ``Game.odds_frozen`` flag; it is
    week-level, not per game, so it lives on the response rather than
    :class:`SlateGame`.
    """

    model_config = ConfigDict(extra="forbid")

    season: int
    week: int
    games: list[SlateGame]
    odds_frozen: bool

    @classmethod
    def from_games(
        cls, *, season: int, week: int, games: list[SlateGame], odds_frozen: bool
    ) -> "SlateResponse":
        """Shape the router's already-built per-game rows into the response.

        The router resolves team identity, the per-game ``locked`` bool and the
        per-``PickType`` eligibility, computes the week-level ``odds_frozen``
        predicate, then passes the finished :class:`SlateGame` rows here — keeping
        the schema decoupled from ORM rows (mirrors
        :meth:`app.schemas.results.WeekResultsResponse.from_results`).
        """
        return cls(season=season, week=week, games=list(games), odds_frozen=odds_frozen)
