"""Pydantic response schemas for the read-only calendar (month-grid) API.

Mirrors the Pydantic v2 ``BaseModel`` + ``ConfigDict(extra="forbid")`` style of
:mod:`app.schemas.slate` / :mod:`app.schemas.results`. ``CalendarResponse`` is
built explicitly from already-shaped row objects via a ``from_*`` classmethod
(mirroring :meth:`app.schemas.slate.SlateResponse.from_games`) rather than
coupling to ORM rows — the router shapes each game (resolving team identity) and
hands the schema the finished values.

Privacy posture: this is a DISPLAY-ONLY public schedule view. It carries no
picks and no per-user data — just the season's games (matchup abbreviations,
raw UTC kickoff, status, and home/away score) over a requested date range. No
``user_id`` is surfaced; the shared-read posture mirrors :mod:`app.schemas.slate`
/ :mod:`app.schemas.results` (authenticated, but the same view for every member).

The CLIENT buckets each game onto its US Eastern (``America/New_York``) calendar
day; the server stays a pure date-range filter and returns the raw UTC
``kickoff_at`` (no timezone math here).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models import GameStatus


class CalendarTeam(BaseModel):
    """Public team reference identity (abbreviation only) for one side."""

    model_config = ConfigDict(extra="forbid")

    abbreviation: str


class CalendarGame(BaseModel):
    """One game's display-only calendar entry.

    ``kickoff_at`` is the RAW UTC instant (persisted value, naive on SQLite) —
    the client buckets it onto its US Eastern calendar day with
    ``Intl.DateTimeFormat`` and renders the ET kickoff time. ``home_score`` /
    ``away_score`` carry the persisted values unchanged; they are only meaningful
    when ``status`` is FINAL (the client renders the score only when FINAL), but
    the schema ships whatever is persisted regardless.
    """

    model_config = ConfigDict(extra="forbid")

    game_id: int
    kickoff_at: datetime | None
    home_team: CalendarTeam
    away_team: CalendarTeam
    status: GameStatus
    home_score: int | None
    away_score: int | None


class CalendarResponse(BaseModel):
    """The season's games whose kickoff falls in ``[from_date, to_date]``.

    ``user_id`` is deliberately absent — this is a display-only public schedule
    view, shared among all members (see :mod:`app.api.calendar`). ``from_date`` /
    ``to_date`` echo the requested window (``YYYY-MM-DD``) so the client can
    correlate the response with the grid it asked for.
    """

    model_config = ConfigDict(extra="forbid")

    from_date: str
    to_date: str
    games: list[CalendarGame]

    @classmethod
    def from_games(
        cls,
        *,
        from_date: str,
        to_date: str,
        games: list[CalendarGame],
    ) -> "CalendarResponse":
        """Shape the router's already-built per-game rows into the response.

        The router resolves team identity and maps the persisted game fields,
        then passes the finished :class:`CalendarGame` rows here — keeping the
        schema decoupled from ORM rows (mirrors
        :meth:`app.schemas.slate.SlateResponse.from_games`).
        """
        return cls(from_date=from_date, to_date=to_date, games=list(games))
