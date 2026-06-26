"""Normalized scoreboard data contract — the output of every ``ScoreboardSource``.

These frozen dataclasses are the *port's data contract*: the shape every adapter
(real ESPN, demo fixture) normalizes into, and the only scoreboard types the
future poller/ingest will consume. They are deliberately decoupled from the
SQLModel :class:`~app.models.Game` row — an adapter knows how to produce these,
the poller knows how to map these onto a ``Game``; neither side knows about the
other's source JSON.

Design conventions mirror the sibling pure services (``scoring`` /
``pick_window``): ``@dataclass(frozen=True)`` value objects, tz-aware datetimes,
and the project's :class:`~app.models.GameStatus` enum (the same enum the poller
writes to the DB).

These map cleanly from BOTH live ESPN site-scoreboard JSON and the seeded 2025
fixture, carrying exactly what the poller needs to match an existing ``Game`` row
(``espn_event_id``, team ids) and update it (status, scores, kickoff, odds).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.models import GameStatus


@dataclass(frozen=True)
class ScoreboardTeam:
    """One side (home or away) of a normalized scoreboard game.

    ``score`` is ``None`` until the game is FINAL (or whenever the source
    withholds a live score) — a present score is never assumed for a
    not-yet-final game.
    """

    espn_team_id: str | None
    abbreviation: str | None
    score: int | None = None


@dataclass(frozen=True)
class ScoreboardOdds:
    """A normalized betting line attached to a scoreboard game (optional block).

    Populated when the source carries odds (DraftKings inline on upcoming live
    games; ESPN BET in the seeded fixture). Provided for completeness so the
    future ingest has the full line — *freezing/persisting* the line is OUT OF
    SCOPE of the scoreboard port.

    ``spread`` carries the RAW normalized value exactly as the source reports it
    (the live/fixture spread is signed home-relative, e.g. ``-3.5`` = home
    favored by 3.5). This type does NOT take ``abs()`` or otherwise reinterpret
    the sign — magnitude/direction handling is the importer's/poller's job and is
    out of scope here. Do not assume this is a positive magnitude.
    """

    provider: str | None = None
    # The chosen provider's id, captured for auditability ALONGSIDE ``provider``
    # (the name). Provider ids drift across ESPN endpoints/time, so the importer
    # persists this exact id from the SAME selected odds item rather than ever
    # hardcoding one. Additive/optional: every existing constructor call (which
    # passes no ``provider_id``) stays valid and yields ``None``.
    provider_id: str | None = None
    spread: float | None = None
    total: float | None = None
    favorite_team_id: str | None = None
    underdog_team_id: str | None = None


@dataclass(frozen=True)
class ScoreboardGame:
    """A single normalized game — the unit a ``ScoreboardSource`` yields per week.

    ``kickoff_at`` is tz-aware (or ``None`` when the source omits a scheduled
    time). For the demo adapter this is the *positioned* kickoff (fixture kickoff
    plus the constructor offset) so downstream window/lock logic compares against
    the schedule as positioned around the real present.

    ``home``/``away`` scores are ``None`` until FINAL / when withheld (see
    :class:`ScoreboardTeam`). ``odds`` is ``None`` when the source carries no
    line for the game.
    """

    espn_event_id: str | None
    season: int
    week: int
    kickoff_at: datetime | None
    status: GameStatus
    home: ScoreboardTeam
    away: ScoreboardTeam
    odds: ScoreboardOdds | None = None
