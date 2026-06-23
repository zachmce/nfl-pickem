"""The ``ScoreboardSource`` port — the seam between ingest and any score source.

This is the keystone abstraction of the scoreboard architecture: the future
poller/ingest depends on this ``Protocol`` and NEVER touches ESPN (or urllib, or
the fixture) directly. Production wires the real ESPN adapter
(:class:`~app.scoreboard.espn.EspnScoreboardSource`); demo/test wires
:class:`~app.scoreboard.demo.Demo2025Source` (the seeded 2025 fixture positioned
around the real present). The poller can be unit-tested entirely against the
fake.

Why a :class:`typing.Protocol` (structural typing) rather than an ABC: the
codebase has no ABC hierarchies — it favors lightweight structural types, frozen
dataclasses, and enums. ``@runtime_checkable`` lets tests assert conformance with
``isinstance`` while keeping adapters fully decoupled (they need not inherit
anything).

Method granularity is **per-week** because the ESPN site scoreboard endpoint is
itself per-week (``...scoreboard?dates={season}&seasontype=2&week={week}``):
``fetch_week(season, week) -> list[ScoreboardGame]``.

Failure contract: an adapter that cannot fetch raises :class:`ScoreboardFetchError`
(a typed error) rather than silently returning an empty list — an empty result
means "this week genuinely has no games", never "the fetch failed".
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.scoreboard.types import ScoreboardGame


class ScoreboardFetchError(RuntimeError):
    """Raised by an adapter when fetching a week's scoreboard fails.

    Carries the failing URL and the underlying reason where available. Adapters
    raise this on transport failure (HTTP error, network/URL error, timeout)
    instead of returning ``[]`` so callers can distinguish a genuine empty week
    from a failed fetch.
    """


@runtime_checkable
class ScoreboardSource(Protocol):
    """A source of normalized weekly NFL scoreboard data.

    Implementations fetch (or synthesize) one regular-season week and return it
    as a list of :class:`~app.scoreboard.types.ScoreboardGame`. They MUST raise
    :class:`ScoreboardFetchError` on fetch failure rather than returning an empty
    list.
    """

    def fetch_week(self, season: int, week: int) -> list[ScoreboardGame]:
        """Return the normalized games for ``(season, week)``.

        :param season: the NFL season year (e.g. ``2025``).
        :param week: the regular-season week number (``1``..``18``).
        :returns: a list of normalized :class:`ScoreboardGame` (possibly empty
            for a week that genuinely has no games).
        :raises ScoreboardFetchError: if the underlying fetch fails.
        """
        ...
