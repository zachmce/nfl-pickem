"""The demo ``ScoreboardSource`` — the seeded 2025 fixture positioned around now.

This is the demo/test source: it serves the packaged 2025 NFL fixture (real final
scores + ESPN BET stand-in odds) but *positioned around the real present* via a
constructor ``offset``, deriving SCHEDULED / IN_PROGRESS / FINAL from the real
current time. It lets the demo driver and tests run the whole season "as if live"
without ESPN, and proves the "position 2025 around the real present" idea before
anything is built on it. It structurally satisfies
:class:`~app.scoreboard.port.ScoreboardSource`.

Design — pure core / thin impure shell (mirrors ``scoring`` and ``pick_window``):

* PURE: :func:`derive_status` is the only place ``now`` is a parameter — it maps a
  positioned kickoff and an injected ``now`` to ``(GameStatus, reveal_score)`` and
  is deterministically testable at every boundary.
* IMPURE shell: :meth:`Demo2025Source.fetch_week` loads the fixture, positions
  each kickoff by the offset, and calls :func:`derive_status` with the REAL
  ``datetime.now(timezone.utc)``. There is no virtual clock anywhere else.

Purity layering: this module re-declares the small conventions it needs
(``GAME_DURATION``, a tz-aware guard, kickoff parsing) rather than importing the
sibling ``pick_window`` service or coupling to the seed beyond reading
``fixture_2025.FIXTURE_PATH`` — exactly as ``pick_window`` stays independent of
``scoring``.

Fidelity limit: the fixture has no partial/live scores, so a game derived as
IN_PROGRESS withholds its score (``reveal_score=False``); the real final score is
revealed only once the game derives as FINAL.

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use the venv interpreter ``.venv/bin/python``.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.models import GameStatus
from app.scoreboard.types import ScoreboardGame, ScoreboardOdds, ScoreboardTeam
from app.seeds.fixture_2025 import FIXTURE_PATH

# Approximate NFL game length (~3.5h). Re-declared locally (the value mirrors
# ``pick_window.DEFAULT_GAME_DURATION``) so this module never imports the sibling
# service — purity layering. Used to derive the FINAL boundary from a kickoff.
GAME_DURATION: timedelta = timedelta(hours=3, minutes=30)


def _require_aware(dt: datetime, label: str) -> None:
    """Raise a deliberate, labeled ``ValueError`` if ``dt`` is naive.

    Mirrors ``pick_window._require_aware`` (re-declared locally, not imported):
    turns the bare "can't compare offset-naive and offset-aware" ``TypeError``
    into an explicit error so a wrong-timezone status decision can never be made
    silently.
    """
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware (got a naive datetime)")


def _parse_kickoff(value: Any) -> datetime | None:
    """Parse the fixture's ISO kickoff (e.g. ``2025-09-05T00:20Z``) tz-aware.

    Mirrors ``fixture_2025._parse_kickoff`` (re-declared locally): normalizes a
    trailing ``Z`` to ``+00:00`` so the result is tz-aware. Returns ``None`` for
    missing/blank input.
    """
    if not value or not isinstance(value, str):
        return None
    normalized = value.replace("Z", "+00:00") if value.endswith("Z") else value
    return datetime.fromisoformat(normalized)


def derive_status(
    effective_kickoff: datetime,
    now: datetime,
    *,
    duration: timedelta = GAME_DURATION,
) -> tuple[GameStatus, bool]:
    """Derive ``(status, reveal_score)`` for a positioned game at ``now``.

    Pure and deterministic — ``now`` is injected (the only place a clock value
    enters this module). Both ``effective_kickoff`` and ``now`` must be
    timezone-aware (a naive value raises :class:`ValueError`).

    Boundaries (half-open on the kickoff, inclusive on the FINAL edge):

    * ``now < effective_kickoff``                       -> ``(SCHEDULED, False)``
    * ``effective_kickoff <= now < eff + duration``     -> ``(IN_PROGRESS, False)``
      (the fixture has no partial scores, so the score is withheld — documented
      fidelity limit)
    * ``now >= effective_kickoff + duration``           -> ``(FINAL, True)``
      (reveal the fixture's real final score)
    """
    _require_aware(effective_kickoff, "effective_kickoff")
    _require_aware(now, "now")
    if now < effective_kickoff:
        return GameStatus.SCHEDULED, False
    if now < effective_kickoff + duration:
        return GameStatus.IN_PROGRESS, False
    return GameStatus.FINAL, True


def _normalize_fixture_odds(odds: Any) -> ScoreboardOdds | None:
    """Map a fixture ``odds`` block into :class:`ScoreboardOdds`, else ``None``.

    The fixture odds shape is the flat
    ``{provider, spread, total, favorite_team_id, underdog_team_id}`` written by
    the generator. ``spread`` is carried RAW (signed home-relative) — no abs().
    """
    if not isinstance(odds, dict):
        return None
    return ScoreboardOdds(
        provider=odds.get("provider"),
        # The fixture carries no provider id, so this resolves to None — the demo
        # path stays behavior-identical (no provider id is ever synthesized).
        provider_id=odds.get("provider_id"),
        spread=odds.get("spread"),
        total=odds.get("total"),
        favorite_team_id=odds.get("favorite_team_id"),
        underdog_team_id=odds.get("underdog_team_id"),
    )


def _team(side: Any, *, score: int | None) -> ScoreboardTeam:
    """Build a :class:`ScoreboardTeam` from a fixture home/away block."""
    side = side if isinstance(side, dict) else {}
    return ScoreboardTeam(
        espn_team_id=side.get("team_id"),
        abbreviation=side.get("abbreviation"),
        score=score,
    )


class Demo2025Source:
    """Demo :class:`~app.scoreboard.port.ScoreboardSource` over the 2025 fixture.

    Constructed with an ``offset`` that shifts the fixture's real-2025 kickoffs
    around the present (e.g. a large negative offset makes the whole season
    derive as FINAL; a positive offset makes it SCHEDULED). Status and score
    revelation are computed against the REAL clock in :meth:`fetch_week`.
    """

    def __init__(
        self,
        offset: timedelta = timedelta(),
        *,
        path: Path | None = None,
        duration: timedelta = GAME_DURATION,
    ) -> None:
        self._offset = offset
        self._path = path or FIXTURE_PATH
        self._duration = duration
        self._fixture: dict | None = None  # loaded lazily in fetch_week

    def _load(self) -> dict:
        if self._fixture is None:
            with open(self._path, encoding="utf-8") as fh:
                self._fixture = json.load(fh)
        return self._fixture

    def fetch_week(self, season: int, week: int) -> list[ScoreboardGame]:
        """Serve the fixture's ``(season, week)`` games positioned around now.

        Honors the ``season`` arg by matching it against the fixture's own season
        (returns ``[]`` for a non-matching season). For each matching game:
        parses the fixture kickoff tz-aware, positions it by the offset, derives
        status/reveal against the real ``datetime.now(timezone.utc)``, and builds
        a :class:`ScoreboardGame` whose ``kickoff_at`` is the POSITIONED kickoff
        (so downstream window/lock logic compares against the positioned
        schedule). Scores are revealed only when the game derives as FINAL; odds
        are carried when the fixture game has them.
        """
        fixture = self._load()
        metadata = fixture.get("metadata", {})
        fixture_season = int(metadata.get("season"))
        if season != fixture_season:
            return []

        now = datetime.now(timezone.utc)
        games: list[ScoreboardGame] = []
        for raw in fixture.get("games", []):
            if int(raw.get("week")) != week:
                continue
            fixture_kickoff = _parse_kickoff(raw.get("kickoff"))
            if fixture_kickoff is None:
                continue
            effective_kickoff = fixture_kickoff + self._offset
            status, reveal_score = derive_status(
                effective_kickoff, now, duration=self._duration
            )
            home_raw = raw.get("home", {})
            away_raw = raw.get("away", {})
            home_score = home_raw.get("score") if reveal_score else None
            away_score = away_raw.get("score") if reveal_score else None

            event_id = raw.get("espn_event_id")
            games.append(
                ScoreboardGame(
                    espn_event_id=str(event_id) if event_id is not None else None,
                    season=fixture_season,
                    week=week,
                    kickoff_at=effective_kickoff,
                    status=status,
                    home=_team(home_raw, score=home_score),
                    away=_team(away_raw, score=away_score),
                    odds=_normalize_fixture_odds(raw.get("odds")),
                )
            )
        return games
