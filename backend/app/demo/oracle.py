"""Pure results oracle for the season-walkthrough proof.

Given the preordained bot-picks dataset (:data:`app.seeds.data.bot_picks_2025.BOT_PICKS`)
and the set of seeded 2025 FINAL games, this oracle computes — by hand-off to the
production scoring engine — each bot's per-week weekly scores and the cumulative
season standings. It is the *precomputed-expected* side of the
walkthrough-as-proof strategy (PROJECT.md): the demo driver (#5) drives the same
bots' picks through the real submit/grade/scoreboard stack and asserts
``actual == expected`` against this oracle.

Purity
------

The oracle is **pure**: it imports only :mod:`app.models`,
:func:`app.services.scoring.score_week`, the static picks dataset, and the
standard library. It **never** imports ``app.db``, opens no session, performs no
network or file I/O, and writes nothing. Games are passed in by the caller (which
seeds/loads them read-only). Calling it twice with the same inputs returns equal
results.

What this oracle does and does NOT validate
-------------------------------------------

This computed oracle validates **integration plumbing**, NOT scoring correctness.
It uses the *same* :func:`~app.services.scoring.score_week` that production uses,
so it is structurally incapable of catching a bug in the scoring engine — if
``score_week`` were wrong, both the oracle and production would be wrong in
lockstep and still agree. Scoring correctness is covered by
``tests/test_scoring.py`` (including hand-built ground-truth cases). What the
``actual == expected`` comparison in #5 *does* prove is that the surrounding
integration is wired correctly: that picks were persisted with the right
``(game, type, mortal-lock)`` triples, that games were finalized with the right
scores/odds, and that the scoreboard assembled the per-week and season totals
correctly through the real stack. The hand-anchored spot-checks in
``tests/test_results_oracle.py`` are the independent check that the oracle's own
numbers are right.

> Note: on this machine there is no bare ``python`` on ``PATH``; use the venv
> interpreter ``.venv/bin/python`` for any commands.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.models import Game, Pick
from app.seeds.data.bot_picks_2025 import BotPick
from app.services.scoring import score_week


@dataclass(frozen=True)
class BotSeasonResult:
    """One bot's computed results: weekly scores + season total.

    ``weekly_scores`` maps week number -> that week's integer score (as returned
    by :func:`app.services.scoring.score_week`). ``season_total`` is their sum.
    Frozen + tuple-free-value so two computations compare equal.
    """

    display_name: str
    weekly_scores: dict[int, int]
    season_total: int


@dataclass(frozen=True)
class Standings:
    """The ordered season standings over all bots.

    ``results`` is ordered by ``(-season_total, display_name)`` — descending
    season total, ties broken deterministically by display_name ascending.
    """

    results: tuple[BotSeasonResult, ...]


def games_by_pk_index(games: list[Game]) -> dict[int, Game]:
    """Build the PK-keyed index ``score_week`` expects (``pick.game_id`` -> Game)."""
    return {g.id: g for g in games}


def _games_by_event_index(games: list[Game]) -> dict[int, Game]:
    """Build the stable-event-id index the dataset references."""
    return {g.espn_event_id: g for g in games}


def _picks_for_week(
    bot_picks: list[BotPick], games_by_event: dict[int, Game]
) -> list[Pick]:
    """Materialize transient ``Pick`` instances from dataset records.

    Resolves each record's stable ``espn_event_id`` to the real seeded ``Game``
    so ``score_week``'s PK-keyed ``pick.game_id`` lookups resolve. These Pick
    objects are never persisted — they exist only to feed the pure scorer.
    """
    picks: list[Pick] = []
    for bp in bot_picks:
        game = games_by_event[bp.espn_event_id]
        picks.append(
            Pick(
                user_id=0,  # transient: never persisted
                game_id=game.id,
                week_id=game.week_id,
                pick_type=bp.pick_type,
                is_mortal_lock=bp.is_mortal_lock,
            )
        )
    return picks


def compute_standings(
    bot_picks: dict[str, dict[int, list[BotPick]]],
    games: list[Game],
) -> Standings:
    """Compute per-bot weekly scores + ordered season standings.

    For each bot, for each week in its dataset, build transient ``Pick`` instances
    (resolving ``espn_event_id`` to the seeded ``Game``), delegate to
    :func:`app.services.scoring.score_week`, and accumulate weekly + season
    totals. A partial roster simply contributes its present picks (absent slots
    score 0 — see ``score_week``). Returns :class:`Standings` ordered by
    ``(-season_total, display_name)``.

    Pure: reads only the passed-in games + the static dataset, opens no session,
    writes nothing.
    """
    games_by_event = _games_by_event_index(games)
    games_by_pk = games_by_pk_index(games)

    results: list[BotSeasonResult] = []
    for display_name, weeks in bot_picks.items():
        weekly_scores: dict[int, int] = {}
        for week_number, records in weeks.items():
            picks = _picks_for_week(records, games_by_event)
            weekly_scores[week_number] = score_week(games_by_pk, picks)
        results.append(
            BotSeasonResult(
                display_name=display_name,
                weekly_scores=weekly_scores,
                season_total=sum(weekly_scores.values()),
            )
        )

    results.sort(key=lambda r: (-r.season_total, r.display_name))
    return Standings(results=tuple(results))
