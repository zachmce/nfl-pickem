"""The bot's deterministic Elo rating engine — its independent "number".

This is the COMPUTE ENGINE ONLY behind the hybrid bot-picks architecture: a
single, model-derived team-strength number the bot uses to EXPLAIN and
CROSS-CHECK the market line, NOT to act as a tipster. It answers "how strong is
each team right now, and what home margin does that imply?" — nothing more. It
is deliberately not wired into the Discord bot, any prediction intent, or any
HTTP endpoint here; that wiring is deferred (seed ``bot-my-picks-command.md``).

How it works
------------

It walks a single chronologically-ordered stream — the UNION of

* every :class:`~app.models.HistoricalGame` row (all are completed), and
* every :class:`~app.models.Game` whose ``status == GameStatus.FINAL`` with both
  scores non-null (a SCHEDULED / IN_PROGRESS game, or a FINAL game missing a
  score, never enters the stream),

ordered by ``(season, week, date)`` — forward, updating an Elo rating per
``Team.id`` after each game. Team identity is ``team.id`` across BOTH tables (no
abbreviation mapping at this layer). When the stream crosses into a new season
every current rating is regressed one-third of the way back toward the 1500
mean before that season's games update it. There is NO warm-up skip: every
completed game updates ratings.

The arithmetic is ported VERBATIM from the validated feasibility spike
``.planning/spikes/002-rating-model-vs-line/backtest.py`` (K=20, HFA=55 Elo,
25 Elo/pt, 1/3 between-season regression, 538-style log-margin MOV multiplier,
``result = home_score - away_score`` sign convention). It is NOT re-tuned.

Purity / side effects
---------------------

This is a **read** service: it reads the passed-in ``Session`` and writes
nothing — no ``add``, no ``commit``. It adds NO table and NO Alembic migration;
ratings are computed on demand (the spike walked 6743 games sub-second over a
bounded corpus, so no cache is needed). ``app.db`` is deliberately NOT imported
(it builds the Postgres engine at import time — same rule the seeder follows).

> Note: on this machine there is no bare ``python`` on ``PATH``; use the venv
> interpreter ``.venv/bin/python`` for any commands.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Mapping

from sqlmodel import Session, select

from app.models import Game, GameStatus, HistoricalGame

# --- Elo params, ported VERBATIM from spike 002 (do NOT re-tune; that spike is
# the source of truth for every constant here) ---
K = 20.0  # update speed (spike 002)
HFA_ELO = 55.0  # home-field advantage in Elo pts (~2.2 pts) (spike 002)
ELO_PER_POINT = 25.0  # Elo points per 1 point of margin (spike 002)
REVERT = 1.0 / 3.0  # between-season regression toward the mean (spike 002)
MEAN = 1500.0  # the league-average baseline rating (spike 002)


def mov_mult(margin: float, elo_diff_winner: float) -> float:
    """538-style margin-of-victory multiplier (dampens blowout autocorrelation).

    Copied verbatim from spike 002's ``mov_mult``.
    """
    return math.log(abs(margin) + 1.0) * (2.2 / (elo_diff_winner * 0.001 + 2.2))


def regress_toward_mean(rating: float) -> float:
    """Pull a rating one-third of the way back toward the 1500 mean.

    This is the exact between-season regression the spike applies at each season
    boundary (``MEAN + (rating - MEAN) * (1 - REVERT)``); a mean rating is left
    unchanged. Exposed as a named function so the season-boundary behavior is
    directly unit-testable.
    """
    return MEAN + (rating - MEAN) * (1.0 - REVERT)


def expected_margin(home_team_id: int, away_team_id: int, ratings: Mapping[int, float]) -> float:
    """Predicted HOME margin in points, INCLUDING home-field advantage.

    ``(home_rating - away_rating + HFA_ELO) / ELO_PER_POINT``. A team that never
    appeared in the stream defaults to the 1500 ``MEAN`` (via ``.get``) rather
    than raising, so with equal ratings the home side carries a
    ``HFA_ELO / ELO_PER_POINT`` (== 2.2 pt) edge.
    """
    home = ratings.get(home_team_id, MEAN)
    away = ratings.get(away_team_id, MEAN)
    return (home - away + HFA_ELO) / ELO_PER_POINT


@dataclass(frozen=True)
class MarginEstimate:
    """The expected home margin plus each side's underlying rating."""

    expected_margin: float
    home_rating: float
    away_rating: float


def estimate(home_team_id: int, away_team_id: int, ratings: Mapping[int, float]) -> MarginEstimate:
    """The expected home margin plus each side's rating, in one call.

    A convenience over :func:`expected_margin` that also surfaces the two team
    ratings (defaulting an absent team to ``MEAN``) so a caller can explain the
    number, not just consume it.
    """
    return MarginEstimate(
        expected_margin=expected_margin(home_team_id, away_team_id, ratings),
        home_rating=ratings.get(home_team_id, MEAN),
        away_rating=ratings.get(away_team_id, MEAN),
    )


def estimate_for_game(game: Game, ratings: Mapping[int, float]) -> MarginEstimate:
    """:func:`estimate` driven straight off a :class:`~app.models.Game` row."""
    return estimate(game.home_team_id, game.away_team_id, ratings)


def _apply_game(
    ratings: defaultdict[int, float],
    home_team_id: int,
    away_team_id: int,
    result: int,
) -> None:
    """Walk-forward Elo update for ONE completed game (in place).

    ``result`` is the HOME margin (``home_score - away_score``), matching the
    spike + ``HistoricalGame.result`` sign convention. Copied verbatim from the
    spike's post-prediction update block: the winning side gains ``delta`` and
    the losing side loses the symmetric ``delta``.
    """
    rh = ratings[home_team_id]
    ra = ratings[away_team_id]
    elo_diff = rh - ra + HFA_ELO
    s_home = 1.0 if result > 0 else (0.5 if result == 0 else 0.0)
    e_home = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))
    winner_diff = elo_diff if result > 0 else -elo_diff
    m = mov_mult(result if result != 0 else 1.0, winner_diff)
    delta = K * m * (s_home - e_home)
    ratings[home_team_id] = rh + delta
    ratings[away_team_id] = ra - delta


@dataclass(frozen=True)
class _StreamGame:
    """One completed game normalized into the merged walk-forward stream."""

    season: int
    week: int
    sort_date: date
    home_team_id: int
    away_team_id: int
    result: int  # home margin (home_score - away_score)


def _build_stream(session: Session) -> list[_StreamGame]:
    """Merge BOTH completed-game sources into one ``(season, week, date)`` stream.

    * ``HistoricalGame``: every row (all completed). ``result`` is read straight
      off ``row.result`` (already ``home_score - away_score``, computed by the
      seeder); ``sort_date`` is ``row.gameday``.
    * ``Game``: only ``status == GameStatus.FINAL`` with both scores non-null —
      anything else (SCHEDULED / IN_PROGRESS, or a FINAL row missing a score)
      never enters the stream. ``result`` is computed here as
      ``home_score - away_score``; ``sort_date`` is ``kickoff_at.date()`` when
      present, else ``date.min`` (a deterministic within-week tiebreaker safety
      fallback — a FINAL game realistically has a kickoff).

    Returned SORTED by ``(season, week, sort_date)`` so all of a season's games
    precede the next season's, which the season-boundary regression relies on.
    """
    stream: list[_StreamGame] = []

    for hist in session.exec(select(HistoricalGame)).all():
        stream.append(
            _StreamGame(
                season=hist.season,
                week=hist.week,
                sort_date=hist.gameday,
                home_team_id=hist.home_team_id,
                away_team_id=hist.away_team_id,
                result=hist.result,
            )
        )

    final_games = session.exec(
        select(Game).where(
            Game.status == GameStatus.FINAL,
            Game.home_score.is_not(None),  # type: ignore[union-attr]
            Game.away_score.is_not(None),  # type: ignore[union-attr]
        )
    ).all()
    for game in final_games:
        # Both scores are non-null by the query above; assert for the type
        # checker and as a defensive guard.
        assert game.home_score is not None
        assert game.away_score is not None
        sort_date = game.kickoff_at.date() if game.kickoff_at is not None else date.min
        stream.append(
            _StreamGame(
                season=game.season,
                week=game.week,
                sort_date=sort_date,
                home_team_id=game.home_team_id,
                away_team_id=game.away_team_id,
                result=game.home_score - game.away_score,
            )
        )

    stream.sort(key=lambda g: (g.season, g.week, g.sort_date))
    return stream


def compute_ratings(session: Session) -> dict[int, float]:
    """Walk the merged stream forward and return the current Elo per ``Team.id``.

    Starts every team at the 1500 ``MEAN``; for each game, if the season has
    advanced since the previous game, regresses EVERY current rating one-third of
    the way toward the mean (:func:`regress_toward_mean`) before applying the
    game. Every completed game updates ratings — there is NO warm-up skip.

    Returns a plain ``dict`` snapshot; teams that never played are simply absent
    (and default to ``MEAN`` via :func:`expected_margin` / :func:`estimate`). A
    pure read — writes nothing.
    """
    ratings: defaultdict[int, float] = defaultdict(lambda: MEAN)
    last_season: int | None = None

    for game in _build_stream(session):
        if last_season is not None and game.season != last_season:
            for team_id in list(ratings.keys()):
                ratings[team_id] = regress_toward_mean(ratings[team_id])
        last_season = game.season
        _apply_game(ratings, game.home_team_id, game.away_team_id, game.result)

    return dict(ratings)
