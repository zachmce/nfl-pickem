"""Pure pick'em scoring engine — grade a single pick and score one week.

This module is the core-value piece of the project: *if everything else fails,
picking and scoring must still work*. It is therefore deliberately **pure**:

* It performs **no I/O of any kind** — no database session, no network, no file
  access, no writes. The database engine module is never imported.
* It imports only :mod:`app.models` (for :class:`~app.models.Game`,
  :class:`~app.models.Pick`, :class:`~app.models.GameStatus` and
  :class:`~app.models.PickType`) plus the standard library.
* It **never mutates** the ``Game`` / ``Pick`` instances handed to it. In
  particular it does not set ``Pick.result`` or ``Pick.points`` — persisting
  those is the caller's responsibility. The engine only computes and returns a
  decision.

For every pick type EXCEPT :attr:`~app.models.PickType.MISC` the stored
``Pick.result`` / ``Pick.points`` columns are *vestigial*: the engine re-derives
the outcome from the game's final score on every read. :attr:`PickType.MISC` is
the ONE exception — a free-text prediction whose outcome cannot be derived from
the game. For a MISC pick the engine passes the admin-set stored
``Pick.result`` / ``Pick.points`` THROUGH verbatim (it still reads only ``Pick``
fields and mutates nothing — the game is irrelevant). Because scoring recomputes
on every read, that passthrough is what makes an admin's MISC grade survive — and
never be silently overwritten by — every standings / week recompute.

Scope is **grade + weekly total only**:

* :func:`grade_pick` decides the outcome (win / loss / push / ineligible /
  ungradeable) and the points for a single pick against a single game.
* :func:`score_week` rolls one user's picks for one week into the ``-1..6``
  weekly score.

The season-long cumulative scoreboard is intentionally out of scope here.

Scoring table (single source of truth, see :func:`_points_for`):

============  ===========  ================
outcome       base points  mortal-lock pts
============  ===========  ================
WIN           ``+1``       ``+2``
LOSS          ``0``        ``-1``
PUSH          ``0``        ``0``
INELIGIBLE    ``0``        ``0``
UNGRADEABLE   ``0``        ``0``
============  ===========  ================

A well-formed single-user week of the four auto-graded base types plus one
mortal lock is bounded in ``[-1, 6]``. A graded :attr:`PickType.MISC` pick can
carry ANY admin-set integer (including a value outside that band, or a negative
penalty the admin set explicitly), so a week that includes a MISC pick can push
the weekly total past ``[-1, 6]``. That band describes only the four auto-graded
types; it is documentation, not a clamp — MISC is intentionally not bounded by
it.

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use the venv interpreter ``.venv/bin/python`` for any
> commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Iterable

from app.models import Game, GameStatus, Pick, PickResult, PickType


class GradeOutcome(str, Enum):
    """The resolved state of grading a pick.

    UPPER_SNAKE members, matching the model enum convention. ``WIN`` / ``LOSS``
    are the only point-bearing outcomes; ``PUSH`` (line landed exactly),
    ``INELIGIBLE`` (e.g. a true pick'em has no spread side to grade) and
    ``UNGRADEABLE`` (game not final / score missing) all score zero.
    """

    WIN = "WIN"
    LOSS = "LOSS"
    PUSH = "PUSH"
    # INELIGIBLE is load-bearing for the odds "line-at-lock" policy: a pick whose
    # type is ineligible at the game's FROZEN line (a true pick'em has no spread
    # side; an absent total has no O/U) voids to 0 — never a loss, even as a
    # mortal lock. Odds can drift until freeze, so a pick made while its type was
    # eligible can land on a frozen line where it no longer is; that case must
    # score 0, not -1. See .planning/notes/scheduled-tasks-and-odds-freeze.md.
    INELIGIBLE = "INELIGIBLE"
    UNGRADEABLE = "UNGRADEABLE"


@dataclass(frozen=True)
class GradeResult:
    """Immutable grading decision: the outcome plus the points it earns."""

    outcome: GradeOutcome
    points: int


# Outcomes that never earn points, regardless of the mortal-lock flag.
_ZERO_OUTCOMES = frozenset(
    {GradeOutcome.PUSH, GradeOutcome.INELIGIBLE, GradeOutcome.UNGRADEABLE}
)

_SPREAD_PICK_TYPES = frozenset({PickType.FAVORITE_COVER, PickType.UNDERDOG_COVER})


def _points_for(outcome: GradeOutcome, *, is_mortal_lock: bool) -> int:
    """The one and only place the scoring table lives.

    Base: WIN ``+1``, LOSS ``0``. Mortal lock: WIN ``+2``, LOSS ``-1``. Every
    non-WIN/LOSS outcome scores ``0`` regardless of the mortal-lock flag.
    """
    if outcome in _ZERO_OUTCOMES:
        return 0
    if outcome is GradeOutcome.WIN:
        return 2 if is_mortal_lock else 1
    # outcome is GradeOutcome.LOSS
    return -1 if is_mortal_lock else 0


def _is_true_pickem(game: Game) -> bool:
    """A game with no gradeable spread side.

    True when the spread is absent or zero, or either side of the line
    (favorite/underdog) is unknown. The two spread pick types are ineligible on
    such a game; Over/Under is unaffected.
    """
    return (
        game.spread is None
        or game.spread == 0
        or game.favorite_team_id is None
        or game.underdog_team_id is None
    )


def _spread_outcome(game: Game, pick_type: PickType) -> GradeOutcome:
    """Grade a FAVORITE_COVER / UNDERDOG_COVER pick (game already FINAL)."""
    if _is_true_pickem(game):
        return GradeOutcome.INELIGIBLE

    # Map the favorite/underdog ids to their home/away scores. The FINAL guard in
    # grade_pick guarantees both scores are present.
    if game.favorite_team_id == game.home_team_id:
        favorite_score = game.home_score
        underdog_score = game.away_score
    else:
        favorite_score = game.away_score
        underdog_score = game.home_score

    # Compare in Decimal space so the int margin lines up with the Decimal spread.
    favorite_margin = Decimal(favorite_score - underdog_score)
    spread = game.spread  # positive magnitude the favorite must cover

    if favorite_margin == spread:
        return GradeOutcome.PUSH
    favorite_covered = favorite_margin > spread

    if pick_type is PickType.FAVORITE_COVER:
        return GradeOutcome.WIN if favorite_covered else GradeOutcome.LOSS
    # UNDERDOG_COVER wins exactly when the favorite did not cover.
    return GradeOutcome.LOSS if favorite_covered else GradeOutcome.WIN


def _total_outcome(game: Game, pick_type: PickType) -> GradeOutcome:
    """Grade an OVER / UNDER pick (game already FINAL)."""
    if game.total is None:
        return GradeOutcome.INELIGIBLE

    combined = Decimal(game.home_score + game.away_score)
    total = game.total

    if combined == total:
        return GradeOutcome.PUSH
    went_over = combined > total

    if pick_type is PickType.OVER:
        return GradeOutcome.WIN if went_over else GradeOutcome.LOSS
    # UNDER wins exactly when the combined score did not exceed the total.
    return GradeOutcome.LOSS if went_over else GradeOutcome.WIN


def grade_pick(game: Game, pick: Pick) -> GradeResult:
    """Grade a single ``pick`` against its ``game`` and return the decision.

    Pure: reads only the fields it needs and mutates nothing. Returns a
    :class:`GradeResult` carrying both the :class:`GradeOutcome` and the points.

    Resolution order:

    * **MISC** first — the ONE manually-graded type. Its outcome is NOT derived
      from the game; instead the admin-set stored ``pick.result`` / ``pick.points``
      are passed THROUGH verbatim (see below). The game is irrelevant to a MISC
      grade, so MISC never reaches the FINAL/score guard or the spread/total
      routing.
    * **UNGRADEABLE** next — game not :attr:`GameStatus.FINAL`, or either score
      is ``None``.
    * Spread picks: **INELIGIBLE** on a true pick'em, else compare the favorite's
      margin to the spread (equality is **PUSH**).
    * Total picks: compare the combined score to the total (equality is
      **PUSH**); no total posted is **INELIGIBLE**.

    MISC passthrough (the load-bearing exception)
    ---------------------------------------------

    :attr:`PickType.MISC` is the single pick type whose stored ``result`` /
    ``points`` are AUTHORITATIVE rather than vestigial. For a MISC pick the engine
    maps the stored :class:`~app.models.PickResult` to a :class:`GradeOutcome` and
    returns the admin-set ``pick.points`` UNCHANGED:

    * ``PickResult.WIN``  -> ``GradeResult(GradeOutcome.WIN,  pick.points)``
    * ``PickResult.LOSS`` -> ``GradeResult(GradeOutcome.LOSS, pick.points)``
    * ``PickResult.PENDING`` (ungraded) -> ``GradeResult(GradeOutcome.UNGRADEABLE, 0)``

    ``pick.points`` flows through verbatim (it may be any int — including a value
    outside the historical ``[-1, 6]`` weekly band, or a negative penalty the
    admin set explicitly). MISC is never routed into ``_spread_outcome`` /
    ``_total_outcome`` and its points are never recomputed via ``_points_for`` —
    so an admin's grade survives every recompute-on-read and is never overwritten.
    """
    if pick.pick_type is PickType.MISC:
        if pick.result is PickResult.WIN:
            return GradeResult(GradeOutcome.WIN, pick.points)
        if pick.result is PickResult.LOSS:
            return GradeResult(GradeOutcome.LOSS, pick.points)
        # PENDING / ungraded MISC: scores nothing until an admin decides it.
        return GradeResult(GradeOutcome.UNGRADEABLE, 0)

    if (
        game.status is not GameStatus.FINAL
        or game.home_score is None
        or game.away_score is None
    ):
        return GradeResult(GradeOutcome.UNGRADEABLE, 0)

    if pick.pick_type in _SPREAD_PICK_TYPES:
        outcome = _spread_outcome(game, pick.pick_type)
    else:
        outcome = _total_outcome(game, pick.pick_type)

    points = _points_for(outcome, is_mortal_lock=pick.is_mortal_lock)
    return GradeResult(outcome, points)


def score_week(games_by_id: dict[int, Game], picks: Iterable[Pick]) -> int:
    """Sum a single user's picks for one week into the weekly score.

    Looks each pick's game up in ``games_by_id`` by ``pick.game_id``, grades it,
    and sums the points. Partial weeks are valid — fewer than five picks simply
    contribute fewer points; absent slots contribute nothing. For a well-formed
    single-user week (≤ 4 base picks + ≤ 1 mortal lock) the result is bounded in
    ``[-1, 6]``.
    """
    return sum(
        grade_pick(games_by_id[pick.game_id], pick).points for pick in picks
    )
