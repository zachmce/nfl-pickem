"""Pure pick conflict + roster validation — legality checks before persistence.

This is a sibling to :mod:`app.services.scoring`: it encodes the *conflict and
eligibility* rules a user's weekly pick roster must satisfy, so a caller (the
future pick-submission endpoint) can reject contradictory picks deterministically
*before* writing them. Like the scoring engine it is deliberately **pure**:

* It performs **no I/O of any kind** — no database session, no network, no file
  access, no writes. The database engine module (``app.db``) is never imported.
* It imports only :mod:`app.models` (for :class:`~app.models.Game`,
  :class:`~app.models.Pick` and :class:`~app.models.PickType`) plus the standard
  library. It does **not** import :mod:`app.services.scoring` — services do not
  import each other (purity layering).
* It **never mutates** the ``Game`` / ``Pick`` instances handed to it. It only
  inspects them and returns a decision.

Scope is **conflict + eligibility validation only**:

* :func:`validate_roster` checks a whole set of picks for duplicates,
  contradictions, more than one base pick of the same type (the *slot* rule),
  more than one mortal lock, and spread picks on a true pick'em.
* :func:`check_new_pick` enforces *first-pick precedence*: it validates a single
  incoming pick against an already-accepted set; if the new pick conflicts, it is
  rejected and the existing pick wins.

Rules (single source of truth):

* **Duplicate** — the same ``pick_type`` twice on the same ``game_id``.
* **Contradiction** — two mutually-exclusive types on the same game:
  ``UNDERDOG_COVER`` vs ``FAVORITE_COVER``, or ``OVER`` vs ``UNDER``. A spread
  pick and a total pick on the same game are *independent* and allowed.
* **Duplicate base type (the slot model)** — more than one *base*
  (non-mortal-lock) pick of the same ``pick_type`` in the week, even across
  DIFFERENT games. The league is "one of each of four bet types" (PROJECT.md):
  each base ``pick_type`` is a single weekly SLOT, so a roster may hold at most
  one base pick of each type. The mortal lock is the *only* allowed same-type
  duplicate (it occupies its own slot alongside the four base ones). This is the
  whole-roster mirror of the DB partial unique index
  ``uq_pick_user_week_type_base`` and of ``pick_submission``'s base-slot upsert:
  submitting a second base pick of a type that already exists is a slot
  *replacement* at submission time, but a single roster that *literally
  contains* two base picks of one type is malformed and rejected here.
* **Multiple mortal locks** — more than one pick flagged ``is_mortal_lock``. The
  mortal-lock flag never exempts a pick from duplicate / contradiction checks.
* **Pick'em spread ineligibility** — a ``FAVORITE_COVER`` / ``UNDERDOG_COVER``
  pick on a true pick'em game (no gradeable spread side). Over/Under is unaffected.

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use the venv interpreter ``.venv/bin/python`` for any
> commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from app.models import Game, Pick, PickType


class ViolationCode(str, Enum):
    """Why a pick (or pair of picks) is illegal.

    UPPER_SNAKE members, matching the model enum convention.
    """

    DUPLICATE_PICK = "DUPLICATE_PICK"
    CONTRADICTORY_PICK = "CONTRADICTORY_PICK"
    DUPLICATE_BASE_TYPE = "DUPLICATE_BASE_TYPE"
    MULTIPLE_MORTAL_LOCKS = "MULTIPLE_MORTAL_LOCKS"
    PICKEM_SPREAD_INELIGIBLE = "PICKEM_SPREAD_INELIGIBLE"


@dataclass(frozen=True)
class Violation:
    """A single rule violation.

    ``picks`` carries the offending pick(s): one entry for single-pick
    violations (duplicate-mortal-lock / pick'em ineligibility), two entries for
    pairwise conflicts (duplicate / contradiction), and the full set of extras
    for :attr:`ViolationCode.MULTIPLE_MORTAL_LOCKS`. This lets a caller map the
    decision back to the rejected pick(s).
    """

    code: ViolationCode
    message: str
    picks: tuple[Pick, ...]


@dataclass(frozen=True)
class ValidationResult:
    """Immutable validation decision.

    ``ok`` is ``True`` exactly when ``violations`` is empty — the two are always
    consistent because the result is only ever built via :func:`_result`.
    """

    ok: bool
    violations: tuple[Violation, ...]


def _result(violations: Iterable[Violation]) -> ValidationResult:
    """The one and only place a :class:`ValidationResult` is constructed.

    Derives ``ok`` from whether any violations were collected, so the two can
    never disagree.
    """
    collected = tuple(violations)
    return ValidationResult(ok=not collected, violations=collected)


# The two spread pick types, mirroring scoring._SPREAD_PICK_TYPES for consistency.
_SPREAD_PICK_TYPES = frozenset({PickType.FAVORITE_COVER, PickType.UNDERDOG_COVER})

# Mutually-exclusive type pairs: picking both on the same game is a contradiction.
_CONTRADICTORY_PAIRS = frozenset(
    {
        frozenset({PickType.UNDERDOG_COVER, PickType.FAVORITE_COVER}),
        frozenset({PickType.OVER, PickType.UNDER}),
    }
)


def _is_true_pickem(game: Game) -> bool:
    """A game with no gradeable spread side.

    True when the spread is absent or zero, or either side of the line
    (favorite/underdog) is unknown. The two spread pick types are ineligible on
    such a game; Over/Under is unaffected. Semantics are intentionally identical
    to :func:`app.services.scoring._is_true_pickem` — duplicated locally rather
    than imported so the two pure services stay independent (scoring.py is the
    consistency anchor).
    """
    return (
        game.spread is None
        or game.spread == 0
        or game.favorite_team_id is None
        or game.underdog_team_id is None
    )


def _contradict(type_a: PickType, type_b: PickType) -> bool:
    """True when two pick types on the same game are mutually exclusive."""
    return frozenset({type_a, type_b}) in _CONTRADICTORY_PAIRS


def validate_roster(
    picks: Iterable[Pick], games_by_id: dict[int, Game]
) -> ValidationResult:
    """Validate a whole roster of picks for legality.

    Detects, across the provided picks:

    * **DUPLICATE_PICK** — two picks sharing ``pick_type`` on the same game.
    * **CONTRADICTORY_PICK** — a pair of mutually-exclusive types on the same
      game (favorite/underdog or over/under).
    * **DUPLICATE_BASE_TYPE** — more than one *base* (non-mortal-lock) pick of the
      same ``pick_type`` in the roster, even on DIFFERENT games (the slot model:
      one base pick per type per week). The mortal lock is the only same-type
      duplicate the slot model permits.
    * **MULTIPLE_MORTAL_LOCKS** — more than one ``is_mortal_lock`` pick (a single
      violation listing the offending picks).
    * **PICKEM_SPREAD_INELIGIBLE** — a spread pick on a true pick'em game.

    The mortal-lock flag never exempts a pick from duplicate / contradiction
    checks. A spread pick and a total pick on the same game are allowed. Picks on
    different games never produce a per-game (duplicate/contradiction) violation,
    but two BASE picks of one type DO violate the slot model regardless of game.

    A pick referencing a ``game_id`` absent from ``games_by_id`` is a programmer
    error (not a validation violation): the ``games_by_id[...]`` lookup raises
    ``KeyError`` and is allowed to propagate, matching
    :func:`app.services.scoring.score_week`.
    """
    pick_list = list(picks)
    violations: list[Violation] = []

    # (d) Pick'em spread ineligibility — per spread pick.
    for pick in pick_list:
        if pick.pick_type in _SPREAD_PICK_TYPES and _is_true_pickem(
            games_by_id[pick.game_id]
        ):
            violations.append(
                Violation(
                    code=ViolationCode.PICKEM_SPREAD_INELIGIBLE,
                    message=(
                        f"{pick.pick_type.value} is ineligible on game "
                        f"{pick.game_id}: it is a true pick'em (no spread side)."
                    ),
                    picks=(pick,),
                )
            )

    # (a)/(b) Duplicates + contradictions — grouped by game, all unordered pairs.
    by_game: dict[int, list[Pick]] = {}
    for pick in pick_list:
        by_game.setdefault(pick.game_id, []).append(pick)

    for game_id, game_picks in by_game.items():
        for i in range(len(game_picks)):
            for j in range(i + 1, len(game_picks)):
                a, b = game_picks[i], game_picks[j]
                if a.pick_type == b.pick_type:
                    violations.append(
                        Violation(
                            code=ViolationCode.DUPLICATE_PICK,
                            message=(
                                f"Duplicate {a.pick_type.value} picks on game "
                                f"{game_id}."
                            ),
                            picks=(a, b),
                        )
                    )
                elif _contradict(a.pick_type, b.pick_type):
                    violations.append(
                        Violation(
                            code=ViolationCode.CONTRADICTORY_PICK,
                            message=(
                                f"Contradictory picks on game {game_id}: "
                                f"{a.pick_type.value} and {b.pick_type.value}."
                            ),
                            picks=(a, b),
                        )
                    )

    # (c) The slot model: at most one BASE (non-mortal-lock) pick per pick_type
    # across the whole roster, even on different games. The mortal lock is the
    # only allowed same-type duplicate — it occupies its own slot. This mirrors
    # the DB partial unique index uq_pick_user_week_type_base.
    base_by_type: dict[PickType, list[Pick]] = {}
    for pick in pick_list:
        if pick.is_mortal_lock:
            continue
        base_by_type.setdefault(pick.pick_type, []).append(pick)
    for pick_type, base_picks in base_by_type.items():
        if len(base_picks) > 1:
            violations.append(
                Violation(
                    code=ViolationCode.DUPLICATE_BASE_TYPE,
                    message=(
                        f"A roster may have at most one base {pick_type.value} "
                        f"pick per week (the slot model); found "
                        f"{len(base_picks)}. Only the mortal lock may repeat a "
                        "type."
                    ),
                    picks=tuple(base_picks),
                )
            )

    # (d) At most one mortal lock across the whole roster.
    mortal_locks = [p for p in pick_list if p.is_mortal_lock]
    if len(mortal_locks) > 1:
        violations.append(
            Violation(
                code=ViolationCode.MULTIPLE_MORTAL_LOCKS,
                message=(
                    f"A roster may have at most one mortal lock; found "
                    f"{len(mortal_locks)}."
                ),
                picks=tuple(mortal_locks),
            )
        )

    return _result(violations)


def check_new_pick(
    new_pick: Pick,
    existing_picks: Iterable[Pick],
    games_by_id: dict[int, Game],
) -> ValidationResult:
    """Validate one incoming ``new_pick`` against an already-accepted set.

    Implements **first-pick precedence**: the existing picks are assumed legal and
    are *not* re-validated against themselves. Only the new pick is judged. It is
    rejected when it would:

    * be a spread pick on a true pick'em game (**PICKEM_SPREAD_INELIGIBLE**),
    * duplicate an existing pick on the same game (**DUPLICATE_PICK**),
    * contradict an existing pick on the same game (**CONTRADICTORY_PICK**), or
    * be a second mortal lock when an existing pick already is one
      (**MULTIPLE_MORTAL_LOCKS**).

    Every violation's ``picks`` tuple carries ``new_pick`` (plus the conflicting
    existing pick for pairwise cases) so the caller knows the *new* pick is the
    one being rejected. Returns ``ok=True`` with no violations when the new pick
    is acceptable.
    """
    violations: list[Violation] = []

    # Spread pick on a true pick'em — independent of the existing set.
    if new_pick.pick_type in _SPREAD_PICK_TYPES and _is_true_pickem(
        games_by_id[new_pick.game_id]
    ):
        violations.append(
            Violation(
                code=ViolationCode.PICKEM_SPREAD_INELIGIBLE,
                message=(
                    f"{new_pick.pick_type.value} is ineligible on game "
                    f"{new_pick.game_id}: it is a true pick'em (no spread side)."
                ),
                picks=(new_pick,),
            )
        )

    # Conflicts against each existing pick on the same game.
    for existing in existing_picks:
        if existing.game_id != new_pick.game_id:
            continue
        if existing.pick_type == new_pick.pick_type:
            violations.append(
                Violation(
                    code=ViolationCode.DUPLICATE_PICK,
                    message=(
                        f"Duplicate {new_pick.pick_type.value} pick on game "
                        f"{new_pick.game_id}: an identical pick already exists."
                    ),
                    picks=(new_pick, existing),
                )
            )
        elif _contradict(existing.pick_type, new_pick.pick_type):
            violations.append(
                Violation(
                    code=ViolationCode.CONTRADICTORY_PICK,
                    message=(
                        f"{new_pick.pick_type.value} on game {new_pick.game_id} "
                        f"contradicts the existing {existing.pick_type.value} pick."
                    ),
                    picks=(new_pick, existing),
                )
            )

    # Second mortal lock — the new one is rejected, the existing lock wins.
    if new_pick.is_mortal_lock and any(p.is_mortal_lock for p in existing_picks):
        violations.append(
            Violation(
                code=ViolationCode.MULTIPLE_MORTAL_LOCKS,
                message=(
                    "A roster may have at most one mortal lock; one already "
                    "exists, so this new mortal lock is rejected."
                ),
                picks=(new_pick,),
            )
        )

    return _result(violations)
