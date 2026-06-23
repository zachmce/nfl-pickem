"""Offline unit tests for the pure pick conflict + roster validation service.

These tests exercise :mod:`app.services.pick_validation` with hand-built
``Game`` / ``Pick`` model instances (no database needed — they are plain
SQLModel objects), covering every conflict, eligibility and roster rule plus the
first-pick-precedence path of :func:`check_new_pick`.

Everything runs offline:

* the synthetic tests touch no database at all,
* there is no network access of any kind,
* the validation service performs no I/O and imports only ``app.models`` plus
  the standard library.

Run from the ``backend/`` directory with the standard library test runner::

    cd backend && python -m unittest tests.test_pick_validation -v

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use ``python3 -m unittest ...`` or the venv
> interpreter ``.venv/bin/python -m unittest ...``.

No pytest dependency is required (none is configured for this project).
"""

from __future__ import annotations

import unittest
from decimal import Decimal

from app.models import Game, GameStatus, Pick, PickType
from app.services.pick_validation import (
    ValidationResult,
    Violation,
    ViolationCode,
    check_new_pick,
    validate_roster,
)

# Small constant team ids for the synthetic games. The validator only compares
# ids; it never loads Team rows, so these need not exist in any database.
HOME = 1
AWAY = 2


def _game(
    *,
    spread: Decimal | None = Decimal("3.5"),
    total: Decimal | None = Decimal("44.5"),
    favorite_team_id: int | None = HOME,
    underdog_team_id: int | None = AWAY,
    status: GameStatus = GameStatus.SCHEDULED,
    game_id: int = 100,
) -> Game:
    """Build a synthetic ``Game`` instance with just the fields the validator reads.

    Defaults describe a normal spread-eligible game (a posted line with both
    sides known). Pass ``spread=Decimal("0")`` / ``None`` with
    ``favorite_team_id=None`` / ``underdog_team_id=None`` for a true pick'em.
    """
    return Game(
        id=game_id,
        espn_event_id=game_id,
        week_id=1,
        season=2025,
        week=1,
        home_team_id=HOME,
        away_team_id=AWAY,
        status=status,
        spread=spread,
        total=total,
        favorite_team_id=favorite_team_id,
        underdog_team_id=underdog_team_id,
    )


def _pickem_game(*, game_id: int = 100) -> Game:
    """A true pick'em: no gradeable spread side (zero spread, sides unknown)."""
    return _game(
        spread=Decimal("0"),
        favorite_team_id=None,
        underdog_team_id=None,
        game_id=game_id,
    )


def _pick(
    pick_type: PickType,
    *,
    is_mortal_lock: bool = False,
    game_id: int = 100,
) -> Pick:
    """Build a synthetic ``Pick`` instance for the validator."""
    return Pick(
        user_id=1,
        game_id=game_id,
        week_id=1,
        pick_type=pick_type,
        is_mortal_lock=is_mortal_lock,
    )


def _games_by_id(*games: Game) -> dict[int, Game]:
    return {g.id: g for g in games}


def _codes(result: ValidationResult) -> set[ViolationCode]:
    return {v.code for v in result.violations}


class RosterValidTests(unittest.TestCase):
    """Legal rosters (full + partial) validate with no violations."""

    def test_valid_full_roster(self) -> None:
        # 4 distinct base types on 4 distinct games + 1 mortal lock on a 5th.
        g1 = _game(game_id=1)
        g2 = _game(game_id=2)
        g3 = _game(game_id=3)
        g4 = _game(game_id=4)
        g5 = _game(game_id=5)
        picks = [
            _pick(PickType.FAVORITE_COVER, game_id=1),
            _pick(PickType.UNDERDOG_COVER, game_id=2),
            _pick(PickType.OVER, game_id=3),
            _pick(PickType.UNDER, game_id=4),
            _pick(PickType.OVER, is_mortal_lock=True, game_id=5),
        ]
        result = validate_roster(picks, _games_by_id(g1, g2, g3, g4, g5))
        self.assertTrue(result.ok)
        self.assertEqual(result.violations, ())

    def test_valid_partial_roster(self) -> None:
        g1 = _game(game_id=1)
        g2 = _game(game_id=2)
        picks = [
            _pick(PickType.FAVORITE_COVER, game_id=1),
            _pick(PickType.OVER, game_id=2),
        ]
        result = validate_roster(picks, _games_by_id(g1, g2))
        self.assertTrue(result.ok)
        self.assertEqual(result.violations, ())

    def test_empty_roster_is_ok(self) -> None:
        result = validate_roster([], {})
        self.assertTrue(result.ok)
        self.assertEqual(result.violations, ())

    def test_different_games_never_conflict(self) -> None:
        # One of each type, all on different games -> no cross-game violation.
        g1 = _game(game_id=1)
        g2 = _game(game_id=2)
        g3 = _game(game_id=3)
        g4 = _game(game_id=4)
        picks = [
            _pick(PickType.FAVORITE_COVER, game_id=1),
            _pick(PickType.UNDERDOG_COVER, game_id=2),
            _pick(PickType.OVER, game_id=3),
            _pick(PickType.UNDER, game_id=4),
        ]
        result = validate_roster(picks, _games_by_id(g1, g2, g3, g4))
        self.assertTrue(result.ok)


class ConflictTests(unittest.TestCase):
    """Duplicates, contradictions, and the allowed spread+total pairing."""

    def test_duplicate_same_type_same_game(self) -> None:
        g = _game(game_id=1)
        picks = [
            _pick(PickType.OVER, game_id=1),
            _pick(PickType.OVER, game_id=1),
        ]
        result = validate_roster(picks, _games_by_id(g))
        self.assertFalse(result.ok)
        self.assertIn(ViolationCode.DUPLICATE_PICK, _codes(result))

    def test_underdog_vs_favorite_same_game_contradicts(self) -> None:
        g = _game(game_id=1)
        picks = [
            _pick(PickType.UNDERDOG_COVER, game_id=1),
            _pick(PickType.FAVORITE_COVER, game_id=1),
        ]
        result = validate_roster(picks, _games_by_id(g))
        self.assertFalse(result.ok)
        self.assertIn(ViolationCode.CONTRADICTORY_PICK, _codes(result))

    def test_over_vs_under_same_game_contradicts(self) -> None:
        g = _game(game_id=1)
        picks = [
            _pick(PickType.OVER, game_id=1),
            _pick(PickType.UNDER, game_id=1),
        ]
        result = validate_roster(picks, _games_by_id(g))
        self.assertFalse(result.ok)
        self.assertIn(ViolationCode.CONTRADICTORY_PICK, _codes(result))

    def test_spread_plus_total_same_game_allowed(self) -> None:
        # FAVORITE_COVER + OVER on the same game are independent outcomes.
        g = _game(game_id=1)
        picks = [
            _pick(PickType.FAVORITE_COVER, game_id=1),
            _pick(PickType.OVER, game_id=1),
        ]
        result = validate_roster(picks, _games_by_id(g))
        self.assertTrue(result.ok)
        self.assertEqual(result.violations, ())

    def test_contradiction_violation_carries_both_picks(self) -> None:
        g = _game(game_id=1)
        p1 = _pick(PickType.OVER, game_id=1)
        p2 = _pick(PickType.UNDER, game_id=1)
        result = validate_roster([p1, p2], _games_by_id(g))
        contradiction = next(
            v for v in result.violations if v.code is ViolationCode.CONTRADICTORY_PICK
        )
        self.assertEqual(len(contradiction.picks), 2)
        self.assertIn(p1, contradiction.picks)
        self.assertIn(p2, contradiction.picks)


class SlotModelTests(unittest.TestCase):
    """The slot model: at most one BASE pick per pick_type per week.

    PROJECT.md is "one of each of four bet types" plus one mortal lock. Each base
    ``pick_type`` is a single weekly slot, so two base picks of the same type —
    even on different games — are a malformed roster. The mortal lock is the only
    same-type duplicate the slot model allows. This mirrors the DB partial unique
    index ``uq_pick_user_week_type_base``.
    """

    def test_two_base_same_type_different_games_rejected(self) -> None:
        # Two base OVER picks on DIFFERENT games — the malformed-batch case.
        g1 = _game(game_id=1)
        g2 = _game(game_id=2)
        picks = [
            _pick(PickType.OVER, game_id=1),
            _pick(PickType.OVER, game_id=2),
        ]
        result = validate_roster(picks, _games_by_id(g1, g2))
        self.assertFalse(result.ok)
        self.assertIn(ViolationCode.DUPLICATE_BASE_TYPE, _codes(result))

    def test_base_plus_mortal_lock_same_type_allowed(self) -> None:
        # A base OVER + a mortal-lock OVER on different games is allowed: the
        # mortal lock occupies its own slot alongside the four base slots.
        g1 = _game(game_id=1)
        g2 = _game(game_id=2)
        picks = [
            _pick(PickType.OVER, game_id=1),
            _pick(PickType.OVER, is_mortal_lock=True, game_id=2),
        ]
        result = validate_roster(picks, _games_by_id(g1, g2))
        self.assertTrue(result.ok)
        self.assertEqual(result.violations, ())

    def test_one_of_each_base_type_allowed(self) -> None:
        # The canonical full roster: one of each of the four base types on four
        # distinct games + a mortal lock that repeats one type.
        g1 = _game(game_id=1)
        g2 = _game(game_id=2)
        g3 = _game(game_id=3)
        g4 = _game(game_id=4)
        g5 = _game(game_id=5)
        picks = [
            _pick(PickType.FAVORITE_COVER, game_id=1),
            _pick(PickType.UNDERDOG_COVER, game_id=2),
            _pick(PickType.OVER, game_id=3),
            _pick(PickType.UNDER, game_id=4),
            _pick(PickType.FAVORITE_COVER, is_mortal_lock=True, game_id=5),
        ]
        result = validate_roster(picks, _games_by_id(g1, g2, g3, g4, g5))
        self.assertTrue(result.ok)
        self.assertEqual(result.violations, ())

    def test_duplicate_base_type_violation_carries_both_picks(self) -> None:
        g1 = _game(game_id=1)
        g2 = _game(game_id=2)
        p1 = _pick(PickType.UNDER, game_id=1)
        p2 = _pick(PickType.UNDER, game_id=2)
        result = validate_roster([p1, p2], _games_by_id(g1, g2))
        dup = next(
            v
            for v in result.violations
            if v.code is ViolationCode.DUPLICATE_BASE_TYPE
        )
        self.assertIn(p1, dup.picks)
        self.assertIn(p2, dup.picks)

    def test_two_mortal_locks_same_type_is_mortal_lock_violation(self) -> None:
        # Two mortal locks of the same type are caught by the mortal-lock rule,
        # NOT the base-type rule (mortal locks are excluded from the slot count).
        g1 = _game(game_id=1)
        g2 = _game(game_id=2)
        picks = [
            _pick(PickType.OVER, is_mortal_lock=True, game_id=1),
            _pick(PickType.OVER, is_mortal_lock=True, game_id=2),
        ]
        result = validate_roster(picks, _games_by_id(g1, g2))
        self.assertFalse(result.ok)
        self.assertIn(ViolationCode.MULTIPLE_MORTAL_LOCKS, _codes(result))
        self.assertNotIn(ViolationCode.DUPLICATE_BASE_TYPE, _codes(result))


class MortalLockTests(unittest.TestCase):
    """At most one mortal lock per roster; the flag never exempts a pick."""

    def test_two_mortal_locks_flagged(self) -> None:
        # Two mortal locks on different, otherwise non-conflicting games.
        g1 = _game(game_id=1)
        g2 = _game(game_id=2)
        picks = [
            _pick(PickType.FAVORITE_COVER, is_mortal_lock=True, game_id=1),
            _pick(PickType.OVER, is_mortal_lock=True, game_id=2),
        ]
        result = validate_roster(picks, _games_by_id(g1, g2))
        self.assertFalse(result.ok)
        self.assertIn(ViolationCode.MULTIPLE_MORTAL_LOCKS, _codes(result))

    def test_single_mortal_lock_ok(self) -> None:
        g1 = _game(game_id=1)
        g2 = _game(game_id=2)
        picks = [
            _pick(PickType.FAVORITE_COVER, is_mortal_lock=True, game_id=1),
            _pick(PickType.OVER, game_id=2),
        ]
        result = validate_roster(picks, _games_by_id(g1, g2))
        self.assertTrue(result.ok)

    def test_mortal_lock_does_not_exempt_contradiction(self) -> None:
        # A mortal-lock OVER + a base UNDER on the same game still contradict.
        g = _game(game_id=1)
        picks = [
            _pick(PickType.OVER, is_mortal_lock=True, game_id=1),
            _pick(PickType.UNDER, game_id=1),
        ]
        result = validate_roster(picks, _games_by_id(g))
        self.assertFalse(result.ok)
        self.assertIn(ViolationCode.CONTRADICTORY_PICK, _codes(result))


class PickemEligibilityTests(unittest.TestCase):
    """Spread picks are ineligible on a true pick'em; Over/Under is unaffected."""

    def test_spread_pick_on_pickem_ineligible(self) -> None:
        g = _pickem_game(game_id=1)
        picks = [_pick(PickType.FAVORITE_COVER, game_id=1)]
        result = validate_roster(picks, _games_by_id(g))
        self.assertFalse(result.ok)
        self.assertIn(ViolationCode.PICKEM_SPREAD_INELIGIBLE, _codes(result))

    def test_underdog_pick_on_pickem_ineligible(self) -> None:
        g = _pickem_game(game_id=1)
        picks = [_pick(PickType.UNDERDOG_COVER, game_id=1)]
        result = validate_roster(picks, _games_by_id(g))
        self.assertFalse(result.ok)
        self.assertIn(ViolationCode.PICKEM_SPREAD_INELIGIBLE, _codes(result))

    def test_over_on_pickem_allowed(self) -> None:
        g = _pickem_game(game_id=1)
        picks = [_pick(PickType.OVER, game_id=1)]
        result = validate_roster(picks, _games_by_id(g))
        self.assertTrue(result.ok)
        self.assertEqual(result.violations, ())

    def test_pickem_via_none_spread_ineligible(self) -> None:
        g = _game(
            spread=None,
            favorite_team_id=None,
            underdog_team_id=None,
            game_id=1,
        )
        picks = [_pick(PickType.FAVORITE_COVER, game_id=1)]
        result = validate_roster(picks, _games_by_id(g))
        self.assertFalse(result.ok)
        self.assertIn(ViolationCode.PICKEM_SPREAD_INELIGIBLE, _codes(result))


class FirstPickPrecedenceTests(unittest.TestCase):
    """check_new_pick: the existing accepted pick wins; the new one is rejected."""

    def test_new_contradiction_rejected(self) -> None:
        g = _game(game_id=1)
        existing = [_pick(PickType.FAVORITE_COVER, game_id=1)]
        new_pick = _pick(PickType.UNDERDOG_COVER, game_id=1)
        result = check_new_pick(new_pick, existing, _games_by_id(g))
        self.assertFalse(result.ok)
        self.assertIn(ViolationCode.CONTRADICTORY_PICK, _codes(result))
        # The violation must carry the NEW pick so the caller knows what to reject.
        offending = [p for v in result.violations for p in v.picks]
        self.assertIn(new_pick, offending)

    def test_new_duplicate_rejected(self) -> None:
        g = _game(game_id=1)
        existing = [_pick(PickType.OVER, game_id=1)]
        new_pick = _pick(PickType.OVER, game_id=1)
        result = check_new_pick(new_pick, existing, _games_by_id(g))
        self.assertFalse(result.ok)
        self.assertIn(ViolationCode.DUPLICATE_PICK, _codes(result))
        offending = [p for v in result.violations for p in v.picks]
        self.assertIn(new_pick, offending)

    def test_new_second_mortal_lock_rejected(self) -> None:
        g1 = _game(game_id=1)
        g2 = _game(game_id=2)
        existing = [_pick(PickType.FAVORITE_COVER, is_mortal_lock=True, game_id=1)]
        new_pick = _pick(PickType.OVER, is_mortal_lock=True, game_id=2)
        result = check_new_pick(new_pick, existing, _games_by_id(g1, g2))
        self.assertFalse(result.ok)
        self.assertIn(ViolationCode.MULTIPLE_MORTAL_LOCKS, _codes(result))

    def test_new_spread_on_pickem_rejected(self) -> None:
        g = _pickem_game(game_id=1)
        result = check_new_pick(
            _pick(PickType.FAVORITE_COVER, game_id=1), [], _games_by_id(g)
        )
        self.assertFalse(result.ok)
        self.assertIn(ViolationCode.PICKEM_SPREAD_INELIGIBLE, _codes(result))

    def test_new_pick_on_different_game_accepted(self) -> None:
        g1 = _game(game_id=1)
        g2 = _game(game_id=2)
        existing = [_pick(PickType.FAVORITE_COVER, game_id=1)]
        new_pick = _pick(PickType.OVER, game_id=2)
        result = check_new_pick(new_pick, existing, _games_by_id(g1, g2))
        self.assertTrue(result.ok)
        self.assertEqual(result.violations, ())

    def test_new_independent_spread_vs_total_same_game_accepted(self) -> None:
        # Existing FAVORITE_COVER; incoming OVER on the same game is independent.
        g = _game(game_id=1)
        existing = [_pick(PickType.FAVORITE_COVER, game_id=1)]
        new_pick = _pick(PickType.OVER, game_id=1)
        result = check_new_pick(new_pick, existing, _games_by_id(g))
        self.assertTrue(result.ok)
        self.assertEqual(result.violations, ())

    def test_new_same_base_type_different_game_accepted(self) -> None:
        # An incoming base OVER on game B while a base OVER on game A exists is a
        # legitimate SLOT REPLACEMENT at submission time (the upsert path in
        # pick_submission), NOT a conflict. The slot-model rejection lives only in
        # the whole-roster validate_roster; check_new_pick stays per-game.
        g1 = _game(game_id=1)
        g2 = _game(game_id=2)
        existing = [_pick(PickType.OVER, game_id=1)]
        new_pick = _pick(PickType.OVER, game_id=2)
        result = check_new_pick(new_pick, existing, _games_by_id(g1, g2))
        self.assertTrue(result.ok)
        self.assertEqual(result.violations, ())

    def test_new_pick_does_not_reflag_existing_conflicts(self) -> None:
        # The existing set itself contains a contradiction, but a clean new pick
        # on a different game is still accepted (existing set is not re-validated).
        g1 = _game(game_id=1)
        g2 = _game(game_id=2)
        existing = [
            _pick(PickType.OVER, game_id=1),
            _pick(PickType.UNDER, game_id=1),
        ]
        new_pick = _pick(PickType.FAVORITE_COVER, game_id=2)
        result = check_new_pick(new_pick, existing, _games_by_id(g1, g2))
        self.assertTrue(result.ok)


if __name__ == "__main__":
    unittest.main()
