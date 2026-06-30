"""Offline unit tests for the pure pick'em scoring engine.

These tests exercise :mod:`app.services.scoring` with hand-built ``Game`` /
``Pick`` model instances (no database needed — they are plain SQLModel objects),
covering every grading branch and the weekly-total roll-up. One additional
*ground-truth* test imports the real 2025 season into an in-memory SQLite db and
grades a real FINAL game against its frozen line, proving the engine works on
real scores fully offline.

Everything runs offline:

* the synthetic tests touch no database at all,
* the ground-truth test uses an in-memory SQLite engine
  (``create_engine("sqlite://")``) and never imports the Postgres engine module,
* there is no network access of any kind.

Run from the ``backend/`` directory with the standard library test runner::

    cd backend && python -m unittest tests.test_scoring -v

> Note: on this machine the interpreter is ``python3`` (there is no bare
> ``python`` on ``PATH``); use ``python3 -m unittest ...`` or the venv
> interpreter ``.venv/bin/python -m unittest ...``.

No pytest dependency is required (none is configured for this project).
"""

from __future__ import annotations

import unittest
from decimal import Decimal

from sqlmodel import Session, SQLModel, create_engine, select

from app.models import Game, GameStatus, Pick, PickResult, PickType
from app.seeds.fixture_2025 import import_fixture_2025
from app.seeds.teams import seed_teams
from app.services.scoring import (
    GradeOutcome,
    GradeResult,
    grade_pick,
    score_week,
)

# Small constant team ids for the synthetic games. The engine only compares ids;
# it never loads Team rows, so these need not exist in any database.
HOME = 1
AWAY = 2


def _game(
    *,
    home_score: int | None = None,
    away_score: int | None = None,
    status: GameStatus = GameStatus.FINAL,
    spread: Decimal | None = None,
    total: Decimal | None = None,
    favorite_team_id: int | None = None,
    underdog_team_id: int | None = None,
    game_id: int = 100,
) -> Game:
    """Build a synthetic ``Game`` instance with just the fields the engine reads."""
    return Game(
        id=game_id,
        espn_event_id=game_id,
        week_id=1,
        season=2025,
        week=1,
        home_team_id=HOME,
        away_team_id=AWAY,
        status=status,
        home_score=home_score,
        away_score=away_score,
        spread=spread,
        total=total,
        favorite_team_id=favorite_team_id,
        underdog_team_id=underdog_team_id,
    )


def _pick(
    pick_type: PickType,
    *,
    is_mortal_lock: bool = False,
    game_id: int = 100,
) -> Pick:
    """Build a synthetic ``Pick`` instance for the engine."""
    return Pick(
        user_id=1,
        game_id=game_id,
        week_id=1,
        pick_type=pick_type,
        is_mortal_lock=is_mortal_lock,
    )


class GradeSpreadTests(unittest.TestCase):
    """FAVORITE_COVER / UNDERDOG_COVER grading against the spread."""

    def test_favorite_cover_win(self) -> None:
        # Favorite (home) wins by 10 vs a 3.5 line -> covered.
        game = _game(
            home_score=30,
            away_score=20,
            spread=Decimal("3.5"),
            favorite_team_id=HOME,
            underdog_team_id=AWAY,
        )
        res = grade_pick(game, _pick(PickType.FAVORITE_COVER))
        self.assertEqual(res, GradeResult(GradeOutcome.WIN, 1))

    def test_favorite_cover_loss(self) -> None:
        # Favorite wins by only 2 vs a 3.5 line -> did not cover.
        game = _game(
            home_score=22,
            away_score=20,
            spread=Decimal("3.5"),
            favorite_team_id=HOME,
            underdog_team_id=AWAY,
        )
        res = grade_pick(game, _pick(PickType.FAVORITE_COVER))
        self.assertEqual(res, GradeResult(GradeOutcome.LOSS, 0))

    def test_underdog_cover_win(self) -> None:
        # Favorite wins by only 2 vs a 3.5 line -> underdog covers.
        game = _game(
            home_score=22,
            away_score=20,
            spread=Decimal("3.5"),
            favorite_team_id=HOME,
            underdog_team_id=AWAY,
        )
        res = grade_pick(game, _pick(PickType.UNDERDOG_COVER))
        self.assertEqual(res, GradeResult(GradeOutcome.WIN, 1))

    def test_underdog_cover_loss(self) -> None:
        # Favorite covers comfortably -> underdog pick loses.
        game = _game(
            home_score=30,
            away_score=20,
            spread=Decimal("3.5"),
            favorite_team_id=HOME,
            underdog_team_id=AWAY,
        )
        res = grade_pick(game, _pick(PickType.UNDERDOG_COVER))
        self.assertEqual(res, GradeResult(GradeOutcome.LOSS, 0))

    def test_favorite_is_away_team(self) -> None:
        # Direction must follow favorite_team_id, not home/away. Away favored by
        # 7 line; away wins by 10 -> covers.
        game = _game(
            home_score=14,
            away_score=24,
            spread=Decimal("7.0"),
            favorite_team_id=AWAY,
            underdog_team_id=HOME,
        )
        # Favorite margin = 24 - 14 = 10 > 7 -> favorite covers.
        self.assertEqual(
            grade_pick(game, _pick(PickType.FAVORITE_COVER)).outcome,
            GradeOutcome.WIN,
        )
        self.assertEqual(
            grade_pick(game, _pick(PickType.UNDERDOG_COVER)).outcome,
            GradeOutcome.LOSS,
        )

    def test_spread_exact_push(self) -> None:
        # Favorite wins by exactly 3 vs a 3.0 line -> PUSH, zero points.
        game = _game(
            home_score=23,
            away_score=20,
            spread=Decimal("3.0"),
            favorite_team_id=HOME,
            underdog_team_id=AWAY,
        )
        self.assertEqual(
            grade_pick(game, _pick(PickType.FAVORITE_COVER)),
            GradeResult(GradeOutcome.PUSH, 0),
        )
        self.assertEqual(
            grade_pick(game, _pick(PickType.UNDERDOG_COVER)),
            GradeResult(GradeOutcome.PUSH, 0),
        )

    def test_spread_push_zero_even_for_mortal_lock(self) -> None:
        game = _game(
            home_score=23,
            away_score=20,
            spread=Decimal("3.0"),
            favorite_team_id=HOME,
            underdog_team_id=AWAY,
        )
        res = grade_pick(game, _pick(PickType.FAVORITE_COVER, is_mortal_lock=True))
        self.assertEqual(res, GradeResult(GradeOutcome.PUSH, 0))


class GradeTotalTests(unittest.TestCase):
    """OVER / UNDER grading against the total."""

    def test_over_win(self) -> None:
        game = _game(home_score=30, away_score=21, total=Decimal("44.5"))
        self.assertEqual(
            grade_pick(game, _pick(PickType.OVER)),
            GradeResult(GradeOutcome.WIN, 1),
        )

    def test_over_loss(self) -> None:
        game = _game(home_score=10, away_score=13, total=Decimal("44.5"))
        self.assertEqual(
            grade_pick(game, _pick(PickType.OVER)),
            GradeResult(GradeOutcome.LOSS, 0),
        )

    def test_under_win(self) -> None:
        game = _game(home_score=10, away_score=13, total=Decimal("44.5"))
        self.assertEqual(
            grade_pick(game, _pick(PickType.UNDER)),
            GradeResult(GradeOutcome.WIN, 1),
        )

    def test_under_loss(self) -> None:
        game = _game(home_score=30, away_score=21, total=Decimal("44.5"))
        self.assertEqual(
            grade_pick(game, _pick(PickType.UNDER)),
            GradeResult(GradeOutcome.LOSS, 0),
        )

    def test_total_exact_push(self) -> None:
        # Combined exactly 44 vs a 44.0 total -> PUSH, zero points.
        game = _game(home_score=24, away_score=20, total=Decimal("44.0"))
        self.assertEqual(
            grade_pick(game, _pick(PickType.OVER)),
            GradeResult(GradeOutcome.PUSH, 0),
        )
        self.assertEqual(
            grade_pick(game, _pick(PickType.UNDER)),
            GradeResult(GradeOutcome.PUSH, 0),
        )


class MortalLockTests(unittest.TestCase):
    """Mortal lock scoring: +2 on a win, -1 on a loss."""

    def test_mortal_lock_win_scores_plus_two(self) -> None:
        game = _game(
            home_score=30,
            away_score=20,
            spread=Decimal("3.5"),
            favorite_team_id=HOME,
            underdog_team_id=AWAY,
        )
        res = grade_pick(game, _pick(PickType.FAVORITE_COVER, is_mortal_lock=True))
        self.assertEqual(res, GradeResult(GradeOutcome.WIN, 2))

    def test_mortal_lock_loss_scores_minus_one(self) -> None:
        game = _game(
            home_score=22,
            away_score=20,
            spread=Decimal("3.5"),
            favorite_team_id=HOME,
            underdog_team_id=AWAY,
        )
        res = grade_pick(game, _pick(PickType.FAVORITE_COVER, is_mortal_lock=True))
        self.assertEqual(res, GradeResult(GradeOutcome.LOSS, -1))


class PickemAndTotalIneligibilityTests(unittest.TestCase):
    """True pick'em: spread picks ineligible, Over/Under still grade."""

    def test_pickem_spread_zero_makes_spread_picks_ineligible(self) -> None:
        game = _game(
            home_score=24,
            away_score=20,
            spread=Decimal("0"),
            favorite_team_id=None,
            underdog_team_id=None,
            total=Decimal("44.5"),
        )
        self.assertEqual(
            grade_pick(game, _pick(PickType.FAVORITE_COVER)),
            GradeResult(GradeOutcome.INELIGIBLE, 0),
        )
        self.assertEqual(
            grade_pick(game, _pick(PickType.UNDERDOG_COVER)),
            GradeResult(GradeOutcome.INELIGIBLE, 0),
        )

    def test_pickem_none_spread_makes_spread_picks_ineligible(self) -> None:
        game = _game(
            home_score=24,
            away_score=20,
            spread=None,
            favorite_team_id=None,
            underdog_team_id=None,
            total=Decimal("44.5"),
        )
        self.assertEqual(
            grade_pick(game, _pick(PickType.UNDERDOG_COVER)).outcome,
            GradeOutcome.INELIGIBLE,
        )

    def test_pickem_over_under_still_grade(self) -> None:
        # combined 44 < 44.5 -> UNDER wins, OVER loses, even on a pick'em.
        game = _game(
            home_score=24,
            away_score=20,
            spread=Decimal("0"),
            favorite_team_id=None,
            underdog_team_id=None,
            total=Decimal("44.5"),
        )
        self.assertEqual(grade_pick(game, _pick(PickType.UNDER)).outcome, GradeOutcome.WIN)
        self.assertEqual(grade_pick(game, _pick(PickType.OVER)).outcome, GradeOutcome.LOSS)

    def test_total_none_makes_over_under_ineligible(self) -> None:
        # No total posted at the frozen line -> OVER/UNDER void to 0 (the
        # _total_outcome None branch). Load-bearing for the odds line-at-lock
        # policy: a total that never posted (or vanished before freeze) must not
        # grade as a loss. See .planning/notes/scheduled-tasks-and-odds-freeze.md.
        game = _game(
            home_score=24,
            away_score=20,
            total=None,
            spread=Decimal("3.5"),
            favorite_team_id=HOME,
            underdog_team_id=AWAY,
        )
        self.assertEqual(
            grade_pick(game, _pick(PickType.OVER)),
            GradeResult(GradeOutcome.INELIGIBLE, 0),
        )
        self.assertEqual(
            grade_pick(game, _pick(PickType.UNDER)),
            GradeResult(GradeOutcome.INELIGIBLE, 0),
        )

    def test_ineligible_zero_even_for_mortal_lock(self) -> None:
        # An ineligible pick voids to 0 even as a mortal lock -- never -1. This
        # is the line-at-lock guarantee: if a pick's type is ineligible at the
        # frozen line (here a pick'em spread), it is a no-action void, not a
        # loss. See .planning/notes/scheduled-tasks-and-odds-freeze.md.
        game = _game(
            home_score=24,
            away_score=20,
            spread=Decimal("0"),
            favorite_team_id=None,
            underdog_team_id=None,
            total=Decimal("44.5"),
        )
        self.assertEqual(
            grade_pick(game, _pick(PickType.FAVORITE_COVER, is_mortal_lock=True)),
            GradeResult(GradeOutcome.INELIGIBLE, 0),
        )


class UngradeableTests(unittest.TestCase):
    """Non-final or missing-score games are ungradeable, never wrong."""

    def test_scheduled_game_is_ungradeable(self) -> None:
        game = _game(
            home_score=None,
            away_score=None,
            status=GameStatus.SCHEDULED,
            spread=Decimal("3.5"),
            favorite_team_id=HOME,
            underdog_team_id=AWAY,
            total=Decimal("44.5"),
        )
        self.assertEqual(
            grade_pick(game, _pick(PickType.FAVORITE_COVER)),
            GradeResult(GradeOutcome.UNGRADEABLE, 0),
        )

    def test_final_game_with_missing_score_is_ungradeable(self) -> None:
        game = _game(home_score=24, away_score=None, status=GameStatus.FINAL, total=Decimal("44.5"))
        self.assertEqual(
            grade_pick(game, _pick(PickType.OVER)),
            GradeResult(GradeOutcome.UNGRADEABLE, 0),
        )

    def test_ungradeable_zero_even_for_mortal_lock(self) -> None:
        game = _game(
            home_score=None, away_score=None, status=GameStatus.IN_PROGRESS, total=Decimal("44.5")
        )
        res = grade_pick(game, _pick(PickType.OVER, is_mortal_lock=True))
        self.assertEqual(res, GradeResult(GradeOutcome.UNGRADEABLE, 0))


class ScoreWeekTests(unittest.TestCase):
    """Weekly roll-up bounded in [-1, 6]."""

    def test_max_win_week_totals_six(self) -> None:
        # 4 correct base picks (+1 each) on 4 distinct games + a correct mortal
        # lock (+2) = 6, the maximum.
        g_fav = _game(
            game_id=1,
            home_score=30,
            away_score=20,
            spread=Decimal("3.5"),
            favorite_team_id=HOME,
            underdog_team_id=AWAY,
        )
        g_dog = _game(
            game_id=2,
            home_score=22,
            away_score=20,
            spread=Decimal("3.5"),
            favorite_team_id=HOME,
            underdog_team_id=AWAY,
        )
        g_over = _game(game_id=3, home_score=30, away_score=21, total=Decimal("44.5"))
        g_under = _game(game_id=4, home_score=10, away_score=13, total=Decimal("44.5"))
        g_lock = _game(
            game_id=5,
            home_score=30,
            away_score=20,
            spread=Decimal("3.5"),
            favorite_team_id=HOME,
            underdog_team_id=AWAY,
        )
        games_by_id = {g.id: g for g in (g_fav, g_dog, g_over, g_under, g_lock)}
        picks = [
            _pick(PickType.FAVORITE_COVER, game_id=1),
            _pick(PickType.UNDERDOG_COVER, game_id=2),
            _pick(PickType.OVER, game_id=3),
            _pick(PickType.UNDER, game_id=4),
            _pick(PickType.FAVORITE_COVER, is_mortal_lock=True, game_id=5),
        ]
        self.assertEqual(score_week(games_by_id, picks), 6)

    def test_lone_losing_mortal_lock_totals_minus_one(self) -> None:
        g = _game(
            game_id=5,
            home_score=22,
            away_score=20,
            spread=Decimal("3.5"),
            favorite_team_id=HOME,
            underdog_team_id=AWAY,
        )
        picks = [_pick(PickType.FAVORITE_COVER, is_mortal_lock=True, game_id=5)]
        self.assertEqual(score_week({g.id: g}, picks), -1)

    def test_partial_week_sums_only_present_picks(self) -> None:
        # Only 2 correct base picks present -> 2; absent slots contribute nothing.
        g1 = _game(
            game_id=1,
            home_score=30,
            away_score=20,
            spread=Decimal("3.5"),
            favorite_team_id=HOME,
            underdog_team_id=AWAY,
        )
        g2 = _game(game_id=3, home_score=30, away_score=21, total=Decimal("44.5"))
        picks = [
            _pick(PickType.FAVORITE_COVER, game_id=1),
            _pick(PickType.OVER, game_id=3),
        ]
        self.assertEqual(score_week({g1.id: g1, g2.id: g2}, picks), 2)

    def test_empty_week_totals_zero(self) -> None:
        self.assertEqual(score_week({}, []), 0)


def _misc_pick(
    *,
    result: PickResult,
    points: int,
    game_id: int = 100,
    is_mortal_lock: bool = False,
) -> Pick:
    """Build a synthetic MISC ``Pick`` carrying an admin-set result/points."""
    return Pick(
        user_id=1,
        game_id=game_id,
        week_id=1,
        pick_type=PickType.MISC,
        is_mortal_lock=is_mortal_lock,
        misc_text="Mahomes throws for 400 yards",
        result=result,
        points=points,
    )


class GradeMiscPassthroughTests(unittest.TestCase):
    """MISC is the ONE type whose stored result/points pass through verbatim.

    The game is irrelevant to a MISC grade: these tests deliberately hand the
    engine a game that would make the OLD (spread/total) path return
    INELIGIBLE/0 (a true pick'em with no total) to prove MISC never routes there.
    """

    def _ungradeable_misc_game(self) -> Game:
        # A true pick'em with NO total: every auto-graded branch would void to 0
        # here, so a non-zero MISC result proves the passthrough fired instead.
        return _game(
            home_score=21,
            away_score=20,
            status=GameStatus.FINAL,
            spread=Decimal("0"),
            favorite_team_id=None,
            underdog_team_id=None,
            total=None,
        )

    def test_misc_win_passes_stored_points_through(self) -> None:
        game = self._ungradeable_misc_game()
        res = grade_pick(game, _misc_pick(result=PickResult.WIN, points=3))
        self.assertEqual(res, GradeResult(GradeOutcome.WIN, 3))

    def test_misc_win_independent_of_game_score_and_status(self) -> None:
        # Even on a SCHEDULED game (no score yet) the admin grade still flows.
        game = _game(
            home_score=None,
            away_score=None,
            status=GameStatus.SCHEDULED,
            total=None,
            spread=Decimal("0"),
        )
        res = grade_pick(game, _misc_pick(result=PickResult.WIN, points=5))
        self.assertEqual(res, GradeResult(GradeOutcome.WIN, 5))

    def test_misc_loss_zero_points(self) -> None:
        game = self._ungradeable_misc_game()
        res = grade_pick(game, _misc_pick(result=PickResult.LOSS, points=0))
        self.assertEqual(res, GradeResult(GradeOutcome.LOSS, 0))

    def test_misc_loss_negative_penalty_passes_through(self) -> None:
        # A negative admin-set penalty passes through verbatim (no implicit -1).
        game = self._ungradeable_misc_game()
        res = grade_pick(game, _misc_pick(result=PickResult.LOSS, points=-2))
        self.assertEqual(res, GradeResult(GradeOutcome.LOSS, -2))

    def test_misc_pending_is_ungradeable_zero(self) -> None:
        game = self._ungradeable_misc_game()
        res = grade_pick(game, _misc_pick(result=PickResult.PENDING, points=0))
        self.assertEqual(res, GradeResult(GradeOutcome.UNGRADEABLE, 0))

    def test_misc_points_can_exceed_weekly_band(self) -> None:
        # MISC is intentionally not bounded by the [-1, 6] band of auto types.
        game = self._ungradeable_misc_game()
        res = grade_pick(game, _misc_pick(result=PickResult.WIN, points=10))
        self.assertEqual(res, GradeResult(GradeOutcome.WIN, 10))

    def test_score_week_sums_graded_misc_on_top_of_regular_picks(self) -> None:
        # Two correct base picks (+1 each) plus a graded MISC (+3) = 5. A
        # recompute over the graded MISC keeps the admin points (not 0): this is
        # the "auto-grade never overwrites the stored MISC grade" guarantee.
        g_fav = _game(
            game_id=1,
            home_score=30,
            away_score=20,
            spread=Decimal("3.5"),
            favorite_team_id=HOME,
            underdog_team_id=AWAY,
        )
        g_over = _game(game_id=3, home_score=30, away_score=21, total=Decimal("44.5"))
        g_misc = self._ungradeable_misc_game()
        g_misc.id = 7
        games_by_id = {1: g_fav, 3: g_over, 7: g_misc}
        picks = [
            _pick(PickType.FAVORITE_COVER, game_id=1),
            _pick(PickType.OVER, game_id=3),
            _misc_pick(result=PickResult.WIN, points=3, game_id=7),
        ]
        self.assertEqual(score_week(games_by_id, picks), 5)

    def test_score_week_recompute_keeps_graded_misc_not_zero(self) -> None:
        # A lone graded MISC scored on its own: the recompute yields the admin's
        # points, proving the stored grade is authoritative for MISC.
        g_misc = self._ungradeable_misc_game()
        g_misc.id = 7
        picks = [_misc_pick(result=PickResult.WIN, points=4, game_id=7)]
        self.assertEqual(score_week({7: g_misc}, picks), 4)


class GroundTruthRealSeasonTests(unittest.TestCase):
    """Grade a real 2025 FINAL game against its frozen line, fully offline."""

    def setUp(self) -> None:
        self.engine = create_engine("sqlite://")
        SQLModel.metadata.create_all(self.engine)

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_grades_real_final_game(self) -> None:
        with Session(self.engine) as session:
            seed_teams(session)
            import_fixture_2025(session)

            # Pick a real FINAL game with a frozen line and a fractional
            # spread+total (so it cannot land on a push) — robust to fixture
            # ordering, no hardcoded event id required.
            game = None
            for g in session.exec(select(Game)).all():
                if (
                    g.status is GameStatus.FINAL
                    and g.odds_frozen
                    and g.spread is not None
                    and g.total is not None
                    and g.home_score is not None
                    and g.away_score is not None
                    and g.spread % 1 != 0
                    and g.total % 1 != 0
                ):
                    game = g
                    break
            self.assertIsNotNone(game, "no suitable real FINAL game found")

            # --- Hand-compute the expected outcomes from the real fields. ---
            if game.favorite_team_id == game.home_team_id:
                fav_score, dog_score = game.home_score, game.away_score
            else:
                fav_score, dog_score = game.away_score, game.home_score
            favorite_margin = Decimal(fav_score - dog_score)
            favorite_covered = favorite_margin > game.spread

            combined = Decimal(game.home_score + game.away_score)
            went_over = combined > game.total

            # Favorite cover.
            fav_pick = _pick(PickType.FAVORITE_COVER, game_id=game.id)
            expected_fav = GradeOutcome.WIN if favorite_covered else GradeOutcome.LOSS
            self.assertEqual(grade_pick(game, fav_pick).outcome, expected_fav)

            # Underdog cover is the mirror image.
            dog_pick = _pick(PickType.UNDERDOG_COVER, game_id=game.id)
            expected_dog = GradeOutcome.LOSS if favorite_covered else GradeOutcome.WIN
            self.assertEqual(grade_pick(game, dog_pick).outcome, expected_dog)

            # Over / Under.
            over_pick = _pick(PickType.OVER, game_id=game.id)
            expected_over = GradeOutcome.WIN if went_over else GradeOutcome.LOSS
            self.assertEqual(grade_pick(game, over_pick).outcome, expected_over)

            under_pick = _pick(PickType.UNDER, game_id=game.id)
            expected_under = GradeOutcome.LOSS if went_over else GradeOutcome.WIN
            self.assertEqual(grade_pick(game, under_pick).outcome, expected_under)

            # And the winning base pick is worth exactly +1.
            winning_total_pick = over_pick if went_over else under_pick
            self.assertEqual(grade_pick(game, winning_total_pick).points, 1)


if __name__ == "__main__":
    unittest.main()
