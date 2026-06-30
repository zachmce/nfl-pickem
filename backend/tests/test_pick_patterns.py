"""Offline unit tests for the pure pick-pattern scanner (260627-nef).

These tests exercise :func:`app.services.pick_patterns.scan_streak` with plain
dicts only — NO database, NO ORM, NO network, NO discord, NO clock. The scanner
is the deterministic FACT owner of the pickem-chat personality layer: it detects
a same-``(team_abbr, side)`` streak of >= 3 consecutive weeks ending at the
target week, where a totals pick is keyed on a REAL team (so a streak survives a
changing opponent).

Run with: ``backend/.venv/bin/python -m unittest tests.test_pick_patterns -v``
(there is no bare ``python`` on PATH on this machine).
"""

from __future__ import annotations

import unittest

from app.services.pick_patterns import scan_streak


def _key(week: int, team_abbr: str, side: str) -> dict:
    """Build one plain pick-key dict the scanner consumes."""
    return {"week": week, "team_abbr": team_abbr, "side": side}


class ScanStreakTests(unittest.TestCase):
    def test_totals_team_keyed_across_changing_opponents(self) -> None:
        """REGRESSION GUARD: OVER on KC@DEN (wk5), KC@LV (wk4), KC@SF (wk3).

        Each totals pick is expanded into its two team keys. KC forms a 3-week
        OVER streak even though the opponent (DEN/LV/SF) changes every week —
        those opponents do NOT form their own 3-week run.
        """
        history = [
            _key(5, "KC", "OVER"),
            _key(5, "DEN", "OVER"),
            _key(4, "KC", "OVER"),
            _key(4, "LV", "OVER"),
            _key(3, "KC", "OVER"),
            _key(3, "SF", "OVER"),
        ]
        slate = [_key(5, "KC", "OVER"), _key(5, "DEN", "OVER")]
        fact = scan_streak(5, slate, history)
        self.assertEqual(fact, {"team_abbr": "KC", "side": "OVER", "streak_weeks": 3})

    def test_spread_streak_same_team(self) -> None:
        history = [
            _key(5, "KC", "FAVORITE"),
            _key(4, "KC", "FAVORITE"),
            _key(3, "KC", "FAVORITE"),
        ]
        slate = [_key(5, "KC", "FAVORITE")]
        fact = scan_streak(5, slate, history)
        self.assertEqual(fact, {"team_abbr": "KC", "side": "FAVORITE", "streak_weeks": 3})

    def test_streak_of_two_is_below_threshold(self) -> None:
        history = [_key(5, "KC", "FAVORITE"), _key(4, "KC", "FAVORITE")]
        slate = [_key(5, "KC", "FAVORITE")]
        self.assertIsNone(scan_streak(5, slate, history))

    def test_gap_breaks_the_run(self) -> None:
        """weeks 5, 4, 2 — the missing week 3 breaks the consecutive run."""
        history = [
            _key(5, "KC", "FAVORITE"),
            _key(4, "KC", "FAVORITE"),
            _key(2, "KC", "FAVORITE"),
        ]
        slate = [_key(5, "KC", "FAVORITE")]
        self.assertIsNone(scan_streak(5, slate, history))

    def test_same_team_different_side_does_not_streak(self) -> None:
        history = [
            _key(5, "KC", "FAVORITE"),
            _key(4, "KC", "UNDERDOG"),
            _key(3, "KC", "FAVORITE"),
        ]
        slate = [_key(5, "KC", "FAVORITE")]
        self.assertIsNone(scan_streak(5, slate, history))

    def test_empty_history_and_slate(self) -> None:
        self.assertIsNone(scan_streak(5, [], []))
        self.assertIsNone(scan_streak(5, [_key(5, "KC", "OVER")], []))

    def test_streak_must_end_at_target_week(self) -> None:
        """A 3-run ending at wk4 (not the target wk5) does not fire."""
        history = [
            _key(4, "KC", "FAVORITE"),
            _key(3, "KC", "FAVORITE"),
            _key(2, "KC", "FAVORITE"),
        ]
        slate = [_key(5, "SF", "OVER")]
        self.assertIsNone(scan_streak(5, slate, history))

    def test_two_qualifying_streaks_returns_longer(self) -> None:
        """KC FAVORITE runs 4 weeks; SF UNDER runs 3 — the longer (KC) wins."""
        history = [
            _key(5, "KC", "FAVORITE"),
            _key(5, "SF", "UNDER"),
            _key(4, "KC", "FAVORITE"),
            _key(4, "SF", "UNDER"),
            _key(3, "KC", "FAVORITE"),
            _key(3, "SF", "UNDER"),
            _key(2, "KC", "FAVORITE"),
        ]
        slate = [_key(5, "KC", "FAVORITE"), _key(5, "SF", "UNDER")]
        fact = scan_streak(5, slate, history)
        self.assertEqual(fact, {"team_abbr": "KC", "side": "FAVORITE", "streak_weeks": 4})

    def test_equal_length_streaks_tie_break_alphabetical(self) -> None:
        """Two 3-week streaks of equal length -> alphabetical team_abbr (KC<SF)."""
        history = [
            _key(5, "SF", "UNDER"),
            _key(5, "KC", "FAVORITE"),
            _key(4, "SF", "UNDER"),
            _key(4, "KC", "FAVORITE"),
            _key(3, "SF", "UNDER"),
            _key(3, "KC", "FAVORITE"),
        ]
        slate = [_key(5, "SF", "UNDER"), _key(5, "KC", "FAVORITE")]
        fact = scan_streak(5, slate, history)
        self.assertEqual(fact, {"team_abbr": "KC", "side": "FAVORITE", "streak_weeks": 3})

    def test_streak_key_must_be_present_in_target_slate(self) -> None:
        """A run in history that the player did NOT re-pick this week is ignored.

        The slate gates which keys are even considered — the streak must include
        the target week (which comes from the slate), so a key absent from the
        slate cannot end at the target week.
        """
        history = [
            _key(4, "KC", "FAVORITE"),
            _key(3, "KC", "FAVORITE"),
            _key(2, "KC", "FAVORITE"),
        ]
        # slate has a DIFFERENT key at the target week
        slate = [_key(5, "DEN", "OVER")]
        self.assertIsNone(scan_streak(5, slate, history))


if __name__ == "__main__":
    unittest.main()
