"""Offline unit tests for the PURE season-storyline core (260703-jun).

These pin the deterministic storyline computations + selection in
:mod:`app.services.storylines` with hand-built, already-normalized inputs — NO db,
NO fixtures, NO LLM. They assert DETERMINISTIC storyline output only (a streak
detected with the right length + kind, tenure resolving "since week N", a lead flip
setting fresh, a superlative picked + this-week freshness, hot/cold form distinct from
the lock streak, selection capping at ~2-3, and no ``user_id`` anywhere) — never any
non-deterministic LLM text.

Run with: ``backend/.venv/bin/python -m unittest tests.test_storylines -v``
(there is no bare ``python`` on PATH on this machine).
"""

from __future__ import annotations

import dataclasses
import unittest

from app.services.storylines import (
    Storyline,
    SuperlativeCandidate,
    form_streak,
    leader_tenure,
    mortal_lock_streak,
    season_superlative,
    select_storylines,
)


def _has_user_id(storyline: Storyline) -> bool:
    return "user_id" in {f.name for f in dataclasses.fields(storyline)}


class MortalLockStreakTests(unittest.TestCase):
    def test_three_week_missed_streak_detected_with_length_and_kind(self) -> None:
        # weeks 1..5; missed the lock in weeks 3,4,5 (a 3-week miss streak ending at W=5).
        seq = [(1, "hit"), (2, "hit"), (3, "miss"), (4, "miss"), (5, "miss")]
        s = mortal_lock_streak("alice", seq, week=5)
        assert s is not None
        self.assertEqual(s.kind, "mortal_lock_streak")
        self.assertEqual(s.subject, "alice")
        self.assertIn("missed", s.text)
        self.assertIn("3", s.text)
        self.assertTrue(s.fresh)  # extended at W=5
        self.assertFalse(_has_user_id(s))

    def test_hit_streak_kind_and_length(self) -> None:
        seq = [(1, "hit"), (2, "hit"), (3, "hit")]
        s = mortal_lock_streak("bob", seq, week=3)
        assert s is not None
        self.assertIn("hit", s.text)
        self.assertIn("3", s.text)

    def test_none_weeks_are_transparent_not_a_break(self) -> None:
        # A PUSH/UNGRADEABLE week ("none") in the middle neither extends nor breaks.
        seq = [(1, "miss"), (2, "none"), (3, "miss")]
        s = mortal_lock_streak("cara", seq, week=3)
        assert s is not None
        self.assertIn("2", s.text)  # weeks 1 and 3, week 2 skipped
        self.assertTrue(s.fresh)

    def test_short_streak_below_threshold_is_none(self) -> None:
        self.assertIsNone(mortal_lock_streak("d", [(1, "hit"), (2, "miss")], week=2))

    def test_not_fresh_when_latest_graded_week_is_before_W(self) -> None:
        # Missed weeks 1,2 then a transparent "none" at W=3 -> streak did not extend at W.
        seq = [(1, "miss"), (2, "miss"), (3, "none")]
        s = mortal_lock_streak("e", seq, week=3)
        assert s is not None
        self.assertFalse(s.fresh)

    def test_deterministic(self) -> None:
        seq = [(1, "miss"), (2, "miss"), (3, "miss")]
        self.assertEqual(mortal_lock_streak("f", seq, week=3), mortal_lock_streak("f", seq, week=3))


class LeaderTenureTests(unittest.TestCase):
    def test_led_since_week_n(self) -> None:
        seq = [(1, "bob"), (2, "bob"), (3, "alice"), (4, "alice"), (5, "alice")]
        s = leader_tenure(seq, week=5)
        assert s is not None
        self.assertEqual(s.subject, "alice")
        self.assertIn("since week 3", s.text)
        self.assertFalse(s.fresh)  # lead did not flip at W=5

    def test_lead_flip_sets_fresh(self) -> None:
        seq = [(1, "bob"), (2, "bob"), (3, "alice")]
        s = leader_tenure(seq, week=3)
        assert s is not None
        self.assertEqual(s.subject, "alice")
        self.assertTrue(s.fresh)
        self.assertIn("week 3", s.text)

    def test_trivial_single_week_lead_is_none(self) -> None:
        self.assertIsNone(leader_tenure([(1, "alice")], week=1))

    def test_empty_is_none(self) -> None:
        self.assertIsNone(leader_tenure([], week=1))


class FormStreakTests(unittest.TestCase):
    def test_cold_last_three_weeks(self) -> None:
        # (week, base_wins, base_total): 0 wins across the last 3 weeks -> cold.
        recs = [(1, 3, 4), (2, 0, 4), (3, 0, 4), (4, 0, 4)]
        s = form_streak("bob", recs, week=4)
        assert s is not None
        self.assertEqual(s.kind, "form")
        self.assertIn("cold", s.text)
        self.assertTrue(s.fresh)

    def test_hot_perfect_window(self) -> None:
        recs = [(2, 4, 4), (3, 4, 4), (4, 4, 4)]
        s = form_streak("alice", recs, week=4)
        assert s is not None
        self.assertIn("hot", s.text)

    def test_middling_form_is_none(self) -> None:
        recs = [(1, 2, 4), (2, 2, 4), (3, 2, 4)]
        self.assertIsNone(form_streak("c", recs, week=3))

    def test_too_few_weeks_is_none(self) -> None:
        self.assertIsNone(form_streak("d", [(1, 0, 4), (2, 0, 4)], week=2))

    def test_form_is_distinct_from_lock_streak(self) -> None:
        # SAME player: an ice-cold OVERALL slate but a HIT mortal-lock streak — the two
        # families report different, non-duplicated facts.
        recs = [(1, 0, 4), (2, 0, 4), (3, 0, 4)]
        locks = [(1, "hit"), (2, "hit"), (3, "hit")]
        form = form_streak("z", recs, week=3)
        lock = mortal_lock_streak("z", locks, week=3)
        assert form is not None and lock is not None
        self.assertIn("cold", form.text)
        self.assertIn("hit", lock.text)
        self.assertNotEqual(form.kind, lock.kind)


class SeasonSuperlativeTests(unittest.TestCase):
    def test_highest_magnitude_picked_and_this_week_is_fresh(self) -> None:
        cands = [
            SuperlativeCandidate("biggest upset", "DAL stunned PHI in Week 2", 10.0, 2),
            SuperlativeCandidate("highest weekly score", "alice scored 6 in Week 4", 6.0, 4),
            SuperlativeCandidate("biggest upset", "NYJ stunned BUF in Week 4", 21.0, 4),
        ]
        s = season_superlative(cands, week=4)
        assert s is not None
        self.assertTrue(s.league_wide)
        self.assertIsNone(s.subject)
        self.assertIn("NYJ stunned BUF", s.text)  # magnitude 21 wins
        self.assertTrue(s.fresh)  # set in Week 4 == W

    def test_no_candidates_is_none(self) -> None:
        self.assertIsNone(season_superlative([], week=3))

    def test_older_superlative_not_fresh(self) -> None:
        cands = [SuperlativeCandidate("biggest upset", "x in Week 1", 30.0, 1)]
        s = season_superlative(cands, week=5)
        assert s is not None
        self.assertFalse(s.fresh)


class SelectStorylinesTests(unittest.TestCase):
    def _pool(self) -> list[Storyline]:
        return [
            Storyline("mortal_lock_streak", "alice missed 3 running", True, "alice"),
            Storyline("leader_tenure", "alice has led since week 2", False, "alice"),
            Storyline("form", "bob ice cold", True, "bob"),
            Storyline("mortal_lock_streak", "carol missed 2", False, "carol"),  # not featured
            Storyline("superlative", "biggest upset", True, None, league_wide=True),
            Storyline("superlative", "old upset", False, None, league_wide=True),
        ]

    def test_caps_at_max_and_includes_featured_plus_one_superlative(self) -> None:
        selected = select_storylines(self._pool(), featured_players=["alice", "bob"], max_total=3)
        self.assertLessEqual(len(selected), 3)
        # exactly one league superlative
        self.assertEqual(sum(1 for s in selected if s.league_wide), 1)
        # only featured players' non-league storylines
        for s in selected:
            if not s.league_wide:
                self.assertIn(s.subject, {"alice", "bob"})
        # carol (not featured) never appears
        self.assertNotIn("carol", [s.subject for s in selected])

    def test_prefers_fresh_but_falls_back_to_relevant(self) -> None:
        # No fresh storylines at all -> selection still returns a relevant fallback.
        stale = [
            Storyline("leader_tenure", "alice has led since week 2", False, "alice"),
            Storyline("superlative", "old upset", False, None, league_wide=True),
        ]
        selected = select_storylines(stale, featured_players=["alice"], max_total=3)
        self.assertTrue(selected)  # non-empty fallback

    def test_fresh_sorts_first(self) -> None:
        pool = [
            Storyline("leader_tenure", "stale", False, "alice"),
            Storyline("mortal_lock_streak", "fresh", True, "alice"),
        ]
        selected = select_storylines(pool, featured_players=["alice"], max_total=3)
        self.assertEqual(selected[0].text, "fresh")

    def test_deterministic(self) -> None:
        pool = self._pool()
        a = select_storylines(pool, featured_players=["alice", "bob"], max_total=3)
        b = select_storylines(pool, featured_players=["alice", "bob"], max_total=3)
        self.assertEqual(a, b)

    def test_no_user_id_in_any_selected_storyline(self) -> None:
        selected = select_storylines(self._pool(), featured_players=["alice", "bob"])
        for s in selected:
            self.assertFalse(_has_user_id(s))


if __name__ == "__main__":
    unittest.main()
