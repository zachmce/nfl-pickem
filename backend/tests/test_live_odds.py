"""Offline unit tests for the live-odds seam's PURE core (260710-mpw Task 1).

These tests NEVER touch the network. They exercise
:func:`app.services.live_odds.select_live_odds_for_event` — the pure indexer that
reuses ``app.scoreboard.espn.select_odds_item`` + ``normalize_odds`` to turn a
site-scoreboard payload into per-event normalized odds — proving it resolves a real
DraftKings inline shape, indexes by event id, and degrades to ``None`` on a missing
event / non-dict / odds-less payload (never raises).

Run with: ``backend/.venv/bin/python -m unittest tests.test_live_odds -v``
(there is no bare ``python`` on PATH on this machine).
"""

from __future__ import annotations

import unittest

from app.services.live_odds import select_live_odds_for_event


def _payload_one_event() -> dict:
    """A synthetic site scoreboard with one event carrying a DraftKings inline line.

    The shape mirrors the LIVE site path: odds live on ``competitions[0].odds[]`` with a
    ``provider.name``, a signed home-relative ``spread``, ``overUnder``, and inline
    ``home/awayTeamOdds`` favorite/underdog flags (see espn.py ``normalize_odds``).
    """
    return {
        "events": [
            {
                "id": "401547001",
                "competitions": [
                    {
                        "odds": [
                            {
                                "provider": {"name": "DraftKings", "priority": 1},
                                "spread": -3.5,
                                "overUnder": 47.5,
                                "awayTeamOdds": {"team": {"id": "1"}, "underdog": True},
                                "homeTeamOdds": {"team": {"id": "2"}, "favorite": True},
                            }
                        ]
                    }
                ],
            }
        ]
    }


class SelectLiveOddsTests(unittest.TestCase):
    def test_resolves_matching_event_via_normalize_odds(self) -> None:
        odds = select_live_odds_for_event(_payload_one_event(), 401547001)
        self.assertIsNotNone(odds)
        assert odds is not None  # narrow for the type checker
        self.assertEqual(odds.provider, "DraftKings")
        self.assertEqual(odds.spread, -3.5)
        self.assertEqual(odds.total, 47.5)
        self.assertEqual(odds.favorite_team_id, "2")
        self.assertEqual(odds.underdog_team_id, "1")

    def test_missing_event_returns_none(self) -> None:
        self.assertIsNone(select_live_odds_for_event(_payload_one_event(), 999999))

    def test_non_dict_payload_degrades_to_none(self) -> None:
        for bad in (None, "nope", 42, []):
            self.assertIsNone(select_live_odds_for_event(bad, 401547001))

    def test_oddsless_event_degrades_to_none(self) -> None:
        payload = {"events": [{"id": "5", "competitions": [{}]}]}
        self.assertIsNone(select_live_odds_for_event(payload, 5))

    def test_event_id_is_matched_as_string(self) -> None:
        # Our DB event id is an int; ESPN reports ids as strings — the lookup coerces.
        odds = select_live_odds_for_event(_payload_one_event(), 401547001)
        self.assertIsNotNone(odds)


if __name__ == "__main__":
    unittest.main()
