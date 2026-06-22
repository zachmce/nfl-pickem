"""Offline unit tests for the pure normalization functions of the 2025 fixture
generator.

These tests never touch the network. They exercise only the pure functions
(``select_odds_item``, ``normalize_odds``, ``parse_team_id_from_ref``,
``normalize_game``) against small inline JSON literals modeled on the real ESPN
shapes documented in ``.planning/notes/espn-ingestion-strategy.md``.

Run from the ``backend/`` directory with the standard library test runner::

    cd backend && python -m unittest tests.test_gen_2025_fixture -v

No pytest dependency is required (none is configured for this project).
"""

from __future__ import annotations

import unittest

from scripts.gen_2025_fixture import (
    normalize_game,
    normalize_odds,
    parse_team_id_from_ref,
    select_odds_item,
)


class SelectOddsItemTests(unittest.TestCase):
    def test_selects_item_by_provider_name(self) -> None:
        response = {
            "items": [
                {"provider": {"id": "999", "name": "SomeBook"}, "spread": -1.0},
                {"provider": {"id": "58", "name": "ESPN BET"}, "spread": -3.5},
            ]
        }
        item = select_odds_item(response)
        self.assertIsNotNone(item)
        assert item is not None  # type narrowing
        self.assertEqual(item["provider"]["name"], "ESPN BET")
        self.assertEqual(item["spread"], -3.5)

    def test_falls_back_to_provider_id_58_when_name_differs(self) -> None:
        # Only item carries provider.id 58 but a non-matching name.
        response = {
            "items": [
                {"provider": {"id": "58", "name": "ESPN BET Sportsbook"}, "spread": -7.0},
            ]
        }
        item = select_odds_item(response)
        self.assertIsNotNone(item)
        assert item is not None
        self.assertEqual(item["spread"], -7.0)

    def test_falls_back_to_integer_provider_id_58(self) -> None:
        response = {"items": [{"provider": {"id": 58, "name": "Other"}, "spread": -2.5}]}
        item = select_odds_item(response)
        self.assertIsNotNone(item)
        assert item is not None
        self.assertEqual(item["spread"], -2.5)

    def test_name_match_wins_over_id_fallback(self) -> None:
        response = {
            "items": [
                {"provider": {"id": "58", "name": "Other"}, "spread": -1.0},
                {"provider": {"id": "59", "name": "ESPN BET"}, "spread": -9.0},
            ]
        }
        item = select_odds_item(response)
        assert item is not None
        self.assertEqual(item["spread"], -9.0)

    def test_empty_items_returns_none(self) -> None:
        self.assertIsNone(select_odds_item({"items": []}))

    def test_no_matching_provider_returns_none(self) -> None:
        response = {"items": [{"provider": {"id": "100", "name": "DraftKings"}}]}
        self.assertIsNone(select_odds_item(response))

    def test_malformed_response_returns_none(self) -> None:
        self.assertIsNone(select_odds_item(None))
        self.assertIsNone(select_odds_item({}))
        self.assertIsNone(select_odds_item({"items": "nope"}))


class NormalizeOddsTests(unittest.TestCase):
    def _full_item(self) -> dict:
        # Models a core-API ESPN BET item with team $ref URLs.
        return {
            "provider": {"id": "58", "name": "ESPN BET"},
            "spread": -3.5,
            "overUnder": 44.5,
            "awayTeamOdds": {
                "favorite": False,
                "underdog": True,
                "team": {"$ref": "http://x/teams/17?lang=en&region=us"},
            },
            "homeTeamOdds": {
                "favorite": True,
                "underdog": False,
                "team": {"$ref": "http://x/teams/26?lang=en&region=us"},
            },
        }

    def test_full_item_yields_complete_odds(self) -> None:
        odds = normalize_odds(self._full_item())
        assert odds is not None
        self.assertEqual(odds["provider"], "ESPN BET")
        self.assertEqual(odds["spread"], -3.5)
        self.assertEqual(odds["total"], 44.5)
        self.assertEqual(odds["favorite_team_id"], "26")
        self.assertEqual(odds["underdog_team_id"], "17")

    def test_prefers_close_line_over_top_level(self) -> None:
        item = self._full_item()
        item["close"] = {"spread": -2.0, "total": 41.0}
        odds = normalize_odds(item)
        assert odds is not None
        self.assertEqual(odds["spread"], -2.0)
        self.assertEqual(odds["total"], 41.0)

    def test_close_overunder_used_when_close_total_absent(self) -> None:
        item = self._full_item()
        item["close"] = {"spread": -2.0, "overUnder": 47.0}
        odds = normalize_odds(item)
        assert odds is not None
        self.assertEqual(odds["spread"], -2.0)
        self.assertEqual(odds["total"], 47.0)

    def test_degrades_to_nulls_without_subobjects(self) -> None:
        # Missing spread/overUnder and team odds objects must not raise.
        odds = normalize_odds({"provider": {"id": "58", "name": "ESPN BET"}})
        assert odds is not None
        self.assertEqual(odds["provider"], "ESPN BET")
        self.assertIsNone(odds["spread"])
        self.assertIsNone(odds["total"])
        self.assertIsNone(odds["favorite_team_id"])
        self.assertIsNone(odds["underdog_team_id"])

    def test_inline_team_id_supported(self) -> None:
        item = {
            "spread": -1.5,
            "overUnder": 40.0,
            "awayTeamOdds": {"underdog": True, "team": {"id": "5"}},
            "homeTeamOdds": {"favorite": True, "team": {"id": "9"}},
        }
        odds = normalize_odds(item)
        assert odds is not None
        self.assertEqual(odds["favorite_team_id"], "9")
        self.assertEqual(odds["underdog_team_id"], "5")

    def test_none_item_returns_none(self) -> None:
        self.assertIsNone(normalize_odds(None))


class ParseTeamIdFromRefTests(unittest.TestCase):
    def test_extracts_id_from_ref_url(self) -> None:
        team = {"$ref": "https://sports.core.api.espn.com/.../teams/26?lang=en&region=us"}
        self.assertEqual(parse_team_id_from_ref(team), "26")

    def test_inline_id(self) -> None:
        self.assertEqual(parse_team_id_from_ref({"id": "12"}), "12")

    def test_inline_numeric_id_coerced_to_string(self) -> None:
        self.assertEqual(parse_team_id_from_ref({"id": 12}), "12")

    def test_malformed_ref_returns_none(self) -> None:
        self.assertIsNone(parse_team_id_from_ref({"$ref": "https://x/no-team-here"}))

    def test_missing_input_returns_none(self) -> None:
        self.assertIsNone(parse_team_id_from_ref(None))
        self.assertIsNone(parse_team_id_from_ref({}))
        self.assertIsNone(parse_team_id_from_ref("not-a-dict"))


class NormalizeGameTests(unittest.TestCase):
    def _sample_event(self) -> dict:
        # Models a completed 2025 site-scoreboard event.
        return {
            "id": "401671800",
            "date": "2025-09-07T17:00Z",
            "competitions": [
                {
                    "id": "401671801",
                    "date": "2025-09-07T17:00Z",
                    "status": {
                        "type": {"state": "post", "completed": True, "name": "STATUS_FINAL"}
                    },
                    "competitors": [
                        {
                            "homeAway": "home",
                            "score": "24",
                            "team": {"id": "26", "abbreviation": "SEA"},
                        },
                        {
                            "homeAway": "away",
                            "score": "17",
                            "team": {"id": "17", "abbreviation": "NE"},
                        },
                    ],
                }
            ],
        }

    def test_normalizes_completed_game(self) -> None:
        game = normalize_game(self._sample_event(), week=1)
        assert game is not None
        self.assertEqual(game["espn_event_id"], "401671800")
        # competition_id derived from competitions[0].id, NOT the event id.
        self.assertEqual(game["competition_id"], "401671801")
        self.assertNotEqual(game["competition_id"], game["espn_event_id"])
        self.assertEqual(game["week"], 1)
        self.assertEqual(game["kickoff"], "2025-09-07T17:00Z")
        self.assertEqual(game["status"]["state"], "post")
        self.assertTrue(game["status"]["completed"])
        self.assertEqual(game["status"]["name"], "STATUS_FINAL")
        self.assertEqual(game["home"], {"team_id": "26", "abbreviation": "SEA", "score": 24})
        self.assertEqual(game["away"], {"team_id": "17", "abbreviation": "NE", "score": 17})
        # odds is initialized to null; orchestrator fills it later.
        self.assertIsNone(game["odds"])

    def test_missing_score_degrades_to_none(self) -> None:
        event = self._sample_event()
        del event["competitions"][0]["competitors"][0]["score"]
        game = normalize_game(event, week=2)
        assert game is not None
        self.assertIsNone(game["home"]["score"])

    def test_malformed_event_returns_none(self) -> None:
        self.assertIsNone(normalize_game(None, week=1))
        self.assertIsNone(normalize_game("nope", week=1))


if __name__ == "__main__":
    unittest.main()
