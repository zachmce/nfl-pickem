"""Offline unit tests for the pure selection/merge logic of the 2025 fixture
odds backfill.

These tests never touch the network. They exercise only the pure functions
(``select_odds_item_permissive``, ``provider_label``, ``merge_odds_into_fixture``,
``recompute_metadata``) against small inline JSON literals modeled on the real
ESPN core-API odds shapes, plus the ``normalize_odds`` round-trip to prove shape
compatibility end-to-end.

Run from the ``backend/`` directory with the standard library test runner::

    cd backend && python -m unittest tests.test_backfill_fixture_odds -v

No pytest dependency is required (none is configured for this project).
"""

from __future__ import annotations

import unittest

from scripts.backfill_fixture_odds import (
    merge_odds_into_fixture,
    provider_label,
    recompute_metadata,
    select_odds_item_permissive,
)
from scripts.gen_2025_fixture import normalize_odds


def _draftkings_item(spread: float = -3.5, total: float = 44.5) -> dict:
    """A complete core-API odds item from a non-ESPN-BET provider (DraftKings)."""
    return {
        "provider": {"id": "100", "name": "DraftKings"},
        "spread": spread,
        "overUnder": total,
        "awayTeamOdds": {
            "favorite": False,
            "underdog": True,
            "team": {"$ref": "http://x/teams/19?lang=en&region=us"},
        },
        "homeTeamOdds": {
            "favorite": True,
            "underdog": False,
            "team": {"$ref": "http://x/teams/17?lang=en&region=us"},
        },
    }


def _espn_bet_item(spread: float = -7.0, total: float = 40.0) -> dict:
    """A complete core-API odds item from ESPN BET."""
    return {
        "provider": {"id": "58", "name": "ESPN BET"},
        "spread": spread,
        "overUnder": total,
        "awayTeamOdds": {
            "favorite": True,
            "underdog": False,
            "team": {"$ref": "http://x/teams/21?lang=en&region=us"},
        },
        "homeTeamOdds": {
            "favorite": False,
            "underdog": True,
            "team": {"$ref": "http://x/teams/6?lang=en&region=us"},
        },
    }


class SelectOddsItemPermissiveTests(unittest.TestCase):
    def test_selects_draftkings_only_item(self) -> None:
        # The existing ESPN-BET-only selector would return None for this case;
        # the permissive selector returns the usable DraftKings line.
        response = {"items": [_draftkings_item()]}
        item = select_odds_item_permissive(response)
        self.assertIsNotNone(item)
        assert item is not None
        self.assertEqual(item["provider"]["name"], "DraftKings")

    def test_prefers_espn_bet_when_both_present(self) -> None:
        response = {"items": [_draftkings_item(), _espn_bet_item()]}
        item = select_odds_item_permissive(response)
        assert item is not None
        self.assertEqual(item["provider"]["name"], "ESPN BET")

    def test_prefers_espn_bet_regardless_of_order(self) -> None:
        response = {"items": [_espn_bet_item(), _draftkings_item()]}
        item = select_odds_item_permissive(response)
        assert item is not None
        self.assertEqual(item["provider"]["name"], "ESPN BET")

    def test_returns_non_espn_bet_provider_for_labeling(self) -> None:
        # Only a non-ESPN-BET provider present: selector returns it and the
        # script can read its real provider name for honest labeling.
        response = {"items": [_draftkings_item()]}
        item = select_odds_item_permissive(response)
        assert item is not None
        self.assertEqual(provider_label(item), "DraftKings")

    def test_partial_line_is_not_usable_returns_none(self) -> None:
        # Item missing team odds -> normalize_odds yields null favorite/underdog,
        # so it is NOT usable and the selector returns None.
        partial = {
            "provider": {"id": "100", "name": "DraftKings"},
            "spread": -3.0,
            "overUnder": 41.0,
        }
        self.assertIsNone(select_odds_item_permissive({"items": [partial]}))

    def test_skips_unusable_falls_through_to_usable(self) -> None:
        partial = {"provider": {"id": "200", "name": "Caesars"}, "spread": -2.0}
        response = {"items": [partial, _draftkings_item()]}
        item = select_odds_item_permissive(response)
        assert item is not None
        self.assertEqual(item["provider"]["name"], "DraftKings")

    def test_malformed_response_returns_none(self) -> None:
        self.assertIsNone(select_odds_item_permissive(None))
        self.assertIsNone(select_odds_item_permissive({}))
        self.assertIsNone(select_odds_item_permissive({"items": "nope"}))
        self.assertIsNone(select_odds_item_permissive({"items": []}))

    def test_items_with_non_dict_entries_skipped(self) -> None:
        response = {"items": ["junk", None, 42, _draftkings_item()]}
        item = select_odds_item_permissive(response)
        assert item is not None
        self.assertEqual(item["provider"]["name"], "DraftKings")


class ProviderLabelTests(unittest.TestCase):
    def test_reads_real_provider_name(self) -> None:
        self.assertEqual(provider_label(_draftkings_item()), "DraftKings")
        self.assertEqual(provider_label(_espn_bet_item()), "ESPN BET")

    def test_falls_back_to_string_id_when_name_missing(self) -> None:
        self.assertEqual(provider_label({"provider": {"id": 100}}), "100")
        self.assertEqual(provider_label({"provider": {"id": "100"}}), "100")

    def test_unknown_when_no_provider_info(self) -> None:
        self.assertEqual(provider_label({}), "unknown")
        self.assertEqual(provider_label({"provider": {}}), "unknown")
        self.assertEqual(provider_label(None), "unknown")
        self.assertEqual(provider_label({"provider": "nope"}), "unknown")


class NormalizeOddsRoundTripTests(unittest.TestCase):
    def test_draftkings_item_yields_complete_odds(self) -> None:
        # End-to-end shape proof: a DraftKings-only response selected by the
        # permissive selector and normalized yields all four fields non-null.
        response = {"items": [_draftkings_item(spread=-6.5, total=48.0)]}
        selected = select_odds_item_permissive(response)
        assert selected is not None
        odds = normalize_odds(selected)
        assert odds is not None
        self.assertEqual(odds["spread"], -6.5)
        self.assertEqual(odds["total"], 48.0)
        self.assertEqual(odds["favorite_team_id"], "17")
        self.assertEqual(odds["underdog_team_id"], "19")

    def test_negative_favorite_sign_preserved(self) -> None:
        # The spread stays negative (home-favored sign convention) so the
        # importer's abs(spread) yields a positive magnitude.
        selected = select_odds_item_permissive({"items": [_draftkings_item(spread=-9.5)]})
        assert selected is not None
        odds = normalize_odds(selected)
        assert odds is not None
        self.assertLess(odds["spread"], 0)
        self.assertEqual(abs(odds["spread"]), 9.5)


class MergeOddsIntoFixtureTests(unittest.TestCase):
    def _fixture(self) -> dict:
        return {
            "metadata": {
                "season": 2025,
                "games_total": 3,
                "games_with_odds": 1,
                "games_without_odds": 2,
            },
            "games": [
                {
                    "espn_event_id": "1001",
                    "competition_id": "1001",
                    "week": 14,
                    "home": {"team_id": "17", "abbreviation": "NE"},
                    "away": {"team_id": "19", "abbreviation": "NYG"},
                    "odds": None,
                },
                {
                    "espn_event_id": "1002",
                    "competition_id": "1002",
                    "week": 1,
                    "home": {"team_id": "21", "abbreviation": "PHI"},
                    "away": {"team_id": "6", "abbreviation": "DAL"},
                    "odds": {
                        "provider": "ESPN BET",
                        "spread": -7.5,
                        "total": 47.5,
                        "favorite_team_id": "21",
                        "underdog_team_id": "6",
                    },
                },
                {
                    "espn_event_id": "1003",
                    "competition_id": "1003",
                    "week": 15,
                    "home": {"team_id": "12", "abbreviation": "KC"},
                    "away": {"team_id": "24", "abbreviation": "LAC"},
                    "odds": None,
                },
            ],
        }

    def _fake_fetch(self) -> "object":
        # event 1001 -> usable DraftKings line; event 1003 -> nothing usable.
        responses = {
            "1001": {"items": [_draftkings_item(spread=-3.5, total=44.5)]},
            "1003": {"items": [{"provider": {"id": "200", "name": "Caesars"}}]},
        }

        def fetch(event_id: str, competition_id: str):  # noqa: ANN202
            return responses.get(event_id)

        return fetch

    def test_null_game_gets_provider_labeled_line(self) -> None:
        fixture = self._fixture()
        report = merge_odds_into_fixture(fixture, self._fake_fetch())
        filled = fixture["games"][0]["odds"]
        self.assertIsNotNone(filled)
        self.assertEqual(filled["provider"], "DraftKings")
        self.assertEqual(filled["spread"], -3.5)
        self.assertEqual(filled["total"], 44.5)
        self.assertEqual(filled["favorite_team_id"], "17")
        self.assertEqual(filled["underdog_team_id"], "19")
        self.assertEqual(report.filled, 1)
        self.assertEqual(report.filled_by_provider["DraftKings"], 1)

    def test_prefilled_game_is_untouched(self) -> None:
        fixture = self._fixture()
        original = dict(fixture["games"][1]["odds"])
        merge_odds_into_fixture(fixture, self._fake_fetch())
        self.assertEqual(fixture["games"][1]["odds"], original)

    def test_order_is_preserved(self) -> None:
        fixture = self._fixture()
        merge_odds_into_fixture(fixture, self._fake_fetch())
        ids = [g["espn_event_id"] for g in fixture["games"]]
        self.assertEqual(ids, ["1001", "1002", "1003"])
        self.assertEqual(len(fixture["games"]), 3)

    def test_unusable_fetch_left_null_and_reported(self) -> None:
        fixture = self._fixture()
        report = merge_odds_into_fixture(fixture, self._fake_fetch())
        # event 1003 yields nothing usable -> stays null AND is reported.
        self.assertIsNone(fixture["games"][2]["odds"])
        self.assertEqual(report.still_null_count, 1)
        reported = report.still_null[0]
        self.assertEqual(reported["espn_event_id"], "1003")
        self.assertEqual(reported["week"], 15)
        self.assertEqual(reported["matchup"], "LAC@KC")

    def test_already_had_odds_counted(self) -> None:
        fixture = self._fixture()
        report = merge_odds_into_fixture(fixture, self._fake_fetch())
        self.assertEqual(report.already_had_odds, 1)
        self.assertEqual(report.games_total, 3)

    def test_idempotent_second_run_no_change(self) -> None:
        fixture = self._fixture()
        merge_odds_into_fixture(fixture, self._fake_fetch())
        snapshot = [dict(g["odds"]) if g["odds"] else None for g in fixture["games"]]
        # Second run: fetch returns nothing usable for the still-null game; the
        # already-filled games must be skipped untouched.
        report2 = merge_odds_into_fixture(fixture, self._fake_fetch())
        after = [dict(g["odds"]) if g["odds"] else None for g in fixture["games"]]
        self.assertEqual(snapshot, after)
        # The previously-filled DraftKings game is now skipped, not re-filled.
        self.assertEqual(report2.already_had_odds, 2)
        self.assertEqual(report2.filled, 0)

    def test_recompute_metadata_reflects_reality(self) -> None:
        fixture = self._fixture()
        merge_odds_into_fixture(fixture, self._fake_fetch())
        counts = recompute_metadata(fixture)
        self.assertEqual(counts["games_total"], 3)
        self.assertEqual(counts["games_with_odds"], 2)  # prefilled + 1001
        self.assertEqual(counts["games_without_odds"], 1)  # 1003 still null
        self.assertEqual(fixture["metadata"]["games_with_odds"], 2)
        self.assertEqual(fixture["metadata"]["games_without_odds"], 1)


if __name__ == "__main__":
    unittest.main()
