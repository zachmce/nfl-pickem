"""Offline tests for the issue #248 knowledge tools: career, awards, FPI, QBR, roster moves.

Fixtures are trimmed real ESPN payloads captured 2026-09-24. No socket is opened: the
fetch tests patch ``espn_extra._fetch_cached`` and the adapter tests patch the fetches.
"""

from __future__ import annotations

import asyncio
import copy
import json
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

from app.bot import qa_open
from app.services import espn_extra

_FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text())


def _spec(name: str) -> dict:
    return next(t.spec["function"] for t in qa_open.TOOLS if t.name == name)


class _FetchCapture:
    """Patch ``_fetch_cached`` and record every URL it would have requested."""

    def __init__(self) -> None:
        self.urls: list[str] = []
        self.keys: list[str] = []

    async def __call__(self, url, *, cache_key, ttl_seconds, label):
        self.urls.append(url)
        self.keys.append(cache_key)
        return {}


def _capture() -> tuple[_FetchCapture, Any]:
    capture = _FetchCapture()
    return capture, patch.object(espn_extra, "_fetch_cached", capture)


class CareerParserTests(unittest.TestCase):
    def test_career_keeps_every_season_row_and_the_career_totals(self) -> None:
        career = espn_extra.parse_athlete_career(_load("espn_athlete_career.json"))
        assert career is not None
        passing = career["categories"]["Passing"]
        self.assertEqual(passing["career_totals"]["YDS"], "89,214")
        self.assertEqual(passing["career_totals"]["TD"], "649")
        self.assertEqual([row["season"] for row in passing["seasons"]], [2000, 2007, 2022])
        self.assertEqual(passing["seasons"][1]["team"], "New England Patriots")
        self.assertEqual(passing["seasons"][1]["TD"], "50")
        self.assertEqual(passing["seasons"][2]["team"], "Tampa Bay Buccaneers")
        self.assertEqual(passing["columns"]["YDS"], "Passing Yards")

    def test_an_all_zero_category_is_dropped(self) -> None:
        career = espn_extra.parse_athlete_career(_load("espn_athlete_career.json"))
        assert career is not None
        self.assertNotIn("Defensive", career["categories"])
        self.assertIn("Rushing", career["categories"])

    def test_a_season_split_between_clubs_uses_the_combined_row_and_names_both(self) -> None:
        payload = _load("espn_athlete_career.json")
        rows = payload["categories"][0]["statistics"]
        moved = copy.deepcopy(rows[-1])
        moved["teamSlug"] = "new-england-patriots"
        moved["teamId"] = "17"
        combined = copy.deepcopy(rows[-1])
        combined.pop("teamId")
        combined["teamSlug"] = "2022 Totals"
        combined["stats"][4] = "9,999"
        rows.extend([moved, combined])
        career = espn_extra.parse_athlete_career(payload)
        assert career is not None
        last = career["categories"]["Passing"]["seasons"][-1]
        self.assertEqual(last["YDS"], "9,999")
        self.assertEqual(last["team"], "Tampa Bay Buccaneers and New England Patriots")

    def test_a_duplicate_short_label_falls_back_to_the_long_name(self) -> None:
        category = {"labels": ["YDS", "YDS"], "displayNames": ["Sack Yards", "Return Yards"]}
        self.assertEqual(espn_extra._column_keys(category), ["Sack Yards", "Return Yards"])

    def test_unusable_shapes_return_none(self) -> None:
        for payload in (None, [], {"categories": "x"}):
            self.assertIsNone(espn_extra.parse_athlete_career(payload))

    def test_bio_reads_draft_college_and_status(self) -> None:
        bio = espn_extra.parse_athlete_bio(_load("espn_athlete_bio.json"))
        assert bio is not None
        self.assertEqual(bio["draft"], "2000: Rd 6, Pk 199 (NE)")
        self.assertEqual(bio["college"], "Michigan")
        self.assertIs(bio["active"], False)
        self.assertIsNone(espn_extra.parse_athlete_bio({"athlete": "x"}))

    def test_awards_count_the_listed_seasons(self) -> None:
        awards = espn_extra.parse_athlete_awards(_load("espn_athlete_overview.json"))
        assert awards is not None
        mvp = next(a for a in awards if a["award"] == "NFL MVP")
        self.assertEqual(mvp, {"award": "NFL MVP", "times": 3, "seasons": ["2007", "2010", "2017"]})
        self.assertEqual(espn_extra.parse_athlete_awards({}), [])


class SeasonAwardParserTests(unittest.TestCase):
    def test_award_reads_the_athlete_id_and_the_team(self) -> None:
        award = espn_extra.parse_season_award(_load("espn_season_awards.json")["2012_477"])
        self.assertEqual(
            award,
            {
                "award": "NFL MVP",
                "description": "AP NFL Most Valuable Player",
                "winners": [{"athlete_id": "10452", "team": "Minnesota Vikings"}],
            },
        )

    def test_a_winner_with_no_athlete_keeps_only_the_team(self) -> None:
        award = espn_extra.parse_season_award(_load("espn_season_awards.json")["1980_477"])
        assert award is not None
        self.assertEqual(award["winners"], [{"athlete_id": None, "team": "Cleveland Browns"}])

    def test_a_ref_on_another_host_yields_no_id(self) -> None:
        # SSRF guard: only an id on the core host counts, and no ref is ever fetched.
        for ref in (
            "http://evil.example.com/v2/sports/football/leagues/nfl/athletes/10452",
            "http://sports.core.api.espn.com.evil.com/v2/sports/football/leagues/nfl/athletes/1",
            "https://169.254.169.254/latest/meta-data",
            "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/athletes/1",
        ):
            with self.subTest(ref=ref):
                self.assertIsNone(espn_extra.core_ref_athlete_id(ref))
                self.assertIsNone(espn_extra.core_ref_team(ref.replace("athletes", "teams")))
        payload = {"winners": [{"athlete": {"$ref": "http://evil.example.com/athletes/5"}}]}
        self.assertEqual(
            espn_extra.parse_season_award(payload),
            {"award": None, "description": None, "winners": []},
        )

    def test_award_names_and_aliases_resolve(self) -> None:
        self.assertEqual(espn_extra.season_award_id("MVP"), "477")
        self.assertEqual(espn_extra.season_award_id("the AP Most Valuable Player"), "477")
        self.assertEqual(espn_extra.season_award_id("NFL most valuable player"), "477")
        self.assertEqual(espn_extra.season_award_id("DPOY"), "479")
        self.assertEqual(espn_extra.season_award_id("super bowl mvp"), "317")
        self.assertIsNone(espn_extra.season_award_id("heisman"))

    def test_season_athlete_reads_the_name(self) -> None:
        self.assertEqual(
            espn_extra.parse_season_athlete(_load("espn_season_athlete.json")),
            {"name": "Adrian Peterson", "position": "RB"},
        )


class PowerIndexQbrTransactionsParserTests(unittest.TestCase):
    def test_power_index_for_one_team(self) -> None:
        facts = espn_extra.parse_power_index(_load("espn_power_index.json"), "sf")
        assert facts is not None
        (row,) = facts["teams"]
        self.assertEqual(row["team"], "San Francisco 49ers")
        self.assertEqual(row["fpi_rank"], 1)
        self.assertEqual(row["playoffs_pct"], 97.1)
        self.assertEqual(row["win_super_bowl_pct"], 24.2)
        self.assertEqual(facts["season"], 2026)

    def test_power_index_top_list_is_ranked_and_an_unknown_team_is_empty(self) -> None:
        facts = espn_extra.parse_power_index(_load("espn_power_index.json"))
        assert facts is not None
        self.assertEqual([r["fpi_rank"] for r in facts["teams"]], [1, 2, 3])
        empty = espn_extra.parse_power_index(_load("espn_power_index.json"), "XXX")
        self.assertEqual(empty, {"season": 2026, "teams": []})

    def test_qbr_season_list_and_player_filter(self) -> None:
        payload = _load("espn_qbr.json")["season_2019"]
        facts = espn_extra.parse_qbr(payload)
        assert facts is not None
        self.assertEqual(facts["quarterbacks"][0]["player"], "Lamar Jackson")
        self.assertEqual(facts["quarterbacks"][0]["total_qbr"], "83.0")
        one = espn_extra.parse_qbr(payload, "mahomes")
        assert one is not None
        self.assertEqual([q["player"] for q in one["quarterbacks"]], ["Patrick Mahomes"])

    def test_weekly_qbr_names_the_opponent(self) -> None:
        facts = espn_extra.parse_qbr(_load("espn_qbr.json")["week_2025_3"])
        assert facts is not None
        self.assertEqual(facts["quarterbacks"][0]["opponent"], "Tennessee Titans")

    def test_transactions_keep_date_team_and_text(self) -> None:
        payload = _load("espn_transactions.json")
        league = espn_extra.parse_transactions(payload["league"])
        assert league is not None
        self.assertEqual(league[0]["team"], "ATL")
        self.assertEqual(league[0]["date"], "2026-09-23")
        team = espn_extra.parse_transactions(payload["team_nyj"], "nyj")
        assert team is not None
        self.assertEqual({m["team"] for m in team}, {"NYJ"})
        self.assertIsNone(espn_extra.parse_transactions({"transactions": "x"}))


class FetchGuardTests(unittest.TestCase):
    def test_postseason_career_uses_the_playoff_url(self) -> None:
        capture, patcher = _capture()
        with patcher:
            asyncio.run(espn_extra.fetch_athlete_career_stats("2330", postseason=True))
            asyncio.run(espn_extra.fetch_athlete_career_stats("2330/../x", postseason=True))
        self.assertEqual(len(capture.urls), 1)
        self.assertTrue(capture.urls[0].endswith("/athletes/2330/stats?seasontype=3"))

    def test_award_fetch_accepts_only_known_ids_and_sane_seasons(self) -> None:
        capture, patcher = _capture()
        with patcher:
            asyncio.run(espn_extra.fetch_season_award(2012, "477"))
            asyncio.run(espn_extra.fetch_season_award(2012, "999"))
            asyncio.run(espn_extra.fetch_season_award(1900, "477"))
            asyncio.run(espn_extra.fetch_season_award(True, "477"))
            asyncio.run(espn_extra.fetch_season_athlete(2012, "10452"))
            asyncio.run(espn_extra.fetch_season_athlete(2012, "x"))
        self.assertEqual(
            capture.urls,
            [
                "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2012/awards/477",
                "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2012/athletes/10452",
            ],
        )

    def test_qbr_fetch_bounds_season_and_week(self) -> None:
        capture, patcher = _capture()
        with patcher:
            asyncio.run(espn_extra.fetch_qbr(2019))
            asyncio.run(espn_extra.fetch_qbr(2025, 3))
            asyncio.run(espn_extra.fetch_qbr(2005))
            asyncio.run(espn_extra.fetch_qbr(2025, 30))
        self.assertEqual(len(capture.urls), 2)
        self.assertIn("qbrType=seasons", capture.urls[0])
        self.assertTrue(
            capture.urls[1].endswith(
                "qbrType=weeks&seasontype=2&isqualified=true&season=2025&week=3"
            )
        )

    def test_transactions_fetch_maps_the_team_to_its_espn_id(self) -> None:
        capture, patcher = _capture()
        with patcher:
            asyncio.run(espn_extra.fetch_transactions("nyj"))
            asyncio.run(espn_extra.fetch_transactions("NYJ&x=1"))
            asyncio.run(espn_extra.fetch_transactions(None, 2024))
        self.assertEqual(len(capture.urls), 2)
        self.assertTrue(capture.urls[0].endswith("transactions?limit=15&team=20"))
        self.assertTrue(capture.urls[1].endswith("transactions?limit=15&season=2024"))


class GameLogAndRecordFixTests(unittest.TestCase):
    def test_postseason_game_log_keeps_only_playoff_games(self) -> None:
        payload = _load("espn_athlete_gamelog.json")
        post = espn_extra.parse_athlete_gamelog(payload, postseason=True)
        regular = espn_extra.parse_athlete_gamelog(payload, postseason=False)
        assert post is not None and regular is not None
        self.assertTrue(post["games"])
        self.assertTrue(all("Postseason" in (g["season_type"] or "") for g in post["games"]))
        self.assertTrue(all("Regular" in (g["season_type"] or "") for g in regular["games"]))

    def test_a_zero_zero_split_is_dropped_but_the_overall_record_stays(self) -> None:
        payload = _load("espn_standings.json")
        for entry in payload["standings"]:
            for record in entry.get("records", []):
                if record.get("type") != "total":
                    record["summary"] = "0-0"
        team_id = espn_extra.NFL_TEAM_ID_BY_ABBR["NE"]
        club = next(e for e in payload["standings"] if f"/teams/{team_id}?" in e["team"]["$ref"])
        overall = next(r for r in club["records"] if r.get("type") == "total")["summary"]
        facts = espn_extra.parse_team_record(payload, "NE")
        assert facts is not None
        self.assertEqual(facts["records"], {"overall record": overall})


class AdapterTests(unittest.TestCase):
    def _run(self, coro) -> dict:
        result = asyncio.run(coro)
        assert isinstance(result, dict)
        return result

    def test_career_adapter_joins_stats_bio_and_awards(self) -> None:
        resolved = {"athlete_id": "2330", "identity": {"player": "Tom Brady"}, "on_roster": None}
        with (
            patch.object(qa_open, "_resolve_player", AsyncMock(return_value=resolved)),
            patch.object(
                espn_extra,
                "fetch_athlete_career_stats",
                AsyncMock(return_value=_load("espn_athlete_career.json")),
            ) as stats,
            patch.object(
                espn_extra,
                "fetch_athlete_bio",
                AsyncMock(return_value=_load("espn_athlete_bio.json")),
            ),
            patch.object(
                espn_extra,
                "fetch_athlete_overview",
                AsyncMock(return_value=_load("espn_athlete_overview.json")),
            ),
        ):
            result = self._run(qa_open._lookup_player_career("tom brady", "postseason"))
        assert isinstance(result, dict)
        stats.assert_awaited_once_with("2330", postseason=True)
        self.assertEqual(result["career"]["Passing"]["career_totals"]["TD"], "649")
        self.assertEqual(result["bio"]["college"], "Michigan")
        self.assertIn("playoff statistics", result["career_statement"])
        self.assertNotIn("athlete_id", json.dumps(result))

    def test_career_adapter_keeps_identity_when_stats_fail(self) -> None:
        resolved = {"athlete_id": "1", "identity": {"player": "A B"}, "on_roster": None}
        with (
            patch.object(qa_open, "_resolve_player", AsyncMock(return_value=resolved)),
            patch.object(espn_extra, "fetch_athlete_career_stats", AsyncMock(return_value=None)),
            patch.object(espn_extra, "fetch_athlete_bio", AsyncMock(return_value=None)),
            patch.object(espn_extra, "fetch_athlete_overview", AsyncMock(return_value=None)),
        ):
            result = self._run(qa_open._lookup_player_career("a b"))
        self.assertEqual(result["player"], "A B")
        self.assertIn("never give a career figure from your own memory", result["note"])

    def test_awards_adapter_names_winners_and_keeps_a_team_only_winner(self) -> None:
        awards = _load("espn_season_awards.json")

        async def award(season, award_id):
            return awards["2012_477"] if award_id == "477" else awards["1980_477"]

        with (
            patch.object(espn_extra, "fetch_season_award", side_effect=award),
            patch.object(
                espn_extra,
                "fetch_season_athlete",
                AsyncMock(return_value=_load("espn_season_athlete.json")),
            ) as athlete,
        ):
            one = self._run(qa_open._lookup_season_awards(2012, "mvp"))
            every = self._run(qa_open._lookup_season_awards(2012))
        self.assertEqual(
            one["awards"],
            [
                {
                    "award": "NFL MVP",
                    "source": "ESPN",
                    "winners": [
                        {"winner": "Adrian Peterson", "position": "RB", "team": "Minnesota Vikings"}
                    ],
                }
            ],
        )
        self.assertEqual(athlete.await_count, 2)  # one per call: the id is shared
        # ESPN named no player for the other awards: the AP tables fill every one they
        # hold, and only the Walter Payton award keeps ESPN's team-only row.
        team_only = [a for a in every["awards"] if a["winners"][0]["winner"] is None]
        self.assertEqual([a["award"] for a in team_only], ["NFL MVP"])
        filled = {a["award"]: a for a in every["awards"] if a["source"] == "AP award tables"}
        self.assertEqual(
            filled["AP Defensive Player of the Year"]["winners"][0]["winner"], "J. J. Watt"
        )
        self.assertEqual(len(every["awards"]), len(espn_extra.SEASON_AWARDS))

    def test_awards_adapter_notes_an_unknown_award_and_a_missing_season(self) -> None:
        self.assertIn(
            "is not an award", self._run(qa_open._lookup_season_awards(2012, "heisman"))["note"]
        )
        self.assertIn("No season", self._run(qa_open._lookup_season_awards(None))["note"])

    def test_outlook_adapter_labels_the_source(self) -> None:
        with patch.object(
            espn_extra, "fetch_power_index", AsyncMock(return_value=_load("espn_power_index.json"))
        ):
            result = self._run(qa_open._lookup_team_outlook("BUF"))
            missing = self._run(qa_open._lookup_team_outlook("XXX"))
        self.assertEqual(result["teams"][0]["team"], "Buffalo Bills")
        self.assertIn("ESPN's Football Power Index", result["outlook_statement"])
        self.assertIn("not in ESPN's Football Power Index", missing["note"])

    def test_qbr_adapter_rejects_a_season_before_2006_without_a_fetch(self) -> None:
        with patch.object(espn_extra, "fetch_qbr", AsyncMock()) as fetch:
            result = self._run(qa_open._lookup_qbr(2001))
        fetch.assert_not_awaited()
        self.assertIn("starts with the 2006 season", result["note"])

    def test_qbr_adapter_names_a_missing_quarterback(self) -> None:
        with patch.object(
            espn_extra, "fetch_qbr", AsyncMock(return_value=_load("espn_qbr.json")["season_2019"])
        ):
            result = self._run(qa_open._lookup_qbr(2019, None, "Zach Wilson"))
        self.assertIn("Zach Wilson is not among the qualified", result["note"])

    def test_transactions_adapter_rejects_an_unknown_team_without_a_fetch(self) -> None:
        with patch.object(espn_extra, "fetch_transactions", AsyncMock()) as fetch:
            result = self._run(qa_open._lookup_transactions("ZZZ"))
        fetch.assert_not_awaited()
        self.assertIn("not an NFL team abbreviation", result["note"])


class RegistryTests(unittest.TestCase):
    def test_new_descriptions_instruct_the_call_and_name_the_neighbour(self) -> None:
        cases = {
            "lookup_player_career": ("Call this tool every time", "lookup_player_season_stats"),
            "lookup_season_awards": ("Call this tool every time", "lookup_player_career"),
            "lookup_team_outlook": ("Call this tool when", "lookup_team_record"),
            "lookup_qbr": ("Call this tool when", "lookup_league_leaders"),
            "lookup_transactions": ("Call this tool when", "lookup_injury_report"),
        }
        for name, (instruction, neighbour) in cases.items():
            with self.subTest(tool=name):
                description = _spec(name)["description"]
                self.assertIn(instruction, description)
                self.assertIn(neighbour, description)

    def test_award_enum_is_derived_from_the_seam(self) -> None:
        enum = _spec("lookup_season_awards")["parameters"]["properties"]["award"]["enum"]
        self.assertEqual(enum, list(espn_extra.SEASON_AWARDS))

    def test_game_log_takes_a_season_type(self) -> None:
        params = _spec("lookup_player_game_log")["parameters"]["properties"]
        self.assertEqual(params["season_type"]["enum"], ["regular", "postseason"])

    def test_no_new_spec_declares_a_url_parameter(self) -> None:
        for name in (
            "lookup_player_career",
            "lookup_season_awards",
            "lookup_team_outlook",
            "lookup_qbr",
            "lookup_transactions",
        ):
            props = set(_spec(name)["parameters"]["properties"])
            self.assertFalse(props & {"url", "endpoint", "path", "host"})


if __name__ == "__main__":
    unittest.main()
