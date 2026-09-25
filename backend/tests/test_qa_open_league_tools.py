"""Offline tests for the issue #248 open-path tools: head-to-head, league records and
game outlook, plus the ESPN core-odds line-movement parser behind the outlook.

Run with: ``backend/.venv/bin/python -m unittest tests.test_qa_open_league_tools -v``
"""

from __future__ import annotations

import asyncio
import json
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import mock

from app.bot import db_bridge, qa_open
from app.services import live_odds, weather

_ODDS_FIXTURE = Path(__file__).parent / "fixtures" / "espn_core_odds_atl_gb.json"


def _run(coro):
    return asyncio.run(coro)


def _db(name: str, value: object, calls: list | None = None):
    async def _fake(*args, **kwargs):
        if calls is not None:
            calls.append((args, kwargs))
        return value

    return mock.patch.object(db_bridge, name, _fake)


def _async_returning(value: object, calls: list | None = None):
    async def _fake(*args, **kwargs):
        if calls is not None:
            calls.append((args, kwargs))
        return value

    return _fake


class LineMovementParserTests(unittest.TestCase):
    def test_the_real_payload_gives_the_opening_and_current_line(self) -> None:
        payload = json.loads(_ODDS_FIXTURE.read_text())
        self.assertEqual(
            live_odds.parse_line_movement(payload),
            {
                "provider": "DraftKings",
                "home_spread_open": "-7.5",
                "home_spread_now": "-4.5",
                "total_open": "46.5",
                "total_now": "42.5",
                "home_moneyline_open": "-360",
                "home_moneyline_now": "-245",
                "away_moneyline_open": "+285",
                "away_moneyline_now": "+200",
            },
        )

    def test_a_finished_game_reads_the_close_as_the_current_line(self) -> None:
        item = {
            "provider": {"name": "ESPN BET"},
            "homeTeamOdds": {
                "open": {"pointSpread": {"american": "-4.5"}},
                "close": {"pointSpread": {"american": "-7.5"}},
            },
        }
        line = live_odds.parse_line_movement({"items": [item]})
        assert line is not None
        self.assertEqual((line["home_spread_open"], line["home_spread_now"]), ("-4.5", "-7.5"))
        self.assertIsNone(line["total_now"])

    def test_junk_is_none_not_a_raise(self) -> None:
        for payload in (None, [], {"items": "x"}, {"items": [{"provider": "x"}]}):
            self.assertIsNone(live_odds.parse_line_movement(payload))

    def test_the_fetch_takes_only_int_ids(self) -> None:
        calls: list = []
        with mock.patch.object(live_odds.http_cache, "fetch_cached", _async_returning(None, calls)):
            self.assertIsNone(_run(live_odds.fetch_line_movement("1/../x")))  # type: ignore[arg-type]
            self.assertIsNone(_run(live_odds.fetch_line_movement(True)))
            _run(live_odds.fetch_line_movement(401, 402))
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][0][0].endswith("/events/401/competitions/402/odds"))


class HeadToHeadToolTests(unittest.TestCase):
    def test_the_series_is_relayed_with_the_since_1999_caveat(self) -> None:
        data = {
            "team": "BUF",
            "opponent": "MIA",
            "first_season": 1999,
            "record": {"meetings": 55, "won": 30, "lost": 25, "tied": 0},
            "playoff_meetings": [{"season": 2022, "game": "wild card"}],
            "recent": [{"season": 2025}],
        }
        calls: list = []
        with _db("get_head_to_head_async", data, calls):
            body = _run(qa_open._lookup_head_to_head(team="buf", opponent=" mia"))
        assert isinstance(body, dict)
        self.assertEqual(calls, [(("BUF", "MIA"), {})])
        self.assertIn("met 55 times", body["head_to_head_statement"])
        self.assertIn("won 30, lost 25 and tied 0", body["head_to_head_statement"])
        self.assertIn("since 1999", body["caveat"])

    def test_every_miss_is_a_note(self) -> None:
        self.assertEqual(
            _run(qa_open._lookup_head_to_head(team="BUF")),
            {"note": qa_open._NO_TEAMS_FOR_H2H_NOTE},
        )
        unknown = {"team": "BUF", "opponent": None, "record": {}}
        with _db("get_head_to_head_async", unknown):
            body = _run(qa_open._lookup_head_to_head(team="BUF", opponent="ZZZ"))
        self.assertEqual(body, {"note": qa_open._UNKNOWN_ATS_TEAM_NOTE.format(team="ZZZ")})
        never = {"team": "BUF", "opponent": "SF", "record": {"meetings": 0}}
        with _db("get_head_to_head_async", never):
            body = _run(qa_open._lookup_head_to_head(team="BUF", opponent="SF"))
        assert isinstance(body, dict)
        self.assertIn("never give a meeting from your own memory", body["note"])


class LeagueRecordsToolTests(unittest.TestCase):
    def test_the_records_are_relayed_with_the_hidden_week(self) -> None:
        data = {
            "season": 2026,
            "weeks_counted": 2,
            "open_week": 3,
            "weekly_winners": [{"week": 1, "score": 9, "winners": ["alice"]}],
            "best_weeks": [],
            "perfect_cards": [],
            "members": [{"member": "alice", "weekly_wins": 1}],
        }
        with _db("get_league_records_async", data):
            body = _run(qa_open._lookup_league_records())
        assert isinstance(body, dict)
        self.assertEqual(body["weekly_winners"], data["weekly_winners"])
        self.assertIn("2 closed weeks", body["records_statement"])
        self.assertIn("Week 3 is still open", body["records_statement"])

    def test_no_season_and_no_closed_week_are_notes(self) -> None:
        with _db("get_league_records_async", {"season": None}):
            self.assertEqual(
                _run(qa_open._lookup_league_records()), {"note": qa_open._NO_SEASON_NOTE}
            )
        with _db("get_league_records_async", {"season": 2026, "weeks_counted": 0}):
            self.assertEqual(
                _run(qa_open._lookup_league_records()), {"note": qa_open._NO_RECORDS_YET_NOTE}
            )


def _outlook(**overrides: object) -> dict:
    data = {
        "found": True,
        "season": 2026,
        "week": 5,
        "home": "GB",
        "away": "ATL",
        "status": "SCHEDULED",
        "home_score": None,
        "away_score": None,
        "kickoff_at": datetime.now(UTC) + timedelta(days=2),
        "espn_event_id": 401872948,
        "espn_competition_id": None,
        "frozen": False,
        "favorite": None,
        "spread": None,
        "total": None,
        "model_home_margin": -2.26,
        "model_home_win_prob": 0.42,
    }
    data.update(overrides)
    return data


class GameOutlookToolTests(unittest.TestCase):
    def _call(self, data: dict, movement: object = None, forecast: object = None, **kwargs):
        with (
            _db("get_game_outlook_async", data),
            mock.patch.object(live_odds, "fetch_line_movement", _async_returning(movement)),
            mock.patch.object(weather, "fetch_forecast", _async_returning({"hourly": {}})),
            mock.patch.object(weather, "parse_forecast", lambda payload, kickoff: forecast),
        ):
            return _run(qa_open._lookup_game_outlook(**kwargs))

    def test_the_model_read_names_the_side_it_favors(self) -> None:
        body = self._call(_outlook(), team="atl", week=5)
        assert isinstance(body, dict)
        # A negative home margin favors the away side, at the complement of the home odds.
        self.assertIn("has the ATL winning by 2.3 points", body["model_read"])
        self.assertIn("a 58 percent chance", body["model_read"])
        self.assertEqual(body["league_line"], qa_open._NO_LEAGUE_LINE_STATEMENT)
        self.assertEqual(body["line_movement"], qa_open._NO_LINE_MOVEMENT_STATEMENT)
        self.assertIn("never a betting tip", body["caveat"])

    def test_the_movement_and_the_forecast_are_spelled_out(self) -> None:
        movement = live_odds.parse_line_movement(json.loads(_ODDS_FIXTURE.read_text()))
        forecast = {"temperature_f": 51.2, "wind_mph": 9.1, "precip_in": None}
        body = self._call(
            _outlook(favorite="GB", spread="4.5", total="42.5", frozen=True),
            movement=movement,
            forecast=forecast,
            team="GB",
        )
        assert isinstance(body, dict)
        self.assertIn(
            "GB favored by 4.5 with a total of 42.5, frozen for picks", body["league_line"]
        )
        self.assertIn("the GB spread opened at -7.5 and is -4.5 now", body["line_movement"])
        self.assertIn("the ATL moneyline opened at +285", body["line_movement"])
        self.assertIn("51.2°F, wind 9.1 mph and not given of precipitation", body["weather"])

    def test_an_indoor_stadium_and_a_far_kickoff_never_fetch_a_forecast(self) -> None:
        indoor = self._call(_outlook(home="DET"), team="DET")
        assert isinstance(indoor, dict)
        self.assertIn("Ford Field, which has a roof", indoor["weather"])
        far = self._call(
            _outlook(kickoff_at=datetime.now(UTC) + timedelta(days=40)),
            forecast={"temperature_f": 1},
            team="GB",
        )
        assert isinstance(far, dict)
        self.assertEqual(
            far["weather"], qa_open._NO_FORECAST_STATEMENT.format(stadium="Lambeau Field", days=15)
        )
        started = self._call(_outlook(kickoff_at=datetime.now(UTC) - timedelta(hours=1)), team="GB")
        assert isinstance(started, dict)
        self.assertEqual(started["weather"], qa_open._KICKED_OFF_WEATHER_STATEMENT)

    def test_a_game_in_progress_sends_the_model_to_the_live_game(self) -> None:
        body = self._call(_outlook(status="IN_PROGRESS"), team="GB")
        assert isinstance(body, dict)
        self.assertNotIn("model_read", body)
        self.assertNotIn("league_line", body)
        self.assertIn("Call lookup_live_game with the team GB", body["note"])

    def test_a_final_game_reports_the_score_and_no_read(self) -> None:
        body = self._call(_outlook(status="FINAL", home_score=24, away_score=20), team="GB", week=3)
        assert isinstance(body, dict)
        self.assertEqual(body["final_score"], "ATL 20, GB 24")
        self.assertNotIn("model_read", body)

    def test_every_miss_is_a_note(self) -> None:
        self.assertEqual(
            _run(qa_open._lookup_game_outlook()), {"note": qa_open._NO_TEAM_FOR_OUTLOOK_NOTE}
        )
        self.assertEqual(
            _run(qa_open._lookup_game_outlook(team="KC", week=40)),
            {"note": qa_open._BAD_WEEK_NOTE},
        )
        with _db("get_game_outlook_async", {"found": False, "week": None}):
            self.assertEqual(
                _run(qa_open._lookup_game_outlook(team="KC")), {"note": qa_open._NO_SEASON_NOTE}
            )
        with _db("get_game_outlook_async", {"found": False, "week": 7, "known_team": True}):
            body = _run(qa_open._lookup_game_outlook(team="KC", week=7))
        self.assertEqual(body, {"note": qa_open._NO_OUTLOOK_GAME_NOTE.format(team="KC", week=7)})


class RegistryDescriptionTests(unittest.TestCase):
    """Each new description tells the model WHEN to call, not only what it holds."""

    def _description(self, name: str) -> str:
        tool = next(t for t in qa_open.TOOLS if t.name == name)
        return tool.spec["function"]["description"]

    def test_each_new_tool_instructs_a_call(self) -> None:
        for name, phrase in (
            ("lookup_head_to_head", "the last time two teams met"),
            ("lookup_league_records", "who has the longest lock streak"),
            ("lookup_game_outlook", "whether a line has moved"),
            ("lookup_team_ats", "over/under record"),
            ("lookup_member_season", "always takes the underdog"),
        ):
            with self.subTest(tool=name):
                self.assertIn("Call this tool when", self._description(name))
                self.assertIn(phrase, self._description(name))

    def test_the_outlook_takes_a_team_and_an_optional_week(self) -> None:
        tool = next(t for t in qa_open.TOOLS if t.name == "lookup_game_outlook")
        params = tool.spec["function"]["parameters"]
        self.assertEqual(set(params["properties"]), {"team", "week"})
        self.assertEqual(params["required"], ["team"])


if __name__ == "__main__":
    unittest.main()
