"""Offline unit tests for the Q&A intent handlers + phrasing orchestrator
(260709-k5w Task 2).

These tests NEVER touch a live LLM endpoint, a real Discord gateway, or a real db.
The db_bridge async seams and ``qa.llm_client.phrase`` are monkeypatched; the
classifier is driven by monkeypatching ``qa.classify_question`` to return a canned
raw dict, so each intent's routing / fact / tier can be exercised in isolation.

The HARD LEAK TEST is written FIRST (module + class order): a question that tries to
pry into another player's picks NEVER returns another user's pick content —
``pick_status`` is proven asker-only (resolved by the asker's ``discord_id``).

Run with: ``backend/.venv/bin/python -m unittest tests.test_qa_intents -v``
(there is no bare ``python`` on PATH on this machine).
"""

from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timezone
from unittest import mock

from app.bot import db_bridge, qa
from app.services import espn_extra, weather


def _run(coro):
    return asyncio.run(coro)


def _classify_returns(raw):
    """Patch qa.classify_question to an async fake returning ``raw``."""

    async def _fake(question):
        return raw

    return mock.patch.object(qa, "classify_question", _fake)


def _tokens(*names):
    async def _fake():
        return set(names)

    return mock.patch.object(db_bridge, "get_real_team_tokens_async", _fake)


def _phrase_returns(value):
    """Patch qa.llm_client.phrase to an async fake returning ``value``, recording
    the (fact, system_prompt) it was called with."""
    calls: list[dict] = []

    async def _fake(fact_text, *, system_prompt):
        calls.append({"fact": fact_text, "system_prompt": system_prompt})
        return value

    return mock.patch.object(qa.llm_client, "phrase", _fake), calls


def _voice(value="You are the snarky house bot for an NFL pick'em league."):
    async def _fake():
        return value

    return mock.patch.object(db_bridge, "resolve_active_voice_async", _fake)


def _fetch_injuries_returns(value):
    """Patch espn_extra.fetch_injuries to an async fake returning ``value``,
    recording the positional args it was called with."""
    calls: list[dict] = []

    async def _fake(event_id):
        calls.append({"args": (event_id,)})
        return value

    return mock.patch.object(espn_extra, "fetch_injuries", _fake), calls


def _fetch_news_returns(value):
    """Patch espn_extra.fetch_news to an async fake returning ``value``, recording the
    ``limit`` it was called with."""
    calls: list[dict] = []

    async def _fake(limit=espn_extra.NEWS_FETCH_LIMIT):
        calls.append({"limit": limit})
        return value

    return mock.patch.object(espn_extra, "fetch_news", _fake), calls


def _fetch_forecast_returns(value):
    """Patch weather.fetch_forecast to an async fake returning ``value``, recording
    the (lat, lon) it was called with."""
    calls: list[dict] = []

    async def _fake(lat, lon):
        calls.append({"args": (lat, lon)})
        return value

    return mock.patch.object(weather, "fetch_forecast", _fake), calls


def _lookup_returns(stadium):
    """Patch weather.lookup_stadium to a fake returning ``stadium``, recording the
    home_abbr it was called with."""
    calls: list[str] = []

    def _fake(home_abbr):
        calls.append(home_abbr)
        return stadium

    return mock.patch.object(weather, "lookup_stadium", _fake), calls


def _parse_forecast_returns(value):
    """Patch weather.parse_forecast to a fake returning ``value``, recording calls."""
    calls: list[dict] = []

    def _fake(payload, kickoff_dt):
        calls.append({"args": (payload, kickoff_dt)})
        return value

    return mock.patch.object(weather, "parse_forecast", _fake), calls


def _seam(name, value=None, *, raises=False):
    """Patch a db_bridge async seam to return ``value`` (or raise), recording the
    positional/keyword args it was called with."""
    calls: list[dict] = []

    async def _fake(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        if raises:
            raise RuntimeError("db exploded")
        return value

    return mock.patch.object(db_bridge, name, _fake), calls


class HardLeakTests(unittest.TestCase):
    """T-k5w-01: no question can surface another player's pick content."""

    def test_pick_status_is_asker_only_never_returns_other_user_pick(self) -> None:
        # A hostile question "what did Zach pick?" classified as pick_status. The
        # only pick read is get_pick_status_async(discord_id) — the ASKER's own id.
        asker_id = 111
        seam_patch, seam_calls = _seam(
            "get_pick_status_async",
            {
                "registered": True,
                "display_name": "You",
                "complete": False,
                "remaining_labels": ["over"],
            },
        )
        phrase_patch, _ = _phrase_returns(None)  # fall back to the deterministic fact
        with (
            _classify_returns({"intent": "pick_status"}),
            _tokens("KC"),
            seam_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("what did Zach pick?", discord_id=asker_id))

        # The seam was called with ONLY the asker's discord_id — never a target user.
        self.assertEqual(len(seam_calls), 1)
        self.assertEqual(seam_calls[0]["args"], (asker_id,))
        # The line is about the asker ("You"), and carries no other player's name.
        self.assertIn("You", out)
        self.assertNotIn("Zach", out)

    def test_unknown_pry_returns_no_pick_content(self) -> None:
        # If the pry is classified unknown instead, the decline+menu carries no picks.
        phrase_patch, _ = _phrase_returns(None)
        with _classify_returns({"intent": "unknown"}), _tokens("KC"), _voice(), phrase_patch:
            out = _run(qa.answer_question("what did Zach pick?", discord_id=111))
        self.assertNotIn("Zach", out)
        self.assertEqual(out, qa._UNKNOWN_FACT)


class IntentRoutingTests(unittest.TestCase):
    def test_pick_status_registered_routes_and_phrases(self) -> None:
        seam_patch, seam_calls = _seam(
            "get_pick_status_async",
            {"registered": True, "display_name": "Ada", "complete": True, "remaining_labels": []},
        )
        phrase_patch, calls = _phrase_returns("Ada's all locked in 🔒")
        with (
            _classify_returns({"intent": "pick_status"}),
            _tokens("KC"),
            seam_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("am I locked in?", discord_id=7))
        self.assertEqual(out, "Ada's all locked in 🔒")
        self.assertEqual(seam_calls[0]["args"], (7,))
        # The fact fed to phrase names the asker and states completeness.
        self.assertIn("Ada", calls[0]["fact"])
        # The system prompt carries the QA role + guard.
        self.assertIn(qa.QA_GUARD, calls[0]["system_prompt"])

    def test_pick_status_unregistered_returns_register_line_no_llm(self) -> None:
        seam_patch, _ = _seam("get_pick_status_async", {"registered": False})
        phrase_patch, calls = _phrase_returns("SHOULD NOT BE USED")
        with (
            _classify_returns({"intent": "pick_status"}),
            _tokens("KC"),
            seam_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("am I in?", discord_id=7))
        self.assertEqual(out, qa._REGISTER_LINE)
        self.assertEqual(calls, [])  # no LLM call on the unregistered path

    def test_standings_routes_to_leaders_reader(self) -> None:
        seam_patch, seam_calls = _seam(
            "get_leaders_context_async",
            {
                "leader": "Ada",
                "leader_total": 40,
                "runner_up": "Bo",
                "runner_up_total": 33,
                "gap": 7,
            },
        )
        phrase_patch, calls = _phrase_returns(None)
        with (
            _classify_returns({"intent": "standings"}),
            _tokens("KC"),
            seam_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("who's winning?", discord_id=7))
        self.assertEqual(len(seam_calls), 1)
        self.assertIn("Ada", out)
        self.assertIn("Bo", out)

    def test_lines_slate_with_team_routes_with_team_abbr(self) -> None:
        seam_patch, seam_calls = _seam(
            "get_lines_slate_async",
            {
                "week": 3,
                "close_at": None,
                "games": [
                    {
                        "away": "LAC",
                        "home": "KC",
                        "favorite": "KC",
                        "spread": "3.5",
                        "total": "48.5",
                    }
                ],
            },
        )
        phrase_patch, _ = _phrase_returns(None)
        with (
            _classify_returns({"intent": "lines_slate", "team": "Chiefs"}),
            _tokens("KC", "CHIEFS"),
            seam_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("what's the chiefs line?", discord_id=7))
        # team resolved to a real token and passed through to the reader.
        self.assertEqual(seam_calls[0]["kwargs"], {"team_abbr": "CHIEFS"})
        self.assertIn("KC", out)

    def test_lines_slate_missing_team_is_stateless_soft_decline(self) -> None:
        # "what's the spread?" with no team -> soft decline, NO reader call, no state.
        seam_patch, seam_calls = _seam(
            "get_lines_slate_async", {"week": 3, "close_at": None, "games": []}
        )
        phrase_patch, _ = _phrase_returns(None)
        with (
            _classify_returns({"intent": "lines_slate", "team": None, "subject": "the spread"}),
            _tokens("KC"),
            seam_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("what's the spread?", discord_id=7))
        self.assertEqual(out, qa._SOFT_DECLINE_FACT)
        self.assertEqual(seam_calls, [])  # stateless: no reader call, no pending slot

    def test_scores_routes_to_week_scores_reader(self) -> None:
        seam_patch, seam_calls = _seam(
            "get_week_scores_async",
            {
                "week": 3,
                "games": [
                    {
                        "away": "LAC",
                        "home": "KC",
                        "away_score": 20,
                        "home_score": 27,
                        "status": "FINAL",
                    }
                ],
            },
        )
        phrase_patch, _ = _phrase_returns(None)
        with (
            _classify_returns({"intent": "scores"}),
            _tokens("KC"),
            seam_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("what's the score?", discord_id=7))
        self.assertEqual(len(seam_calls), 1)
        self.assertIn("27", out)
        self.assertIn("final", out)

    def test_coming_soon_is_tier2_wink_no_db_read(self) -> None:
        # A recognized-but-planned topic — no DB read, no capability menu.
        phrase_patch, _ = _phrase_returns(None)
        with _classify_returns({"intent": "coming_soon"}), _tokens("KC"), _voice(), phrase_patch:
            out = _run(qa.answer_question("any injuries this week?", discord_id=7))
        self.assertEqual(out, qa._COMING_SOON_FACT)
        # The capability menu must NOT appear on coming_soon.
        self.assertNotIn("bug the developer", out)

    def test_unknown_is_tier3_decline_with_capability_menu(self) -> None:
        phrase_patch, _ = _phrase_returns(None)
        with _classify_returns({"intent": "unknown"}), _tokens("KC"), _voice(), phrase_patch:
            out = _run(qa.answer_question("banana helicopter?", discord_id=7))
        self.assertEqual(out, qa._UNKNOWN_FACT)
        # The capability-menu + bug-the-dev nudge appears ONLY on unknown.
        self.assertIn("bug the developer", out)


class BestEffortTests(unittest.TestCase):
    def test_falls_back_to_fact_when_phrase_returns_none(self) -> None:
        seam_patch, _ = _seam(
            "get_leaders_context_async",
            {
                "leader": "Ada",
                "leader_total": 40,
                "runner_up": None,
                "runner_up_total": None,
                "gap": None,
            },
        )
        phrase_patch, _ = _phrase_returns(None)
        with (
            _classify_returns({"intent": "standings"}),
            _tokens("KC"),
            seam_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("standings?", discord_id=7))
        # Exactly one line lands — the deterministic fact itself.
        self.assertIn("Ada", out)
        self.assertIn("leads the season", out)

    def test_never_raises_when_a_seam_raises(self) -> None:
        seam_patch, _ = _seam("get_leaders_context_async", raises=True)
        phrase_patch, _ = _phrase_returns("unused")
        with (
            _classify_returns({"intent": "standings"}),
            _tokens("KC"),
            seam_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("standings?", discord_id=7))
        # No exception escaped; a deterministic line is returned.
        self.assertEqual(out, qa._ERROR_LINE)

    def test_never_raises_when_token_seam_raises(self) -> None:
        async def _boom():
            raise RuntimeError("tokens exploded")

        phrase_patch, _ = _phrase_returns("unused")
        with (
            _classify_returns({"intent": "standings"}),
            mock.patch.object(db_bridge, "get_real_team_tokens_async", _boom),
            phrase_patch,
        ):
            out = _run(qa.answer_question("standings?", discord_id=7))
        self.assertEqual(out, qa._ERROR_LINE)


class ListAnswerAndFormattingTests(unittest.TestCase):
    """List intents keep the full block; close times are formatted + tense-correct."""

    def test_multi_game_slate_appends_full_deterministic_block(self) -> None:
        # A whole-slate question: the header is phrased, but EVERY game must survive
        # verbatim (the one-line phrasing guard must not summarize the list away).
        slate = {
            "week": 1,
            "close_at": datetime(2026, 7, 6, 12, 22, tzinfo=timezone.utc),
            "pick_open": False,
            "games": [
                {"away": "DAL", "home": "PHI", "favorite": "PHI", "spread": "7.5", "total": "47.5"},
                {"away": "KC", "home": "LAC", "favorite": "KC", "spread": "3.5", "total": "47.5"},
            ],
        }
        seam_patch, _ = _seam("get_lines_slate_async", slate)
        phrase_patch, calls = _phrase_returns("Week 1's carnage 👇")
        with (
            _classify_returns({"intent": "lines_slate", "team": None}),
            _tokens("KC"),
            seam_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("what games are on this week?", discord_id=7))
        # Phrased header on top, then both games, verbatim.
        self.assertTrue(out.startswith("Week 1's carnage 👇"))
        self.assertIn("DAL @ PHI — PHI -7.5 (O/U 47.5)", out)
        self.assertIn("KC @ LAC — KC -3.5 (O/U 47.5)", out)
        # ONLY the short header is phrased (not the game list), tense-correct + formatted.
        self.assertIn("closed", calls[0]["fact"])
        self.assertIn("12:22 PM UTC", calls[0]["fact"])
        self.assertNotIn("DAL", calls[0]["fact"])

    def test_multi_game_scores_appends_full_scoreboard(self) -> None:
        scores = {
            "week": 1,
            "games": [
                {
                    "away": "DAL",
                    "home": "PHI",
                    "away_score": 20,
                    "home_score": 24,
                    "status": "FINAL",
                },
                {
                    "away": "BAL",
                    "home": "BUF",
                    "away_score": 40,
                    "home_score": 41,
                    "status": "IN_PROGRESS",
                },
            ],
        }
        seam_patch, _ = _seam("get_week_scores_async", scores)
        phrase_patch, calls = _phrase_returns("Scoreboard 👇")
        with (
            _classify_returns({"intent": "scores"}),
            _tokens("KC"),
            seam_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("what are the scores?", discord_id=7))
        self.assertTrue(out.startswith("Scoreboard 👇"))
        self.assertIn("DAL 20 @ PHI 24 (final)", out)
        self.assertIn("BAL 40 @ BUF 41 (in progress)", out)
        self.assertIn("1 final, 1 in progress", calls[0]["fact"])

    def test_single_game_slate_formats_close_time_and_open_tense(self) -> None:
        slate = {
            "week": 3,
            "close_at": datetime(2026, 7, 6, 12, 22, 31, 79408, tzinfo=timezone.utc),
            "pick_open": True,
            "games": [
                {"away": "LAC", "home": "KC", "favorite": "KC", "spread": "3.5", "total": "47.5"}
            ],
        }
        seam_patch, _ = _seam("get_lines_slate_async", slate)
        phrase_patch, _ = _phrase_returns(None)  # fall back to the deterministic fact
        with (
            _classify_returns({"intent": "lines_slate", "team": "Chiefs"}),
            _tokens("KC", "CHIEFS"),
            seam_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("what's the chiefs line?", discord_id=7))
        self.assertIn("12:22 PM UTC", out)
        self.assertIn("Picks close", out)  # window still open -> present tense
        self.assertNotIn("Picks closed", out)
        self.assertNotIn("079408", out)  # no raw microseconds
        self.assertNotIn("+00", out)  # no raw offset

    def test_pick_status_closed_window_reads_as_locked(self) -> None:
        seam_patch, _ = _seam(
            "get_pick_status_async",
            {
                "registered": True,
                "display_name": "Ada",
                "complete": False,
                "remaining_labels": ["over", "mortal lock"],
                "pick_open": False,
            },
        )
        phrase_patch, calls = _phrase_returns(None)
        with (
            _classify_returns({"intent": "pick_status"}),
            _tokens("KC"),
            seam_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("am I locked in?", discord_id=7))
        # Window closed + incomplete: a SHORT verdict (locked + incomplete), NOT a
        # to-do and NOT an unactionable slot enumeration (which the one-line phrasing
        # guard would trim away anyway — the bug this fixes).
        self.assertIn("locked", out)
        self.assertIn("incomplete", out)
        self.assertNotIn("still needs to make", out)
        self.assertNotIn("over", calls[0]["fact"])  # no slot list when the window's closed

    def test_pick_status_closed_window_complete_reads_as_locked_in(self) -> None:
        seam_patch, _ = _seam(
            "get_pick_status_async",
            {
                "registered": True,
                "display_name": "Ada",
                "complete": True,
                "remaining_labels": [],
                "pick_open": False,
            },
        )
        phrase_patch, _ = _phrase_returns(None)
        with (
            _classify_returns({"intent": "pick_status"}),
            _tokens("KC"),
            seam_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("am I locked in?", discord_id=7))
        self.assertIn("locked in", out)
        self.assertIn("full card", out)

    def test_fmt_when_is_clean_and_none_safe(self) -> None:
        s = qa._fmt_when(datetime(2026, 7, 6, 12, 22, 31, 79408, tzinfo=timezone.utc))
        assert s is not None
        self.assertIn("Jul 6", s)
        self.assertIn("12:22 PM UTC", s)
        self.assertNotIn("079408", s)
        self.assertNotIn("+00", s)
        # 12-hour edges.
        midnight = qa._fmt_when(datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))
        afternoon = qa._fmt_when(datetime(2026, 1, 1, 13, 5, tzinfo=timezone.utc))
        assert midnight is not None and afternoon is not None
        self.assertIn("12:00 AM UTC", midnight)
        self.assertIn("1:05 PM UTC", afternoon)
        # Non-datetime / None -> None (never raises).
        self.assertIsNone(qa._fmt_when(None))
        self.assertIsNone(qa._fmt_when("2026-07-06"))


class InjuriesIntentTests(unittest.TestCase):
    """The Path-B injuries intent: routes to ESPN via the espn_extra seam, builds a
    deterministic FACT, and NEVER invents an injury on any resolution/fetch failure."""

    def _kc_multi_payload(self) -> dict:
        # A summary carrying BOTH teams; KC has two players, LAC (the other team) one.
        return {
            "injuries": [
                {
                    "team": {"abbreviation": "KC"},
                    "injuries": [
                        {
                            "status": "Out",
                            "date": "2026-01-05T18:00Z",
                            "athlete": {
                                "displayName": "Player One",
                                "position": {"abbreviation": "RB"},
                            },
                            "details": {"type": "Knee", "returnDate": "2026-01-19"},
                        },
                        {
                            "status": "Questionable",
                            "date": "2026-01-05T18:00Z",
                            "athlete": {
                                "displayName": "Player Two",
                                "position": {"abbreviation": "WR"},
                            },
                            "details": {"type": "Ankle"},
                        },
                    ],
                },
                {
                    "team": {"abbreviation": "LAC"},
                    "injuries": [{"status": "Out", "athlete": {"displayName": "Other Team Guy"}}],
                },
            ]
        }

    def test_teamless_injuries_soft_declines_no_event_lookup(self) -> None:
        # A teamless injuries question -> soft-decline, NO event lookup, no HTTP.
        seam_patch, seam_calls = _seam("get_injuries_event_id_async", (123, "KC"))
        fetch_patch, fetch_calls = _fetch_injuries_returns({"unused": True})
        phrase_patch, _ = _phrase_returns(None)
        with (
            _classify_returns({"intent": "injuries", "team": None}),
            _tokens("KC", "CHIEFS"),
            seam_patch,
            fetch_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("any injuries this week?", discord_id=7))
        self.assertEqual(out, qa._INJURIES_NO_TEAM_FACT)
        self.assertEqual(seam_calls, [])  # stateless: no event lookup
        self.assertEqual(fetch_calls, [])  # and no HTTP

    def test_team_resolves_multi_player_yields_full_list_answer(self) -> None:
        # The whole roster must survive: header phrased, every player verbatim, and
        # the OTHER team's player filtered out.
        seam_patch, seam_calls = _seam("get_injuries_event_id_async", (999, "KC"))
        fetch_patch, fetch_calls = _fetch_injuries_returns(self._kc_multi_payload())
        phrase_patch, calls = _phrase_returns("KC banged up 👇")
        with (
            _classify_returns({"intent": "injuries", "team": "Chiefs"}),
            _tokens("KC", "CHIEFS"),
            seam_patch,
            fetch_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("any injuries for the Chiefs?", discord_id=7))
        # The event id resolved by the seam is what the fetch was called with.
        self.assertEqual(seam_calls[0]["args"], ("CHIEFS",))
        self.assertEqual(fetch_calls[0]["args"], (999,))
        # Phrased header on top, then BOTH KC players verbatim (list not trimmed).
        self.assertTrue(out.startswith("KC banged up 👇"))
        self.assertIn("Player One (RB): Out — Knee, expected back 2026-01-19", out)
        self.assertIn("Player Two (WR): Questionable — Ankle", out)
        # Team-filtered: the other team's injured player NEVER appears.
        self.assertNotIn("Other Team Guy", out)
        # ONLY the short header is phrased (not the player block).
        self.assertIn("2 listed", calls[0]["fact"])
        self.assertNotIn("Player One", calls[0]["fact"])

    def test_team_resolves_single_player_is_one_liner(self) -> None:
        payload = {
            "injuries": [
                {
                    "team": {"abbreviation": "KC"},
                    "injuries": [
                        {
                            "status": "Out",
                            "date": "2026-01-05T18:00Z",
                            "athlete": {
                                "displayName": "Solo Guy",
                                "position": {"abbreviation": "QB"},
                            },
                            "details": {"type": "Shoulder"},
                        }
                    ],
                }
            ]
        }
        seam_patch, _ = _seam("get_injuries_event_id_async", (7, "KC"))
        fetch_patch, _ = _fetch_injuries_returns(payload)
        phrase_patch, _ = _phrase_returns(None)  # fall back to the deterministic fact
        with (
            _classify_returns({"intent": "injuries", "team": "Chiefs"}),
            _tokens("KC", "CHIEFS"),
            seam_patch,
            fetch_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("is anyone hurt on KC?", discord_id=7))
        self.assertIn("Solo Guy (QB): Out — Shoulder", out)
        self.assertIn("as of 2026-01-05T18:00Z", out)

    def test_team_resolves_no_injuries_returns_clean_line(self) -> None:
        payload = {"injuries": [{"team": {"abbreviation": "KC"}, "injuries": []}]}
        seam_patch, _ = _seam("get_injuries_event_id_async", (1, "KC"))
        fetch_patch, _ = _fetch_injuries_returns(payload)
        phrase_patch, _ = _phrase_returns(None)
        with (
            _classify_returns({"intent": "injuries", "team": "Chiefs"}),
            _tokens("KC", "CHIEFS"),
            seam_patch,
            fetch_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("any injuries for the Chiefs?", discord_id=7))
        self.assertIn("No injuries listed for KC", out)

    def test_event_id_unresolved_degrades_without_inventing(self) -> None:
        # No game/event resolved -> the degrade line, and the fetch is NEVER attempted.
        seam_patch, _ = _seam("get_injuries_event_id_async", None)
        fetch_patch, fetch_calls = _fetch_injuries_returns({"should": "not be used"})
        phrase_patch, _ = _phrase_returns(None)
        with (
            _classify_returns({"intent": "injuries", "team": "Chiefs"}),
            _tokens("KC", "CHIEFS"),
            seam_patch,
            fetch_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("any injuries for the Chiefs?", discord_id=7))
        self.assertEqual(out, qa._INJURIES_DEGRADE_FACT)
        self.assertEqual(fetch_calls, [])  # no HTTP when the event can't be resolved

    def test_fetch_failure_degrades_without_inventing(self) -> None:
        # The event resolves but ESPN is down (fetch None) -> degrade, never an injury.
        seam_patch, _ = _seam("get_injuries_event_id_async", (5, "KC"))
        fetch_patch, fetch_calls = _fetch_injuries_returns(None)
        phrase_patch, _ = _phrase_returns(None)
        with (
            _classify_returns({"intent": "injuries", "team": "Chiefs"}),
            _tokens("KC", "CHIEFS"),
            seam_patch,
            fetch_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("any injuries for the Chiefs?", discord_id=7))
        self.assertEqual(out, qa._INJURIES_DEGRADE_FACT)
        self.assertEqual(fetch_calls[0]["args"], (5,))  # fetch was attempted


class WeatherIntentTests(unittest.TestCase):
    """The Path-B weather intent: resolves the game's HOME stadium, fetches Open-Meteo
    for an OUTDOOR game, short-circuits a DOME with no fetch, and NEVER invents a
    forecast on any resolution/fetch/parse failure."""

    def _kickoff(self) -> datetime:
        return datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)

    def _outdoor(self) -> weather.Stadium:
        return weather.Stadium("Test Field", 42.77, -78.79, False)

    def _indoor(self) -> weather.Stadium:
        return weather.Stadium("Test Dome", 30.0, -90.0, True)

    def test_teamless_weather_soft_declines_no_seam_no_fetch(self) -> None:
        seam_patch, seam_calls = _seam("get_weather_target_async", ("BUF", self._kickoff()))
        fetch_patch, fetch_calls = _fetch_forecast_returns({"unused": True})
        phrase_patch, _ = _phrase_returns(None)
        with (
            _classify_returns({"intent": "weather", "team": None}),
            _tokens("KC", "CHIEFS"),
            seam_patch,
            fetch_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("what's the weather this week?", discord_id=7))
        self.assertEqual(out, qa._WEATHER_NO_TEAM_FACT)
        self.assertEqual(seam_calls, [])  # stateless: no weather-target lookup
        self.assertEqual(fetch_calls, [])  # and no HTTP

    def test_indoor_stadium_short_circuits_with_no_fetch(self) -> None:
        seam_patch, seam_calls = _seam("get_weather_target_async", ("NO", self._kickoff()))
        lookup_patch, lookup_calls = _lookup_returns(self._indoor())
        fetch_patch, fetch_calls = _fetch_forecast_returns({"should": "not be used"})
        phrase_patch, _ = _phrase_returns(None)  # fall back to the deterministic fact
        with (
            _classify_returns({"intent": "weather", "team": "Saints"}),
            _tokens("NO", "SAINTS"),
            seam_patch,
            lookup_patch,
            fetch_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("weather for the Saints game?", discord_id=7))
        # The deterministic dome line names the stadium; NO forecast fetch happened.
        self.assertIn("Test Dome", out)
        self.assertIn("covered dome", out)
        self.assertEqual(seam_calls[0]["args"], ("SAINTS",))
        self.assertEqual(lookup_calls, ["NO"])  # looked up the HOME abbr
        self.assertEqual(fetch_calls, [])  # dome short-circuit — never fetched

    def test_outdoor_good_forecast_builds_fact_with_values(self) -> None:
        seam_patch, seam_calls = _seam("get_weather_target_async", ("BUF", self._kickoff()))
        lookup_patch, _ = _lookup_returns(self._outdoor())
        fetch_patch, fetch_calls = _fetch_forecast_returns({"hourly": {"time": []}})
        parse_patch, _ = _parse_forecast_returns(
            {
                "temperature_f": 34.0,
                "wind_mph": 12.0,
                "precip_in": 0.05,
                "hour": "2026-01-05T14:00",
            }
        )
        phrase_patch, calls = _phrase_returns(None)  # fall back to the deterministic fact
        with (
            _classify_returns({"intent": "weather", "team": "Bills"}),
            _tokens("BUF", "BILLS"),
            seam_patch,
            lookup_patch,
            fetch_patch,
            parse_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("weather for the Bills game?", discord_id=7))
        # The weather-target seam was called with the asked team token.
        self.assertEqual(seam_calls[0]["args"], ("BILLS",))
        # The forecast fetch was called with the stadium's lat/lon.
        self.assertEqual(fetch_calls[0]["args"], (42.77, -78.79))
        # The deterministic FACT names temp / wind / precip at kickoff.
        self.assertIn("Test Field at kickoff", out)
        self.assertIn("34.0°F", out)
        self.assertIn("wind 12.0 mph", out)
        self.assertIn("0.05 in precip", out)
        self.assertIn("2026-01-05T14:00 GMT", out)
        # The fact was the thing phrased (deterministic fallback path).
        self.assertIn("34.0°F", calls[0]["fact"])

    def test_zero_precip_reads_as_no_precip_expected(self) -> None:
        seam_patch, _ = _seam("get_weather_target_async", ("BUF", self._kickoff()))
        lookup_patch, _ = _lookup_returns(self._outdoor())
        fetch_patch, _ = _fetch_forecast_returns({"hourly": {"time": []}})
        parse_patch, _ = _parse_forecast_returns(
            {"temperature_f": 50.0, "wind_mph": 5.0, "precip_in": 0.0, "hour": "2026-01-05T14:00"}
        )
        phrase_patch, _ = _phrase_returns(None)
        with (
            _classify_returns({"intent": "weather", "team": "Bills"}),
            _tokens("BUF", "BILLS"),
            seam_patch,
            lookup_patch,
            fetch_patch,
            parse_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("weather for the Bills game?", discord_id=7))
        self.assertIn("no precip expected", out)
        self.assertNotIn("in precip", out)

    def test_unresolved_game_degrades_without_fetch(self) -> None:
        seam_patch, _ = _seam("get_weather_target_async", None)
        lookup_patch, lookup_calls = _lookup_returns(self._outdoor())
        fetch_patch, fetch_calls = _fetch_forecast_returns({"should": "not be used"})
        phrase_patch, _ = _phrase_returns(None)
        with (
            _classify_returns({"intent": "weather", "team": "Bills"}),
            _tokens("BUF", "BILLS"),
            seam_patch,
            lookup_patch,
            fetch_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("weather for the Bills game?", discord_id=7))
        self.assertEqual(out, qa._WEATHER_DEGRADE_FACT)
        self.assertEqual(lookup_calls, [])  # never reached the stadium lookup
        self.assertEqual(fetch_calls, [])  # and no HTTP

    def test_missing_stadium_row_degrades_without_fetch(self) -> None:
        seam_patch, _ = _seam("get_weather_target_async", ("BUF", self._kickoff()))
        lookup_patch, _ = _lookup_returns(None)  # no table row
        fetch_patch, fetch_calls = _fetch_forecast_returns({"should": "not be used"})
        phrase_patch, _ = _phrase_returns(None)
        with (
            _classify_returns({"intent": "weather", "team": "Bills"}),
            _tokens("BUF", "BILLS"),
            seam_patch,
            lookup_patch,
            fetch_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("weather for the Bills game?", discord_id=7))
        self.assertEqual(out, qa._WEATHER_DEGRADE_FACT)
        self.assertEqual(fetch_calls, [])  # no fetch when the stadium is unknown

    def test_fetch_none_degrades_without_inventing(self) -> None:
        seam_patch, _ = _seam("get_weather_target_async", ("BUF", self._kickoff()))
        lookup_patch, _ = _lookup_returns(self._outdoor())
        fetch_patch, fetch_calls = _fetch_forecast_returns(None)  # Open-Meteo down
        phrase_patch, _ = _phrase_returns(None)
        with (
            _classify_returns({"intent": "weather", "team": "Bills"}),
            _tokens("BUF", "BILLS"),
            seam_patch,
            lookup_patch,
            fetch_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("weather for the Bills game?", discord_id=7))
        self.assertEqual(out, qa._WEATHER_DEGRADE_FACT)
        self.assertEqual(fetch_calls[0]["args"], (42.77, -78.79))  # fetch was attempted

    def test_parse_none_degrades_without_inventing(self) -> None:
        seam_patch, _ = _seam("get_weather_target_async", ("BUF", self._kickoff()))
        lookup_patch, _ = _lookup_returns(self._outdoor())
        fetch_patch, _ = _fetch_forecast_returns({"hourly": {"time": []}})
        parse_patch, parse_calls = _parse_forecast_returns(None)  # kickoff hour absent
        phrase_patch, _ = _phrase_returns(None)
        with (
            _classify_returns({"intent": "weather", "team": "Bills"}),
            _tokens("BUF", "BILLS"),
            seam_patch,
            lookup_patch,
            fetch_patch,
            parse_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("weather for the Bills game?", discord_id=7))
        self.assertEqual(out, qa._WEATHER_DEGRADE_FACT)
        self.assertEqual(len(parse_calls), 1)  # parse was attempted, returned None


class NewsIntentTests(unittest.TestCase):
    """The Path-B news intent: team-OPTIONAL, headlines relayed VERBATIM through the
    _ListAnswer body and NEVER passed to the LLM (the no-rephrasing invariant), with a
    fixed honest miss on any failure and a concrete empty line on nothing-to-show."""

    # A distinctive KC headline whose EXACT text the no-rephrasing regression asserts
    # survives unchanged in the reply and is absent from the phrased header.
    _KC_HEADLINE = "Chiefs clinch the AFC West with a road rout of the Broncos"
    _DEN_HEADLINE = "Broncos fire their offensive coordinator after the blowout"

    def _payload(self) -> dict:
        # A league news page carrying one KC-tagged and one Denver-tagged article; the
        # REAL espn_extra.parse_news filters this client-side via categories[].team.
        return {
            "articles": [
                {
                    "headline": self._KC_HEADLINE,
                    "description": "Kansas City wraps up the division.",
                    "published": "2026-01-05T18:00Z",
                    "links": [{"href": "https://www.espn.com/nfl/story/kc"}],
                    "categories": [
                        {
                            "type": "team",
                            "description": "Kansas City Chiefs",
                            "team": {"abbreviation": "KC", "displayName": "Kansas City Chiefs"},
                        }
                    ],
                },
                {
                    "headline": self._DEN_HEADLINE,
                    "description": "Denver shakes up its staff.",
                    "published": "2026-01-04T12:00Z",
                    "links": [{"href": "https://www.espn.com/nfl/story/den"}],
                    "categories": [
                        {
                            "type": "team",
                            "description": "Denver Broncos",
                            "team": {"abbreviation": "DEN", "displayName": "Denver Broncos"},
                        }
                    ],
                },
            ]
        }

    def test_team_resolves_headlines_relayed_verbatim_never_phrased(self) -> None:
        seam_patch, seam_calls = _seam("get_news_team_filter_async", ("KC", "Kansas City Chiefs"))
        fetch_patch, fetch_calls = _fetch_news_returns(self._payload())
        phrase_patch, calls = _phrase_returns("Fresh off the wire 👇")
        with (
            _classify_returns({"intent": "news", "team": "Chiefs"}),
            _tokens("KC", "CHIEFS"),
            seam_patch,
            fetch_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("any Chiefs news?", discord_id=7))
        # The team-topic seam was called with the asked (validated) team token.
        self.assertEqual(seam_calls[0]["args"], ("CHIEFS",))
        self.assertEqual(len(fetch_calls), 1)  # the league page was fetched once
        # Fixed deterministic wrapper on top, then the KC headline VERBATIM, rendered
        # as a clickable masked link to the ESPN source (headline text unchanged).
        self.assertTrue(out.startswith("Latest on KC (ESPN"))
        self.assertIn(self._KC_HEADLINE, out)
        self.assertIn(f"[{self._KC_HEADLINE}](https://www.espn.com/nfl/story/kc)", out)
        # THE NO-REPHRASING REGRESSION (absolute): the whole news answer — wrapper AND
        # headlines — is deterministic; the LLM is NEVER called, so nothing can be
        # rephrased or inverted.
        self.assertEqual(calls, [])
        # Client-side team filter: the non-KC headline NEVER appears.
        self.assertNotIn(self._DEN_HEADLINE, out)

    def test_teamless_returns_league_headlines_no_seam_call(self) -> None:
        seam_patch, seam_calls = _seam("get_news_team_filter_async", ("KC", "Kansas City Chiefs"))
        fetch_patch, fetch_calls = _fetch_news_returns(self._payload())
        phrase_patch, calls = _phrase_returns("Around the league 👇")
        with (
            _classify_returns({"intent": "news", "team": None}),
            _tokens("KC", "CHIEFS"),
            seam_patch,
            fetch_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("any NFL news?", discord_id=7))
        # A null team is a VALID answer: the team-topic seam is NEVER called.
        self.assertEqual(seam_calls, [])
        self.assertEqual(len(fetch_calls), 1)
        # Both league headlines land verbatim under the fixed deterministic league
        # header, each a clickable masked link to its ESPN source.
        self.assertTrue(out.startswith("Latest NFL headlines (ESPN"))
        self.assertIn(f"[{self._KC_HEADLINE}](https://www.espn.com/nfl/story/kc)", out)
        self.assertIn(f"[{self._DEN_HEADLINE}](https://www.espn.com/nfl/story/den)", out)
        # Deterministic wrapper: the LLM is never called for a news answer.
        self.assertEqual(calls, [])

    def test_empty_league_news_is_concrete_empty_line_not_failure(self) -> None:
        seam_patch, _ = _seam("get_news_team_filter_async", ("KC", "Kansas City Chiefs"))
        fetch_patch, _ = _fetch_news_returns({"articles": []})
        phrase_patch, _ = _phrase_returns(None)
        with (
            _classify_returns({"intent": "news", "team": None}),
            _tokens("KC", "CHIEFS"),
            seam_patch,
            fetch_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("any NFL news?", discord_id=7))
        self.assertEqual(out, qa._NEWS_EMPTY_LEAGUE_FACT)
        self.assertIn("No fresh NFL headlines", out)
        self.assertNotEqual(out, qa._NEWS_DEGRADE_FACT)  # concrete empty, not the miss

    def test_team_filtered_to_nothing_is_concrete_team_empty_line(self) -> None:
        # A resolved team whose filter matches nothing -> a concrete team empty line.
        seam_patch, _ = _seam("get_news_team_filter_async", ("SF", "San Francisco 49ers"))
        fetch_patch, _ = _fetch_news_returns(self._payload())
        phrase_patch, _ = _phrase_returns(None)
        with (
            _classify_returns({"intent": "news", "team": "49ers"}),
            _tokens("SF", "49ERS"),
            seam_patch,
            fetch_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("any 49ers news?", discord_id=7))
        self.assertIn("No fresh ESPN headlines on SF", out)
        self.assertNotIn(self._KC_HEADLINE, out)  # never an invented/wrong headline

    def test_fetch_none_degrades_without_inventing(self) -> None:
        seam_patch, _ = _seam("get_news_team_filter_async", ("KC", "Kansas City Chiefs"))
        fetch_patch, fetch_calls = _fetch_news_returns(None)  # ESPN down
        phrase_patch, _ = _phrase_returns(None)
        with (
            _classify_returns({"intent": "news", "team": "Chiefs"}),
            _tokens("KC", "CHIEFS"),
            seam_patch,
            fetch_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("any Chiefs news?", discord_id=7))
        self.assertEqual(out, qa._NEWS_DEGRADE_FACT)
        self.assertEqual(len(fetch_calls), 1)  # fetch was attempted, returned None

    def test_unresolvable_team_degrades_without_fetch(self) -> None:
        seam_patch, _ = _seam("get_news_team_filter_async", None)  # team un-resolvable
        fetch_patch, fetch_calls = _fetch_news_returns({"should": "not be used"})
        phrase_patch, _ = _phrase_returns(None)
        with (
            _classify_returns({"intent": "news", "team": "Chiefs"}),
            _tokens("KC", "CHIEFS"),
            seam_patch,
            fetch_patch,
            _voice(),
            phrase_patch,
        ):
            out = _run(qa.answer_question("any Chiefs news?", discord_id=7))
        self.assertEqual(out, qa._NEWS_DEGRADE_FACT)
        self.assertEqual(fetch_calls, [])  # no HTTP when the team can't be resolved


if __name__ == "__main__":
    unittest.main()
