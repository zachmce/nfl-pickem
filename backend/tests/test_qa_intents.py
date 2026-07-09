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


if __name__ == "__main__":
    unittest.main()
