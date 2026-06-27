"""Offline unit tests for the best-effort chat personality layer (260627-t5u).

These tests NEVER touch a live LLM endpoint: ``chat_personality.llm_client.phrase``
is monkeypatched with an async fake that returns a canned line, ``None``, or raises.
They assert the Tier-1 contract: the three handled events
(``window.opened`` / ``game.final`` / ``roster.complete``) return the LLM line when
configured and the deterministic ``render_chat`` line on any failure — never
``None``, never a raise. ``window.closed`` / ``week.recap`` / unknown types return
``None`` (the notifier owns those via the existing path). It also pins the two HARD
rules: the ``game.final`` margin descriptor is COMPUTED (not invented) and the
``roster.complete`` fact carries NO pick content (LEAK-SAFE).

Run with: ``backend/.venv/bin/python -m unittest tests.test_chat_personality -v``
(there is no bare ``python`` on PATH on this machine).
"""

from __future__ import annotations

import asyncio
import unittest
from unittest import mock

from app.bot import chat_personality
from app.bot.notifier import render_chat
from app.services.notifications import (
    game_final_event,
    roster_complete_event,
    week_recap_event,
    window_closed_event,
    window_opened_event,
)


def _run(coro):
    return asyncio.run(coro)


def _phrase_returns(value):
    """Patch the module's phrase() to an async fn returning ``value``, recording
    the fact + system_prompt it was called with for assertions."""
    calls: list[dict] = []

    async def _fake(fact_text, *, system_prompt):
        calls.append({"fact": fact_text, "system_prompt": system_prompt})
        return value

    return mock.patch.object(chat_personality.llm_client, "phrase", _fake), calls


# Pick-content tokens that must NEVER appear in a roster.complete fact (LEAK-SAFE).
_PICK_TOKENS = [
    "over",
    "under",
    "favorite",
    "underdog",
    "spread",
    "cover",
    "moneyline",
    "mortal",
    "lock",
    "slot",
    "pick",
]


class EmbellishChatHandledTypesTests(unittest.TestCase):
    """The three Tier-1 events return the LLM line when present, the deterministic
    render_chat line on None — always a non-None string."""

    def test_window_opened_returns_llm_line_when_configured(self) -> None:
        event = window_opened_event(week=3)
        patcher, calls = _phrase_returns("LET'S GO WEEK 3 🏈")
        with patcher:
            out = _run(chat_personality.embellish_chat(event))
        self.assertEqual(out, "LET'S GO WEEK 3 🏈")
        self.assertEqual(len(calls), 1)
        self.assertIn("3", calls[0]["fact"])

    def test_window_opened_falls_back_to_render_chat_on_none(self) -> None:
        event = window_opened_event(week=3)
        patcher, _ = _phrase_returns(None)
        with patcher:
            out = _run(chat_personality.embellish_chat(event))
        self.assertEqual(out, render_chat(event))
        self.assertIsNotNone(out)

    def test_game_final_returns_llm_line_when_configured(self) -> None:
        event = game_final_event(
            week=3, away_abbr="LAC", home_abbr="KC", away_score=20, home_score=27
        )
        patcher, calls = _phrase_returns("KC squeaks it out 🔥")
        with patcher:
            out = _run(chat_personality.embellish_chat(event))
        self.assertEqual(out, "KC squeaks it out 🔥")
        # The fact carries both abbrs + both scores (display-only).
        fact = calls[0]["fact"]
        self.assertIn("KC", fact)
        self.assertIn("LAC", fact)
        self.assertIn("27", fact)
        self.assertIn("20", fact)

    def test_game_final_falls_back_to_render_chat_on_none(self) -> None:
        event = game_final_event(
            week=3, away_abbr="LAC", home_abbr="KC", away_score=20, home_score=27
        )
        patcher, _ = _phrase_returns(None)
        with patcher:
            out = _run(chat_personality.embellish_chat(event))
        self.assertEqual(out, render_chat(event))

    def test_roster_complete_returns_llm_line_when_configured(self) -> None:
        event = roster_complete_event(actor="Bob", week=3)
        patcher, calls = _phrase_returns("Bob's all in for Week 3 👀")
        with patcher:
            out = _run(chat_personality.embellish_chat(event))
        self.assertEqual(out, "Bob's all in for Week 3 👀")
        self.assertIn("Bob", calls[0]["fact"])
        self.assertIn("3", calls[0]["fact"])

    def test_roster_complete_falls_back_to_render_chat_on_none(self) -> None:
        event = roster_complete_event(actor="Bob", week=3)
        patcher, _ = _phrase_returns(None)
        with patcher:
            out = _run(chat_personality.embellish_chat(event))
        self.assertEqual(out, render_chat(event))


class EmbellishChatDescriptorTests(unittest.TestCase):
    """The game.final margin descriptor is COMPUTED from abs(score diff), not
    invented — assert the chosen word appears in the fact handed to the LLM."""

    def test_blowout_descriptor_for_large_margin(self) -> None:
        # 27 - 3 = 24-point margin -> blowout.
        event = game_final_event(
            week=3, away_abbr="LAC", home_abbr="KC", away_score=3, home_score=27
        )
        patcher, calls = _phrase_returns("x")
        with patcher:
            _run(chat_personality.embellish_chat(event))
        self.assertIn("blowout", calls[0]["fact"].lower())

    def test_nail_biter_descriptor_for_small_margin(self) -> None:
        # 24 - 23 = 1-point margin -> nail-biter.
        event = game_final_event(
            week=3, away_abbr="LAC", home_abbr="KC", away_score=23, home_score=24
        )
        patcher, calls = _phrase_returns("x")
        with patcher:
            _run(chat_personality.embellish_chat(event))
        self.assertIn("nail-biter", calls[0]["fact"].lower())

    def test_descriptor_helper_is_pure_and_computed(self) -> None:
        self.assertEqual(chat_personality._final_descriptor(27, 3), "blowout")
        self.assertEqual(chat_personality._final_descriptor(24, 23), "nail-biter")
        # A middling margin gets neither extreme word.
        mid = chat_personality._final_descriptor(24, 14)
        self.assertNotIn(mid, ("blowout", "nail-biter"))


class EmbellishChatLeakSafeTests(unittest.TestCase):
    """HARD rule: the roster.complete fact references ONLY actor + week — it can
    never carry a pick type or team abbreviation, because the event carries none."""

    def test_roster_fact_has_no_pick_content(self) -> None:
        event = roster_complete_event(actor="Bob", week=3)
        patcher, calls = _phrase_returns("x")
        with patcher:
            _run(chat_personality.embellish_chat(event))
        fact_lower = calls[0]["fact"].lower()
        for token in _PICK_TOKENS:
            self.assertNotIn(
                token, fact_lower, f"roster.complete fact leaked pick token: {token}"
            )


class EmbellishChatUnhandledTypesTests(unittest.TestCase):
    """window.closed / week.recap / unknown types are NOT this seam's job — they
    return None so the notifier keeps owning them via the existing path."""

    def test_window_closed_returns_none(self) -> None:
        patcher, calls = _phrase_returns("should-not-be-used")
        with patcher:
            out = _run(chat_personality.embellish_chat(window_closed_event(week=3)))
        self.assertIsNone(out)
        self.assertEqual(calls, [])  # no LLM call for an unhandled type

    def test_week_recap_returns_none(self) -> None:
        event = week_recap_event(
            week=3, winner="Carol", winner_score=6, leader="Dave", leader_score=18
        )
        patcher, calls = _phrase_returns("should-not-be-used")
        with patcher:
            out = _run(chat_personality.embellish_chat(event))
        self.assertIsNone(out)
        self.assertEqual(calls, [])

    def test_unknown_type_returns_none(self) -> None:
        patcher, _ = _phrase_returns("x")
        with patcher:
            out = _run(chat_personality.embellish_chat({"v": 1, "type": "totally.unknown"}))
        self.assertIsNone(out)


class EmbellishChatNeverRaisesTests(unittest.TestCase):
    """If the LLM client RAISES, it is caught and the deterministic render_chat
    line is returned — the notifier loop must never see an exception."""

    def test_llm_raise_falls_back_to_deterministic_line(self) -> None:
        event = window_opened_event(week=3)

        async def _boom(fact_text, *, system_prompt):
            raise RuntimeError("llm exploded")

        with mock.patch.object(chat_personality.llm_client, "phrase", _boom):
            out = _run(chat_personality.embellish_chat(event))
        self.assertEqual(out, render_chat(event))


# --------------------------------------------------------------------------- #
# 260627-vpc — enriched STATE-FACTS-FIRST embellishment from DB context.
# --------------------------------------------------------------------------- #


def _ctx_seam(attr: str, value):
    """Patch a chat_personality DB-context seam to an async fn returning ``value``.

    The enriched embellish_chat reads context through thin async seams
    (``_game_final_context`` / ``_roster_complete_context`` / ``_leaders_context``)
    so tests can inject a fixed context dict without a real db.
    """

    async def _fake(*args, **kwargs):
        return value

    return mock.patch.object(chat_personality, attr, _fake)


def _ctx_seam_raises(attr: str):
    """Patch a context seam to an async fn that RAISES (db-read failure)."""

    async def _boom(*args, **kwargs):
        raise RuntimeError("db read exploded")

    return mock.patch.object(chat_personality, attr, _boom)


class EmbellishChatEnrichedGameFinalTests(unittest.TestCase):
    """game.final FACT STATES teams + score + line result + a notable pick impact
    when the DB context resolves."""

    def _ctx(self) -> dict:
        return {
            "found": True,
            "away": "LAC",
            "home": "KC",
            "away_score": 20,
            "home_score": 27,
            "spread_result": {
                "favorite_abbr": "KC",
                "spread": "3.5",
                "did_cover": True,
            },
            "total_result": {"total": "44.5", "went_over": True},
            "pick_impacts": [
                {
                    "display_name": "Bob",
                    "side_label": "Underdog (LAC)",
                    "is_mortal_lock": True,
                    "outcome": "LOSS",
                }
            ],
        }

    def test_game_final_fact_states_score_line_and_impact(self) -> None:
        event = game_final_event(
            week=3, away_abbr="LAC", home_abbr="KC", away_score=20, home_score=27
        )
        patcher, calls = _phrase_returns("KC covers, Bob's lock busts 🔥")
        with _ctx_seam("_game_final_context", self._ctx()), patcher:
            out = _run(chat_personality.embellish_chat(event))
        self.assertEqual(out, "KC covers, Bob's lock busts 🔥")
        fact = calls[0]["fact"]
        # Teams + final score.
        self.assertIn("KC", fact)
        self.assertIn("LAC", fact)
        self.assertIn("27", fact)
        self.assertIn("20", fact)
        # Line result (spread cover) + a notable pick impact by display_name.
        self.assertIn("3.5", fact)
        self.assertIn("Bob", fact)

    def test_game_final_phrase_none_falls_back_to_render_chat(self) -> None:
        event = game_final_event(
            week=3, away_abbr="LAC", home_abbr="KC", away_score=20, home_score=27
        )
        patcher, _ = _phrase_returns(None)
        with _ctx_seam("_game_final_context", self._ctx()), patcher:
            out = _run(chat_personality.embellish_chat(event))
        self.assertEqual(out, render_chat(event))

    def test_game_final_not_found_context_uses_basic_fact(self) -> None:
        # When the context can't resolve the game, the basic event-field fact (the
        # scores from the event) is used — still phrased, never a raise.
        event = game_final_event(
            week=3, away_abbr="LAC", home_abbr="KC", away_score=20, home_score=27
        )
        not_found = {"found": False, "pick_impacts": [], "spread_result": None,
                     "total_result": None}
        patcher, calls = _phrase_returns("x")
        with _ctx_seam("_game_final_context", not_found), patcher:
            out = _run(chat_personality.embellish_chat(event))
        self.assertEqual(out, "x")
        self.assertIn("KC", calls[0]["fact"])
        self.assertIn("27", calls[0]["fact"])

    def test_game_final_db_read_raise_falls_back(self) -> None:
        event = game_final_event(
            week=3, away_abbr="LAC", home_abbr="KC", away_score=20, home_score=27
        )
        patcher, calls = _phrase_returns("x")
        with _ctx_seam_raises("_game_final_context"), patcher:
            out = _run(chat_personality.embellish_chat(event))
        # A db-read failure still produces a phrased line off the basic fact.
        self.assertEqual(out, "x")
        self.assertIn("KC", calls[0]["fact"])


class EmbellishChatEnrichedRosterCompleteTests(unittest.TestCase):
    """roster.complete FACT STATES the actor's rank + season total and the
    completion COUNT — never names of the outstanding, never pick content."""

    def test_roster_fact_states_rank_total_and_count(self) -> None:
        event = roster_complete_event(actor="Bob", week=3)
        ctx = {
            "actor": "Bob",
            "rank": 2,
            "season_total": 41,
            "completed_count": 3,
            "total_players": 5,
            "outstanding_count": 2,
        }
        patcher, calls = _phrase_returns("Bob's in at #2 👀")
        with _ctx_seam("_roster_complete_context", ctx), patcher:
            out = _run(chat_personality.embellish_chat(event))
        self.assertEqual(out, "Bob's in at #2 👀")
        fact = calls[0]["fact"]
        self.assertIn("Bob", fact)
        self.assertIn("41", fact)  # season total
        self.assertIn("2", fact)  # rank or outstanding count

    def test_roster_first_to_lock_in_wording(self) -> None:
        event = roster_complete_event(actor="Bob", week=3)
        ctx = {
            "actor": "Bob",
            "rank": 1,
            "season_total": 50,
            "completed_count": 1,
            "total_players": 5,
            "outstanding_count": 4,
        }
        patcher, calls = _phrase_returns("x")
        with _ctx_seam("_roster_complete_context", ctx), patcher:
            _run(chat_personality.embellish_chat(event))
        self.assertIn("first to lock in", calls[0]["fact"].lower())

    def test_roster_fact_is_leak_safe(self) -> None:
        # The enriched fact must carry the COUNT only — never an outstanding name
        # and never any pick-content token (the window is OPEN).
        event = roster_complete_event(actor="Bob", week=3)
        ctx = {
            "actor": "Bob",
            "rank": 2,
            "season_total": 41,
            "completed_count": 3,
            "total_players": 5,
            "outstanding_count": 2,
        }
        patcher, calls = _phrase_returns("x")
        with _ctx_seam("_roster_complete_context", ctx), patcher:
            _run(chat_personality.embellish_chat(event))
        fact_lower = calls[0]["fact"].lower()
        for token in _PICK_TOKENS:
            self.assertNotIn(
                token, fact_lower, f"roster.complete fact leaked pick token: {token}"
            )

    def test_roster_db_read_raise_falls_back_leak_safe(self) -> None:
        event = roster_complete_event(actor="Bob", week=3)
        patcher, calls = _phrase_returns("x")
        with _ctx_seam_raises("_roster_complete_context"), patcher:
            out = _run(chat_personality.embellish_chat(event))
        self.assertEqual(out, "x")
        # The basic fallback fact is still leak-safe.
        fact_lower = calls[0]["fact"].lower()
        for token in _PICK_TOKENS:
            self.assertNotIn(token, fact_lower)


class EmbellishChatEnrichedWindowOpenedTests(unittest.TestCase):
    """window.opened FACT STATES the season leader (+ runner-up + gap) by
    display_name and total."""

    def test_window_opened_fact_states_leader(self) -> None:
        event = window_opened_event(week=3)
        ctx = {
            "leader": "Carol",
            "leader_total": 52,
            "runner_up": "Dave",
            "runner_up_total": 47,
            "gap": 5,
        }
        patcher, calls = _phrase_returns("Carol leads — Week 3 is open! 🏈")
        with _ctx_seam("_leaders_context", ctx), patcher:
            out = _run(chat_personality.embellish_chat(event))
        self.assertEqual(out, "Carol leads — Week 3 is open! 🏈")
        fact = calls[0]["fact"]
        self.assertIn("Carol", fact)
        self.assertIn("52", fact)
        self.assertIn("Dave", fact)

    def test_window_opened_empty_leaders_uses_basic_fact(self) -> None:
        event = window_opened_event(week=3)
        ctx = {"leader": None, "leader_total": 0, "runner_up": None,
               "runner_up_total": None, "gap": None}
        patcher, calls = _phrase_returns("x")
        with _ctx_seam("_leaders_context", ctx), patcher:
            out = _run(chat_personality.embellish_chat(event))
        self.assertEqual(out, "x")
        self.assertIn("3", calls[0]["fact"])  # week number basic fact

    def test_window_opened_db_read_raise_falls_back(self) -> None:
        event = window_opened_event(week=3)
        patcher, calls = _phrase_returns("x")
        with _ctx_seam_raises("_leaders_context"), patcher:
            out = _run(chat_personality.embellish_chat(event))
        self.assertEqual(out, "x")
        self.assertIn("3", calls[0]["fact"])


if __name__ == "__main__":
    unittest.main()
