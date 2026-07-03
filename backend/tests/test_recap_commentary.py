"""Offline unit tests for the Tier-2 recap orchestrator (260627-tfb).

These tests NEVER touch a live LLM endpoint or a real db. ``recap.llm_client.phrase``
is monkeypatched with an async fake that returns a canned line or ``None``, and the
module's ``_recap_context`` indirection is monkeypatched with an async fake returning
a canned context dict (or raising). They pin the Tier-2 contract:

* :func:`app.bot.recap.build_week_recap` returns the LLM-phrased column when phrase
  succeeds; it narrates the deterministic fact built from the week's per-player
  scores + the season standings.
* phrase is called with ``system_prompt == recap.RECAP_PROMPT``.
* On phrase ``None``, a reader RAISE, or an EMPTY context, ``build_week_recap``
  returns the existing deterministic ``render_chat(event)`` one-liner and NEVER raises.

Run with: ``backend/.venv/bin/python -m unittest tests.test_recap_commentary -v``
(there is no bare ``python`` on PATH on this machine).
"""

from __future__ import annotations

import asyncio
import unittest
from unittest import mock

from app.bot import recap
from app.bot.notifier import render_chat
from app.services.notifications import week_recap_event


def _run(coro):
    return asyncio.run(coro)


# A populated, display-only recap context (mirrors get_recap_context output).
_CTX = {
    "week": 3,
    "weekly_scores": [
        {"display_name": "alice", "weekly_score": 9},
        {"display_name": "bob", "weekly_score": 2},
    ],
    "season_standings": [
        {"display_name": "alice", "season_total": 30, "rank": 1, "gap_to_leader": 0},
        {"display_name": "bob", "season_total": 18, "rank": 2, "gap_to_leader": 12},
    ],
}


# A populated context carrying a display-only storyline bundle (260703-jun).
_CTX_WITH_STORYLINES = {
    **_CTX,
    "storylines": [
        {"kind": "mortal_lock_streak", "text": "alice has missed their mortal lock 3 weeks running", "fresh": True},
        {"kind": "superlative", "text": "the biggest upset so far: T2 stunned T1 in Week 2", "fresh": False},
    ],
}


def _event():
    return week_recap_event(week=3, winner="alice", winner_score=9, leader="alice", leader_score=30)


def _phrase_returns(value):
    """Patch recap.llm_client.phrase to an async fn returning ``value``, recording
    the fact + system_prompt it was called with."""
    calls: list[dict] = []

    async def _fake(fact_text, *, system_prompt):
        calls.append({"fact": fact_text, "system_prompt": system_prompt})
        return value

    return mock.patch.object(recap.llm_client, "phrase", _fake), calls


def _context_returns(value):
    """Patch recap._recap_context to an async fn returning ``value``."""

    async def _fake(week):
        return value

    return mock.patch.object(recap, "_recap_context", _fake)


def _context_raises(exc):
    async def _fake(week):
        raise exc

    return mock.patch.object(recap, "_recap_context", _fake)


class BuildWeekRecapTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_phrased_column_when_configured(self) -> None:
        patcher, calls = _phrase_returns("Big week for alice...")
        with _context_returns(_CTX), patcher:
            out = await recap.build_week_recap(_event())
        self.assertEqual(out, "Big week for alice...")
        self.assertEqual(len(calls), 1)

    async def test_fact_carries_week_scores_and_standings(self) -> None:
        patcher, calls = _phrase_returns("x")
        with _context_returns(_CTX), patcher:
            await recap.build_week_recap(_event())
        fact = calls[0]["fact"]
        # Week number, a seeded display_name, and an integer score all appear.
        self.assertIn("3", fact)
        self.assertIn("alice", fact)
        self.assertIn("9", fact)
        # Season standings text carries the season total + rank.
        self.assertIn("30", fact)

    async def test_phrase_called_with_recap_prompt(self) -> None:
        patcher, calls = _phrase_returns("x")
        with _context_returns(_CTX), patcher:
            await recap.build_week_recap(_event())
        self.assertEqual(calls[0]["system_prompt"], recap.RECAP_PROMPT)

    async def test_falls_back_to_render_chat_on_phrase_none(self) -> None:
        event = _event()
        patcher, _ = _phrase_returns(None)
        with _context_returns(_CTX), patcher:
            out = await recap.build_week_recap(event)
        self.assertEqual(out, render_chat(event))

    async def test_falls_back_to_render_chat_when_reader_raises(self) -> None:
        event = _event()
        patcher, _ = _phrase_returns("should-not-be-used")
        with _context_raises(RuntimeError("db boom")), patcher:
            out = await recap.build_week_recap(event)
        self.assertEqual(out, render_chat(event))
        self.assertIsNotNone(out)

    async def test_falls_back_to_render_chat_on_empty_context(self) -> None:
        event = _event()
        empty = {"week": 3, "weekly_scores": [], "season_standings": []}
        patcher, calls = _phrase_returns("should-not-be-used")
        with _context_returns(empty), patcher:
            out = await recap.build_week_recap(event)
        self.assertEqual(out, render_chat(event))
        # Nothing to narrate -> phrase is never called.
        self.assertEqual(calls, [])


class RecapFactTests(unittest.TestCase):
    """The pure fact builder returns None on an empty week and a deterministic
    multi-line string otherwise, carrying display-only numbers only."""

    def test_returns_none_on_empty_weekly_scores(self) -> None:
        self.assertIsNone(
            recap._recap_fact({"week": 3, "weekly_scores": [], "season_standings": []})
        )

    def test_deterministic_multi_line_fact(self) -> None:
        fact = recap._recap_fact(_CTX)
        self.assertIsNotNone(fact)
        # Same input -> same output (pure).
        self.assertEqual(fact, recap._recap_fact(_CTX))
        self.assertIn("alice", fact)
        self.assertIn("bob", fact)
        self.assertNotIn("user_id", fact)

    def test_supplied_storylines_render_into_fact(self) -> None:
        fact = recap._recap_fact(_CTX_WITH_STORYLINES)
        self.assertIsNotNone(fact)
        self.assertIn("Season storylines:", fact)
        self.assertIn("missed their mortal lock 3 weeks running", fact)
        self.assertIn("biggest upset", fact)
        # Deterministic (pure).
        self.assertEqual(fact, recap._recap_fact(_CTX_WITH_STORYLINES))

    def test_absent_or_empty_storylines_add_no_section(self) -> None:
        # Byte-identical to the no-storyline output whether the key is absent or [].
        baseline = recap._recap_fact(_CTX)
        self.assertNotIn("Season storylines:", baseline or "")
        self.assertEqual(recap._recap_fact({**_CTX, "storylines": []}), baseline)


if __name__ == "__main__":
    unittest.main()
