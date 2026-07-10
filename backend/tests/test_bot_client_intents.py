"""Regression guard for the bot's gateway intents (260709-k5w follow-up).

The inbound @mention Q&A listener silently did nothing in every server because
``build_intents`` enabled ``message_content`` (READ the text) but NOT
``guild_messages`` (RECEIVE the event). The listener unit tests call ``on_message``
directly with a fake message, so they never exercise the real gateway subscription —
this test asserts the intent set itself so that gap can't regress.

Run with: ``backend/.venv/bin/python -m unittest tests.test_bot_client_intents -v``
"""

from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timezone
from unittest import mock

from app.bot import db_bridge, qa
from app.bot.client import build_intents
from app.scoreboard.types import ScoreboardOdds
from app.services import espn_extra, live_odds, weather


class BuildIntentsTests(unittest.TestCase):
    def test_guild_messages_enabled_so_on_message_fires_in_servers(self) -> None:
        # THE regression: without guild_messages the @mention listener never fires.
        self.assertTrue(
            build_intents().guild_messages,
            "guild_messages must be enabled or on_message never fires in a guild",
        )

    def test_message_content_enabled_so_question_text_is_readable(self) -> None:
        self.assertTrue(build_intents().message_content)

    def test_required_intents_all_enabled(self) -> None:
        intents = build_intents()
        for name in ("guilds", "guild_messages", "dm_messages", "members", "message_content"):
            self.assertTrue(getattr(intents, name), f"{name} intent must be enabled")

    def test_stays_minimal_no_unneeded_intents(self) -> None:
        # Minimal-by-default: things the bot does not use stay off (e.g. presences,
        # typing, reactions, voice) — guards against an accidental Intents.all().
        intents = build_intents()
        for name in ("presences", "typing", "guild_reactions", "voice_states", "bans"):
            self.assertFalse(getattr(intents, name), f"{name} intent should stay off")


class PredictionIntentRoutingTests(unittest.TestCase):
    """End-to-end routing guard (260710-mpw): a ``prediction`` classification with a real
    team routes through ``_build_fact`` to ``_prediction_fact`` and yields a non-empty
    derived-facts answer (the pick + cover math reach the reply verbatim)."""

    _INPUTS = {
        "asked_team": "KC",
        "home": "KC",
        "away": "LAC",
        "favorite": "KC",
        "underdog": "LAC",
        "spread": "3.0",
        "total": "47.5",
        "espn_event_id": 555,
        "kickoff_at": datetime(2026, 1, 5, 18, 0, tzinfo=timezone.utc),
        "season": 2025,
        "week": 5,
        "record": "4-1",
        "ats": "3-2",
    }

    def test_prediction_with_real_team_routes_to_non_empty_answer(self) -> None:
        async def _classify(_question):
            return {"intent": "prediction", "team": "Chiefs"}

        async def _tokens():
            return {"KC", "CHIEFS"}

        async def _inputs(_team):
            return dict(self._INPUTS)

        async def _odds(_season, _week, _event_id):
            return ScoreboardOdds(provider="DraftKings", spread=-3.0, total=47.5)

        async def _injuries(_event_id):
            return None

        async def _voice():
            return "You are the snarky house bot."

        async def _phrase(_fact, *, system_prompt):
            return None  # fall back to the verbatim header + body

        with (
            mock.patch.object(qa, "classify_question", _classify),
            mock.patch.object(db_bridge, "get_real_team_tokens_async", _tokens),
            mock.patch.object(db_bridge, "get_prediction_inputs_async", _inputs),
            mock.patch.object(live_odds, "fetch_live_odds", _odds),
            mock.patch.object(espn_extra, "fetch_injuries", _injuries),
            mock.patch.object(weather, "lookup_stadium", lambda _abbr: None),
            mock.patch.object(db_bridge, "resolve_active_voice_async", _voice),
            mock.patch.object(qa.llm_client, "phrase", _phrase),
        ):
            out = asyncio.run(qa.answer_question("who wins the Chiefs game?", discord_id=7))

        self.assertTrue(out)
        self.assertIn("**My call: KC to cover", out)
        self.assertIn("4-1 straight up and 3-2 against the spread", out)


if __name__ == "__main__":
    unittest.main()
