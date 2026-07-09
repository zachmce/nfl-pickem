"""Regression guard for the bot's gateway intents (260709-k5w follow-up).

The inbound @mention Q&A listener silently did nothing in every server because
``build_intents`` enabled ``message_content`` (READ the text) but NOT
``guild_messages`` (RECEIVE the event). The listener unit tests call ``on_message``
directly with a fake message, so they never exercise the real gateway subscription —
this test asserts the intent set itself so that gap can't regress.

Run with: ``backend/.venv/bin/python -m unittest tests.test_bot_client_intents -v``
"""

from __future__ import annotations

import unittest

from app.bot.client import build_intents


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


if __name__ == "__main__":
    unittest.main()
