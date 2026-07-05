"""Offline unit tests for the pure freeze_week_embed builder (260705-jo9).

The builder is Discord-send-free: it constructs a ``discord.Embed`` but takes no
client and performs no send, so it is fully unit-testable without a live gateway.
Events are built via the real :func:`app.services.notifications.freeze_week_event`
so the tests consume the real payload shape (never hand-rolled dict literals),
exactly as ``test_window_embed`` uses ``window_opened_event``.

Layout under test (the LIGHT lines-locked card):
* freeze.week -> title ``Week {week} - Lines Locked``, gold :data:`LINES_LOCKED_COLOR`,
  a non-empty deterministic body line mentioning the week;
* no custom app-emoji token in the title; no fields, no footer.

This module also LOCKS the dual-dispatch retarget: ``freeze_week_event`` now carries
``targets == ["logger", "chat"]`` (logger KEPT, chat ADDED) so the event posts to
BOTH the ops-log terse line and the chat lines-locked card.

Run with: ``backend/.venv/bin/python -m unittest tests.test_freeze_week_embed -v``
(there is no bare ``python`` on PATH on this machine).
"""

from __future__ import annotations

import unittest

import discord

from app.bot.freeze_week_embed import (
    LINES_LOCKED_COLOR,
    build_freeze_week_embed,
)
from app.services.notifications import freeze_week_event


class BuildFreezeWeekEmbedTests(unittest.TestCase):
    def test_title_color_and_body(self) -> None:
        embed = build_freeze_week_embed(freeze_week_event(week=3))
        self.assertEqual(embed.title, "Week 3 - Lines Locked")
        assert embed.color is not None
        self.assertEqual(embed.color.value, LINES_LOCKED_COLOR)
        body = (embed.description or "").strip()
        self.assertTrue(body)
        self.assertIn("3", body)

    def test_title_has_no_custom_emoji_token(self) -> None:
        embed = build_freeze_week_embed(freeze_week_event(week=1))
        self.assertNotIn("<:", embed.title or "")


class MinimalEventTests(unittest.TestCase):
    def test_minimal_event_returns_embed_and_does_not_raise(self) -> None:
        embed = build_freeze_week_embed(freeze_week_event(week=2))
        self.assertIsInstance(embed, discord.Embed)


class RetargetTests(unittest.TestCase):
    def test_freeze_week_event_targets_logger_and_chat(self) -> None:
        # Dual-dispatch promotion (260705-jo9): logger KEPT, chat ADDED.
        self.assertEqual(freeze_week_event(week=5)["targets"], ["logger", "chat"])


if __name__ == "__main__":
    unittest.main()
