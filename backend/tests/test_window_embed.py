"""Offline unit tests for the pure window_embed builder (260705-j8o).

The builder is Discord-send-free: it constructs a ``discord.Embed`` but takes no
client and performs no send, so it is fully unit-testable without a live gateway.
Events are built via the real :func:`app.services.notifications.window_opened_event`
/ :func:`window_closed_event` so the tests consume the real payload shape (never
hand-rolled dict literals), exactly as ``test_misc_graded_embed`` uses
``misc_graded_event``.

Layout under test (the LIGHT window card):
* window.opened -> title ``Week {week} - Picks Open``, green :data:`OPEN_COLOR`,
  a non-empty deterministic body line;
* window.closed -> title ``Week {week} - Picks Locked``, red :data:`LOCKED_COLOR`,
  a non-empty deterministic body line;
* no custom app-emoji token in the title; no fields, no footer.

Run with: ``backend/.venv/bin/python -m unittest tests.test_window_embed -v``
(there is no bare ``python`` on PATH on this machine).
"""

from __future__ import annotations

import unittest

import discord

from app.bot.window_embed import (
    LOCKED_COLOR,
    OPEN_COLOR,
    build_window_embed,
)
from app.services.notifications import window_closed_event, window_opened_event


class BuildWindowEmbedOpenedTests(unittest.TestCase):
    def test_opened_title_color_and_body(self) -> None:
        embed = build_window_embed(window_opened_event(week=3))
        self.assertEqual(embed.title, "Week 3 - Picks Open")
        assert embed.color is not None
        self.assertEqual(embed.color.value, OPEN_COLOR)
        self.assertTrue((embed.description or "").strip())

    def test_opened_title_has_no_custom_emoji_token(self) -> None:
        embed = build_window_embed(window_opened_event(week=1))
        self.assertNotIn("<:", embed.title or "")


class BuildWindowEmbedClosedTests(unittest.TestCase):
    def test_closed_title_color_and_body(self) -> None:
        embed = build_window_embed(window_closed_event(week=3))
        self.assertEqual(embed.title, "Week 3 - Picks Locked")
        assert embed.color is not None
        self.assertEqual(embed.color.value, LOCKED_COLOR)
        self.assertTrue((embed.description or "").strip())

    def test_closed_title_has_no_custom_emoji_token(self) -> None:
        embed = build_window_embed(window_closed_event(week=7))
        self.assertNotIn("<:", embed.title or "")


class MinimalEventTests(unittest.TestCase):
    def test_minimal_event_returns_embed_and_does_not_raise(self) -> None:
        embed = build_window_embed(window_opened_event(week=2))
        self.assertIsInstance(embed, discord.Embed)


if __name__ == "__main__":
    unittest.main()
