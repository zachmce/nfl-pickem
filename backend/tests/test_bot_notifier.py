"""Offline unit tests for the bot-side channel resolver (QT-1).

These tests NEVER touch a live Redis or a live Discord gateway. The resilient
``redis.asyncio`` subscriber loop is exercised by the manual/live confirm step in
the SUMMARY; here we unit-test the pure seam — ``resolve_channel`` — which is the
guild-scoping guard (T-kd8-03): it searches ONLY the passed guild's channels and
matches by numeric id OR by name.

Run with: ``backend/.venv/bin/python -m unittest tests.test_bot_notifier -v``
(there is no bare ``python`` on PATH on this machine).
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from app.bot.notifier import resolve_channel


@dataclass
class _FakeChannel:
    id: int
    name: str


class _FakeGuild:
    """A guild-like object exposing only the ``.channels`` iterable the resolver
    reads — proves the resolver needs no real Discord object."""

    def __init__(self, channels: list[_FakeChannel]) -> None:
        self.channels = channels


def _guild() -> _FakeGuild:
    return _FakeGuild(
        [
            _FakeChannel(id=123, name="pickem-logger"),
            _FakeChannel(id=456, name="pickem-chat"),
        ]
    )


class ResolveChannelTests(unittest.TestCase):
    def test_match_by_numeric_id(self) -> None:
        ch = resolve_channel(_guild(), "123")
        self.assertIsNotNone(ch)
        self.assertEqual(ch.id, 123)
        self.assertEqual(ch.name, "pickem-logger")

    def test_match_by_name(self) -> None:
        ch = resolve_channel(_guild(), "pickem-logger")
        self.assertIsNotNone(ch)
        self.assertEqual(ch.id, 123)

    def test_miss_returns_none(self) -> None:
        self.assertIsNone(resolve_channel(_guild(), "nonexistent"))

    def test_none_setting_returns_none(self) -> None:
        self.assertIsNone(resolve_channel(_guild(), None))

    def test_blank_setting_returns_none(self) -> None:
        self.assertIsNone(resolve_channel(_guild(), "   "))

    def test_numeric_id_not_present_returns_none(self) -> None:
        # An int that matches no channel id must NOT fall back to a name match.
        self.assertIsNone(resolve_channel(_guild(), "999"))

    def test_none_guild_returns_none(self) -> None:
        # get_guild() can return None (bot not in guild yet) — must not raise.
        self.assertIsNone(resolve_channel(None, "pickem-logger"))


if __name__ == "__main__":
    unittest.main()
