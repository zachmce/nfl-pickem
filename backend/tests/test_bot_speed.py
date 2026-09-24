"""The 2026-09-24 audit's speed fixes that live outside ``qa_open``: the cached team
tokens and the avatar sweep that skips unchanged hashes. Offline; no Discord, no DB."""

from __future__ import annotations

import asyncio
import contextlib
import unittest
from types import SimpleNamespace
from typing import cast
from unittest import mock

from app.bot import client as bot_client
from app.bot import db_bridge


class TeamTokensCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        db_bridge._team_tokens_cache = None
        self.addCleanup(setattr, db_bridge, "_team_tokens_cache", None)

    def _read(self, tokens: set[str], reads: list[int]):
        def _fake(_session) -> set[str]:
            reads.append(1)
            return set(tokens)

        return (
            mock.patch.object(db_bridge, "task_session", contextlib.nullcontext),
            mock.patch.object(db_bridge, "get_real_team_tokens", _fake),
        )

    def test_the_token_set_is_read_once_per_ttl(self) -> None:
        reads: list[int] = []
        session_patch, read_patch = self._read({"KC", "CHIEFS"}, reads)
        with session_patch, read_patch:
            first = asyncio.run(db_bridge.get_real_team_tokens_async())
            second = asyncio.run(db_bridge.get_real_team_tokens_async())
        self.assertEqual(first, {"KC", "CHIEFS"})
        self.assertEqual(second, first)
        self.assertEqual(len(reads), 1)

    def test_an_empty_set_is_never_cached(self) -> None:
        reads: list[int] = []
        session_patch, read_patch = self._read(set(), reads)
        with session_patch, read_patch:
            asyncio.run(db_bridge.get_real_team_tokens_async())
            asyncio.run(db_bridge.get_real_team_tokens_async())
        self.assertEqual(len(reads), 2)


class AvatarSweepTests(unittest.TestCase):
    def test_an_unchanged_stored_hash_is_not_written_again(self) -> None:
        members = [
            SimpleNamespace(id=1, avatar=SimpleNamespace(key="abc")),
            SimpleNamespace(id=2, avatar=None),
            SimpleNamespace(id=3, avatar=SimpleNamespace(key="new")),
        ]
        fake_bot = SimpleNamespace(
            get_guild=lambda _id: SimpleNamespace(members=members),
            _stored_avatars={},
        )
        writes: list[tuple[int, str | None]] = []

        async def _upsert(discord_id: int, avatar_hash: str | None) -> bool:
            writes.append((discord_id, avatar_hash))
            return discord_id != 3  # member 3 has no account yet

        sweep = bot_client.PickemBot._avatar_sweep_loop.coro
        with mock.patch.object(db_bridge, "upsert_avatar_hash_async", _upsert):
            asyncio.run(sweep(cast(bot_client.PickemBot, fake_bot)))
            asyncio.run(sweep(cast(bot_client.PickemBot, fake_bot)))
        self.assertEqual(writes, [(1, "abc"), (2, None), (3, "new"), (3, "new")])


if __name__ == "__main__":
    unittest.main()
