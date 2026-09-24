"""Offline tests for the per-answer bot telemetry (issue #248, item 15).

Run from backend/ with ``.venv/bin/python -m unittest tests.test_bot_telemetry -v``.
"""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest import mock

from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.bot import db_bridge, qa, qa_open
from app.db import get_session
from app.main import app
from app.models import User
from app.services import bot_telemetry
from app.services.auth import create_session_cookie, hash_password


def _run(coro):
    return asyncio.run(coro)


class _Pushed:
    """Capture every record the detached Redis write would store."""

    def __init__(self) -> None:
        self.records: list[dict] = []

    async def __call__(self, record: dict) -> None:
        self.records.append(record)


def _patch_answer_seams(*, intent: str, open_answer: object = "open answer"):
    async def _classify(question, *, history=(), asker_name=None):
        return {"intent": intent, "team": None, "week": None, "subject": None, "nfl": True}

    async def _tokens():
        return set()

    async def _voice():
        return "voice"

    async def _open(question, **kwargs):
        bot_telemetry.note_round()
        bot_telemetry.note_tool(
            "lookup_standings", {"asker_discord_id": 42, "team": "KC" * 100}, outcome="ok"
        )
        return open_answer

    return (
        mock.patch.object(qa, "classify_question", _classify),
        mock.patch.object(db_bridge, "get_real_team_tokens_async", _tokens),
        mock.patch.object(db_bridge, "resolve_active_voice_async", _voice),
        mock.patch.object(qa_open, "answer_open", _open),
    )


class AnswerTraceTests(unittest.TestCase):
    def _answer(self, *, intent: str, open_answer: object = "open answer") -> dict:
        pushed = _Pushed()
        a, b, c, d = _patch_answer_seams(intent=intent, open_answer=open_answer)
        with a, b, c, d, mock.patch.object(bot_telemetry, "_push", pushed):

            async def _go() -> str:
                answer = await qa.answer_question(
                    "who leads? " * 100, discord_id=7, conversation_key="123", asker_name="ada"
                )
                await asyncio.sleep(0)  # let the detached write run
                return answer

            answer = _run(_go())
        self.assertEqual(len(pushed.records), 1)
        record = pushed.records[0]
        record["_answer"] = answer
        return record

    def test_an_open_answer_records_the_tools_rounds_and_path(self) -> None:
        record = self._answer(intent="open_nfl")
        self.assertEqual(record["intent"], "open_nfl")
        self.assertEqual(record["path"], "open")
        self.assertEqual(record["rounds"], 1)
        self.assertEqual(record["conversation"], "123")
        self.assertEqual(record["asker"], "ada")
        self.assertIsNone(record["fallback"])
        self.assertEqual(record["answer"], "open answer")
        (tool,) = record["tools"]
        self.assertEqual(tool["name"], "lookup_standings")
        self.assertNotIn("asker_discord_id", tool["args"])
        self.assertLessEqual(len(tool["args"]["team"]), 120)
        self.assertLessEqual(len(record["question"]), bot_telemetry.TEXT_LIMIT)
        self.assertIsInstance(record["latency_ms"], int)
        self.assertNotIn("7", json.dumps(record["tools"]))

    def test_a_degraded_open_answer_is_marked(self) -> None:
        record = self._answer(intent="open_nfl", open_answer=None)
        self.assertEqual(record["_answer"], qa._OPEN_DEGRADE_FACT)
        self.assertEqual(record["fallback"], "open_degrade")

    def test_a_trace_outside_an_answer_is_ignored(self) -> None:
        bot_telemetry.note_tool("x", {})
        bot_telemetry.note_round()
        bot_telemetry.note_fallback("error")  # no active trace: nothing raises

    def test_finish_without_a_running_loop_never_raises(self) -> None:
        trace = bot_telemetry.AnswerTrace(question="q", conversation_key=None, asker_name=None)
        bot_telemetry.finish(trace, "a")


class _FailingRedis:
    def pipeline(self, transaction=True):
        raise ConnectionError("redis down")

    async def aclose(self) -> None:
        return None


class StoreTests(unittest.TestCase):
    def test_a_redis_outage_on_write_fails_open(self) -> None:
        with mock.patch.object(bot_telemetry, "_redis_client", lambda: _FailingRedis()):
            _run(bot_telemetry._push({"a": 1}))

    def test_read_recent_decodes_and_skips_junk(self) -> None:
        client = mock.Mock()
        client.lrange.return_value = [json.dumps({"intent": "scores"}).encode(), b"not json"]
        with mock.patch("redis.Redis.from_url", return_value=client):
            self.assertEqual(bot_telemetry.read_recent(5), [{"intent": "scores"}])
        client.lrange.assert_called_once_with(bot_telemetry.REDIS_KEY, 0, 4)

    def test_read_recent_is_none_when_redis_is_down(self) -> None:
        with mock.patch("redis.Redis.from_url", side_effect=ConnectionError("down")):
            self.assertIsNone(bot_telemetry.read_recent())


class BotAnswersApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
        SQLModel.metadata.create_all(self.engine)
        pw = hash_password("correct horse battery staple")
        with Session(self.engine) as session:
            admin = User(
                display_name="admin", password_hash=pw, is_admin=True, is_active=True, discord_id=1
            )
            member = User(display_name="member", password_hash=pw, is_active=True, discord_id=2)
            session.add_all([admin, member])
            session.commit()
            session.refresh(admin)
            session.refresh(member)
            self.admin_id, self.member_id = admin.id, member.id

        def _override_get_session():
            with Session(self.engine) as session:
                yield session

        app.dependency_overrides[get_session] = _override_get_session
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.pop(get_session, None)
        self.client.close()
        self.engine.dispose()

    def _get(self, user_id: int | None):
        headers = {"Authorization": f"Bearer {create_session_cookie(user_id)}"} if user_id else {}
        return self.client.get("/api/admin/bot-answers", headers=headers)

    def test_the_admin_gets_the_stored_answers(self) -> None:
        stored = [{"intent": "scores", "tools": [{"name": "t", "args": {"week": 2}}]}, {"x": 1}]
        with mock.patch.object(bot_telemetry, "read_recent", return_value=stored):
            resp = self._get(self.admin_id)
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertTrue(body["available"])
        self.assertEqual(body["answers"][0]["intent"], "scores")
        self.assertEqual(body["answers"][0]["tools"][0]["outcome"], "ok")
        self.assertEqual(len(body["answers"]), 2)

    def test_a_redis_outage_is_reported_not_raised(self) -> None:
        with mock.patch.object(bot_telemetry, "read_recent", return_value=None):
            resp = self._get(self.admin_id)
        self.assertEqual(resp.json(), {"available": False, "answers": []})

    def test_only_an_admin_may_read(self) -> None:
        with mock.patch.object(bot_telemetry, "read_recent", return_value=[]):
            self.assertEqual(self._get(None).status_code, 401)
            self.assertEqual(self._get(self.member_id).status_code, 403)


if __name__ == "__main__":
    unittest.main()
