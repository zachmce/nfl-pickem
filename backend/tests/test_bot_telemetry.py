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

    def test_the_classifier_output_and_a_tool_note_are_kept(self) -> None:
        trace = bot_telemetry.start("q", conversation_key=None, asker_name=None)
        bot_telemetry.note_classification({"intent": "open_nfl", "team": "KC"})
        bot_telemetry.note_tool("lookup_x", {"team": "KC"}, outcome="note", note="n" * 500)
        record = trace.record("a")
        bot_telemetry.finish(trace, "a")
        self.assertEqual(record["classifier"], {"intent": "open_nfl", "team": "KC"})
        self.assertEqual(record["tools"][0]["outcome"], "note")
        self.assertLessEqual(len(record["tools"][0]["note"]), 200)

    def test_a_trace_the_cog_opened_is_not_finished_by_the_answer(self) -> None:
        pushed = _Pushed()
        a, b, c, d = _patch_answer_seams(intent="open_nfl")
        with a, b, c, d, mock.patch.object(bot_telemetry, "_push", pushed):

            async def _go() -> bot_telemetry.AnswerTrace:
                trace = bot_telemetry.start("q", conversation_key="1", asker_name="ada")
                await qa.answer_question("q", discord_id=7, history=[("user", "x")])
                await asyncio.sleep(0)
                self.assertEqual(pushed.records, [])  # the cog finishes it, after the send
                return trace

            trace = _run(_go())
        self.assertEqual(trace.history_turns, 1)
        self.assertEqual(trace.tools[0]["name"], "lookup_standings")

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
        client.lrange.assert_called_once_with(
            bot_telemetry.REDIS_KEY, 0, bot_telemetry.MAX_ENTRIES - 1
        )

    def test_read_recent_keeps_only_the_time_window(self) -> None:
        from datetime import UTC, datetime

        client = mock.Mock()
        client.lrange.return_value = [
            json.dumps({"at": at}).encode()
            for at in ("2026-09-24T12:00:00+00:00", "2026-09-20T12:00:00+00:00", "junk")
        ]
        with mock.patch("redis.Redis.from_url", return_value=client):
            out = bot_telemetry.read_recent(
                10, since=datetime(2026, 9, 22, tzinfo=UTC), until=datetime(2026, 9, 25, tzinfo=UTC)
            )
        self.assertEqual(out, [{"at": "2026-09-24T12:00:00+00:00"}])

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
        return self.client.get("/api/admin/bot-transcript", headers=headers)

    def test_the_admin_gets_the_stored_answers(self) -> None:
        stored = [{"intent": "scores", "tools": [{"name": "t", "args": {"week": 2}}]}, {"x": 1}]
        with mock.patch.object(bot_telemetry, "read_recent", return_value=stored):
            resp = self._get(self.admin_id)
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertTrue(body["available"])
        self.assertEqual(body["entries"][0]["intent"], "scores")
        self.assertEqual(body["entries"][0]["tools"][0]["outcome"], "ok")
        self.assertEqual(len(body["entries"]), 2)

    def test_the_export_is_json_lines_oldest_first_for_the_window(self) -> None:
        stored = [
            {"at": "2026-09-24T12:00:00+00:00", "n": 2},
            {"at": "2026-09-23T12:00:00+00:00", "n": 1},
        ]
        assert self.admin_id is not None and self.member_id is not None
        headers = {"Authorization": f"Bearer {create_session_cookie(self.admin_id)}"}
        with mock.patch.object(bot_telemetry, "read_recent", return_value=stored) as read:
            resp = self.client.get(
                "/api/admin/bot-transcript/export?since=2026-09-23T00:00:00", headers=headers
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertIn("attachment;", resp.headers["content-disposition"])
        lines = [json.loads(line) for line in resp.text.splitlines()]
        self.assertEqual([line["n"] for line in lines], [1, 2])
        self.assertIsNotNone(read.call_args.kwargs["since"].tzinfo)
        member = {"Authorization": f"Bearer {create_session_cookie(self.member_id)}"}
        self.assertEqual(
            self.client.get("/api/admin/bot-transcript/export", headers=member).status_code, 403
        )

    def test_a_redis_outage_is_reported_not_raised(self) -> None:
        with mock.patch.object(bot_telemetry, "read_recent", return_value=None):
            resp = self._get(self.admin_id)
        self.assertEqual(resp.json(), {"available": False, "entries": []})

    def test_only_an_admin_may_read(self) -> None:
        with mock.patch.object(bot_telemetry, "read_recent", return_value=[]):
            self.assertEqual(self._get(None).status_code, 401)
            self.assertEqual(self._get(self.member_id).status_code, 403)


if __name__ == "__main__":
    unittest.main()
