"""Per-answer telemetry for the bot's Q&A path (issue #248, item 15).

One :class:`AnswerTrace` rides a ContextVar through ``qa.answer_question``; the open
path's tool loop adds each tool call and round to it. When the answer is done the trace
is logged as ONE structlog event and pushed onto a capped Redis list for the admin view.

Storage is Redis, not a table: the view is a debugging aid over the last few hundred
answers, both the bot and the API already reach Redis, and a capped list needs no
migration or pruning job. The write runs as a detached task with a short timeout and
fails open, so a Redis outage never delays or breaks an answer.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

from app.config import settings

logger = structlog.get_logger(__name__)

REDIS_KEY = "bot:answers"
MAX_ANSWERS = 200
TEXT_LIMIT = 500
_ARG_TEXT_LIMIT = 120
_REDIS_TIMEOUT_SECONDS = 1.0

_CURRENT: ContextVar[AnswerTrace | None] = ContextVar("bot_answer_trace", default=None)
# Detached writes are held here so the loop does not garbage-collect them mid-flight.
_PENDING: set[asyncio.Task] = set()


def _clip(text: object, limit: int = TEXT_LIMIT) -> str | None:
    if text is None:
        return None
    value = str(text)
    return value if len(value) <= limit else value[: limit - 1] + "…"


@dataclass
class AnswerTrace:
    question: str
    conversation_key: str | None
    asker_name: str | None
    started: float = field(default_factory=time.monotonic)
    at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    intent: str | None = None
    path: str = "grounded"
    tools: list[dict] = field(default_factory=list)
    rounds: int = 0
    fallback: str | None = None

    def record(self, answer: str | None) -> dict:
        model = settings.llm_api_model
        if self.path == "open" and settings.llm_api_open_model:
            model = settings.llm_api_open_model
        return {
            "at": self.at,
            "conversation": self.conversation_key,
            "asker": self.asker_name,
            "question": _clip(self.question),
            "intent": self.intent,
            "path": self.path,
            "tools": self.tools,
            "rounds": self.rounds,
            "fallback": self.fallback,
            "latency_ms": round((time.monotonic() - self.started) * 1000),
            "vendor": settings.llm_api_vendor,
            "model": model,
            "answer": _clip(answer),
        }


def start(question: str, *, conversation_key: str | None, asker_name: str | None) -> AnswerTrace:
    trace = AnswerTrace(question=question, conversation_key=conversation_key, asker_name=asker_name)
    _CURRENT.set(trace)
    return trace


def _current() -> AnswerTrace | None:
    return _CURRENT.get()


def note_intent(intent: str) -> None:
    if (trace := _current()) is not None:
        trace.intent = intent


def note_open_path() -> None:
    if (trace := _current()) is not None:
        trace.path = "open"


def note_round() -> None:
    if (trace := _current()) is not None:
        trace.rounds += 1


def note_tool(name: str, arguments: dict[str, Any], *, outcome: str = "ok") -> None:
    """Record one tool call. The asker's Discord id is bound in code and never kept."""
    if (trace := _current()) is None:
        return
    args = {
        k: _clip(v, _ARG_TEXT_LIMIT) if isinstance(v, str) else v
        for k, v in arguments.items()
        if k != "asker_discord_id"
    }
    trace.tools.append({"name": name, "args": args, "outcome": outcome})


def note_fallback(kind: str) -> None:
    if (trace := _current()) is not None and trace.fallback is None:
        trace.fallback = kind


def finish(trace: AnswerTrace, answer: str | None) -> None:
    """Log the trace and push it to Redis in the background. Never raises."""
    try:
        record = trace.record(answer)
        logger.info(
            "bot_answer",
            **{k: v for k, v in record.items() if k not in ("question", "answer")},
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(_push(record))
        _PENDING.add(task)
        task.add_done_callback(_PENDING.discard)
    except Exception:
        logger.warning("bot_answer_trace_failed", exc_info=True)
    finally:
        _CURRENT.set(None)


def _redis_client():
    import redis.asyncio as aioredis

    return aioredis.Redis.from_url(
        settings.redis_url,
        socket_timeout=_REDIS_TIMEOUT_SECONDS,
        socket_connect_timeout=_REDIS_TIMEOUT_SECONDS,
    )


async def _push(record: dict) -> None:
    client = None
    try:
        client = _redis_client()
        async with client.pipeline(transaction=True) as pipe:
            pipe.lpush(REDIS_KEY, json.dumps(record, default=str))
            pipe.ltrim(REDIS_KEY, 0, MAX_ANSWERS - 1)
            await asyncio.wait_for(pipe.execute(), timeout=_REDIS_TIMEOUT_SECONDS)
    except Exception:
        logger.warning("bot_answer_store_failed")
    finally:
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                pass


def read_recent(limit: int = MAX_ANSWERS) -> list[dict] | None:
    """The newest ``limit`` stored answers, newest first, or ``None`` if Redis is down."""
    import redis

    try:
        client = redis.Redis.from_url(
            settings.redis_url,
            socket_timeout=_REDIS_TIMEOUT_SECONDS,
            socket_connect_timeout=_REDIS_TIMEOUT_SECONDS,
        )
        try:
            raw = client.lrange(REDIS_KEY, 0, max(0, min(limit, MAX_ANSWERS)) - 1)
        finally:
            client.close()
    except Exception:
        logger.warning("bot_answer_read_failed")
        return None
    answers: list[dict] = []
    for item in raw or []:  # type: ignore[union-attr]
        try:
            decoded = json.loads(item)
        except Exception:
            continue
        if isinstance(decoded, dict):
            answers.append(decoded)
    return answers
