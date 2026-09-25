"""The bot's channel transcript with debug data (issue #248 item 15, issue #252).

One entry per message in the league chat channel, and per message addressed to the bot
anywhere else: who wrote it, whether the bot answered and why not, and for an answer the
classifier's output, the path, each tool call with its outcome, the fallback and the
reply. The bot's own event posts are entries too, so an export reads like the channel.
Tool payloads are never stored; a tool's outcome is enough to debug a reply.

Storage is a capped Redis list. The write is a detached task with a short timeout that
fails open, so a Redis outage never delays or breaks an answer. An :class:`AnswerTrace`
rides a ContextVar through the answer so the tool loop can add to it.
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

REDIS_KEY = "bot:transcript"
# About 1 KB an entry: weeks of league chat in a few MB.
MAX_ENTRIES = 5000
TEXT_LIMIT = 1000
ANSWER_LIMIT = 2000
_ARG_TEXT_LIMIT = 120
_NOTE_LIMIT = 200
_REDIS_TIMEOUT_SECONDS = 1.0

_CURRENT: ContextVar[AnswerTrace | None] = ContextVar("bot_answer_trace", default=None)
# Detached writes are held here so the loop does not garbage-collect them mid-flight.
_PENDING: set[asyncio.Task] = set()


def _clip(text: object, limit: int = TEXT_LIMIT) -> str | None:
    if text is None:
        return None
    value = str(text)
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class AnswerTrace:
    question: str
    conversation_key: str | None
    asker_name: str | None
    message: dict[str, Any] = field(default_factory=dict)
    started: float = field(default_factory=time.monotonic)
    at: str = field(default_factory=_now)
    decision: str | None = None
    intent: str | None = None
    classifier: dict[str, Any] | None = None
    path: str = "grounded"
    tools: list[dict] = field(default_factory=list)
    rounds: int = 0
    fallback: str | None = None
    history_turns: int | None = None

    def record(self, answer: str | None) -> dict:
        model = settings.llm_api_model
        if self.path == "open" and settings.llm_api_open_model:
            model = settings.llm_api_open_model
        return {
            "at": self.at,
            "kind": "member",
            **self.message,
            "conversation": self.conversation_key,
            "asker": self.asker_name,
            "question": _clip(self.question),
            "decision": self.decision or ("answered" if answer is not None else None),
            "intent": self.intent,
            "classifier": self.classifier,
            "path": self.path,
            "tools": self.tools,
            "rounds": self.rounds,
            "fallback": self.fallback,
            "history_turns": self.history_turns,
            "latency_ms": round((time.monotonic() - self.started) * 1000),
            "vendor": settings.llm_api_vendor,
            "model": model,
            "answer": _clip(answer, ANSWER_LIMIT),
        }


def start(
    question: str,
    *,
    conversation_key: str | None,
    asker_name: str | None,
    message: dict[str, Any] | None = None,
) -> AnswerTrace:
    trace = AnswerTrace(
        question=question,
        conversation_key=conversation_key,
        asker_name=asker_name,
        message=dict(message or {}),
    )
    _CURRENT.set(trace)
    return trace


def current() -> AnswerTrace | None:
    return _CURRENT.get()


def note_intent(intent: str) -> None:
    if (trace := current()) is not None:
        trace.intent = intent


def note_classification(raw: object) -> None:
    """The classifier's raw output, as the model wrote it."""
    if (trace := current()) is not None and isinstance(raw, dict):
        trace.classifier = {
            k: _clip(v, _ARG_TEXT_LIMIT) if isinstance(v, str) else v for k, v in raw.items()
        }


def note_open_path() -> None:
    if (trace := current()) is not None:
        trace.path = "open"


def note_round() -> None:
    if (trace := current()) is not None:
        trace.rounds += 1


def note_history(turns: int) -> None:
    if (trace := current()) is not None:
        trace.history_turns = turns


def note_tool(
    name: str, arguments: dict[str, Any], *, outcome: str = "ok", note: object = None
) -> None:
    """Record one tool call. The asker's Discord id is bound in code and never kept."""
    if (trace := current()) is None:
        return
    args = {
        k: _clip(v, _ARG_TEXT_LIMIT) if isinstance(v, str) else v
        for k, v in arguments.items()
        if k != "asker_discord_id"
    }
    call: dict[str, Any] = {"name": name, "args": args, "outcome": outcome}
    if note is not None:
        call["note"] = _clip(note, _NOTE_LIMIT)
    trace.tools.append(call)


def note_fallback(kind: str) -> None:
    if (trace := current()) is not None and trace.fallback is None:
        trace.fallback = kind


def finish(trace: AnswerTrace, answer: str | None, *, decision: str | None = None) -> None:
    """Log the trace and store it in the background. Never raises."""
    try:
        if decision is not None:
            trace.decision = decision
        _store(trace.record(answer))
    except Exception:
        logger.warning("bot_answer_trace_failed", exc_info=True)
    finally:
        _CURRENT.set(None)


def log_message(**fields: Any) -> None:
    """Store one entry that is not an answer: a skipped message or a bot post. Never raises."""
    try:
        entry = {"at": _now(), **fields}
        for key in ("question", "content"):
            if key in entry:
                entry[key] = _clip(entry[key])
        _store(entry)
    except Exception:
        logger.warning("bot_transcript_log_failed", exc_info=True)


def _store(record: dict) -> None:
    logger.info(
        "bot_transcript_entry",
        **{k: v for k, v in record.items() if k not in ("question", "answer", "content")},
    )
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    task = loop.create_task(_push(record))
    _PENDING.add(task)
    task.add_done_callback(_PENDING.discard)


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
            pipe.ltrim(REDIS_KEY, 0, MAX_ENTRIES - 1)
            await asyncio.wait_for(pipe.execute(), timeout=_REDIS_TIMEOUT_SECONDS)
    except Exception:
        logger.warning("bot_answer_store_failed")
    finally:
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                pass


async def find_message(message_id: str) -> dict | None:
    """The stored entry for Discord message ``message_id``, else ``None``. Never raises."""
    client = None
    try:
        client = _redis_client()
        raw = await asyncio.wait_for(
            client.lrange(REDIS_KEY, 0, MAX_ENTRIES - 1), timeout=_REDIS_TIMEOUT_SECONDS
        )
    except Exception:
        logger.warning("bot_transcript_find_failed")
        return None
    finally:
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                pass
    needle = f'"message_id": "{message_id}"'.encode()
    for item in raw or []:
        if needle not in (item if isinstance(item, bytes) else str(item).encode()):
            continue
        try:
            decoded = json.loads(item)
        except Exception:
            continue
        if isinstance(decoded, dict) and decoded.get("message_id") == message_id:
            return decoded
    return None


def read_recent(
    limit: int = MAX_ENTRIES, *, since: datetime | None = None, until: datetime | None = None
) -> list[dict] | None:
    """Stored entries newest first, inside ``[since, until]`` when given; ``None`` if
    Redis is down."""
    import redis

    try:
        client = redis.Redis.from_url(
            settings.redis_url,
            socket_timeout=_REDIS_TIMEOUT_SECONDS,
            socket_connect_timeout=_REDIS_TIMEOUT_SECONDS,
        )
        try:
            raw = client.lrange(REDIS_KEY, 0, MAX_ENTRIES - 1)
        finally:
            client.close()
    except Exception:
        logger.warning("bot_answer_read_failed")
        return None
    entries: list[dict] = []
    for item in raw or []:  # type: ignore[union-attr]
        try:
            decoded = json.loads(item)
        except Exception:
            continue
        if not isinstance(decoded, dict) or not _in_window(decoded.get("at"), since, until):
            continue
        entries.append(decoded)
        if len(entries) >= limit:
            break
    return entries


def _in_window(at: object, since: datetime | None, until: datetime | None) -> bool:
    if since is None and until is None:
        return True
    try:
        moment = datetime.fromisoformat(str(at))
    except ValueError:
        return False
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return (since is None or moment >= since) and (until is None or moment <= until)
