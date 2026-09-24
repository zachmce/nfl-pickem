"""Regression eval set for the @mention bot (issue #248 item 13).

Runs every case in ``bot_eval_cases.json`` through ``qa.answer_question`` — the full
path, classifier then grounded or open — against the LIVE model and the local DB, and
checks the intent, the tools called and the answer text. See ``scripts/README.md``.

It never starts the bot: a second bot process on the one Discord app takes over prod
replies. Run from ``backend/`` with the repo-root ``.env`` sourced::

    set -a && . ../.env && set +a
    export POSTGRES_HOST=localhost REDIS_URL=redis://127.0.0.1:1/0
    uv run python -m scripts.bot_eval --samples 2
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import json
import re
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

CASES_PATH = Path(__file__).with_name("bot_eval_cases.json")
DEFAULT_REPORT = Path("/tmp/bot_eval_report.json")

ENV_HELP = (
    "bot_eval needs the live LLM and the local DB. From backend/ run:\n"
    "  set -a && . ../.env && set +a\n"
    "  export POSTGRES_HOST=localhost REDIS_URL=redis://127.0.0.1:1/0\n"
    "  docker compose up -d db   # the DB only; NEVER the bot\n"
    "REDIS_URL points at a closed port on purpose: every cache read fails open at once."
)

# Answers that mean the bot gave up. Exact lines from app.bot.qa, read at run time.
_DEGRADE_NAMES = (
    "_ERROR_LINE",
    "_OPEN_DEGRADE_FACT",
    "_INJURIES_DEGRADE_FACT",
    "_WEATHER_DEGRADE_FACT",
    "_NEWS_DEGRADE_FACT",
)


@dataclass
class Case:
    id: str
    question: str
    samples: int
    discord_id: int
    asker_name: str
    history: list[tuple[str, str]] = field(default_factory=list)
    setup: list[str] = field(default_factory=list)
    expect_intent: list[str] = field(default_factory=list)
    open_path: bool | None = None
    tools_any: list[str] = field(default_factory=list)
    tools_all: list[str] = field(default_factory=list)
    tools_none: list[str] = field(default_factory=list)
    no_tools: bool = False
    must_match: list[str] = field(default_factory=list)
    must_not_match: list[str] = field(default_factory=list)
    allow_degrade: bool = False
    requires: list[str] = field(default_factory=list)

    @property
    def group(self) -> str:
        return self.id.split(".", 1)[0]


def _as_list(value: Any) -> list:
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def load_cases(path: Path = CASES_PATH) -> list[Case]:
    """Read and validate the case file. Raises ``ValueError`` on a bad case."""
    data = json.loads(path.read_text())
    defaults = data.get("defaults", {})
    known = set(Case.__dataclass_fields__)
    cases: list[Case] = []
    seen: set[str] = set()
    for raw in data["cases"]:
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"case {raw.get('id')}: unknown keys {sorted(unknown)}")
        merged = {**defaults, **raw}
        case = Case(
            id=merged["id"],
            question=merged["question"],
            samples=int(merged.get("samples", 2)),
            discord_id=int(merged.get("discord_id", 0)),
            asker_name=merged.get("asker_name", "member"),
            history=[(str(r), str(t)) for r, t in merged.get("history", [])],
            setup=_as_list(merged.get("setup")),
            expect_intent=_as_list(merged.get("expect_intent")),
            open_path=merged.get("open_path"),
            tools_any=_as_list(merged.get("tools_any")),
            tools_all=_as_list(merged.get("tools_all")),
            tools_none=_as_list(merged.get("tools_none")),
            no_tools=bool(merged.get("no_tools", False)),
            must_match=_as_list(merged.get("must_match")),
            must_not_match=_as_list(merged.get("must_not_match")),
            allow_degrade=bool(merged.get("allow_degrade", False)),
            requires=_as_list(merged.get("requires")),
        )
        if case.id in seen:
            raise ValueError(f"duplicate case id {case.id}")
        seen.add(case.id)
        for pattern in case.must_match + case.must_not_match:
            re.compile(pattern)
        cases.append(case)
    return cases


def fill(text: str, context: dict[str, Any]) -> str | None:
    """``text`` with ``{name}`` placeholders filled, or ``None`` if one has no value."""
    names = re.findall(r"\{(\w+)\}", text)
    if any(context.get(n) is None for n in names):
        return None
    return text.format(**{n: context[n] for n in names})


def score(
    case: Case,
    *,
    answer: str | None,
    intent: str | None,
    open_path: bool | None,
    tools: list[str],
    degrade_lines: set[str],
) -> list[str]:
    """Every reason one sample fails; an empty list is a pass."""
    reasons: list[str] = []
    if not answer or not answer.strip():
        return ["no answer"]
    if not case.allow_degrade and answer.strip() in degrade_lines:
        reasons.append("degrade line")
    if case.expect_intent and intent not in case.expect_intent:
        reasons.append(f"intent {intent} not in {case.expect_intent}")
    if case.open_path is not None and open_path is not case.open_path:
        reasons.append(f"open_path {open_path} != {case.open_path}")
    if case.tools_any and not set(case.tools_any) & set(tools):
        reasons.append(f"none of {case.tools_any} called (got {tools})")
    missing = [t for t in case.tools_all if t not in tools]
    if missing:
        reasons.append(f"missing tools {missing} (got {tools})")
    forbidden = [t for t in case.tools_none if t in tools]
    if forbidden:
        reasons.append(f"forbidden tools called {forbidden}")
    if case.no_tools and tools:
        reasons.append(f"tools called {tools}")
    for pattern in case.must_match:
        if not re.search(pattern, answer, re.IGNORECASE):
            reasons.append(f"no match /{pattern}/")
    for pattern in case.must_not_match:
        if re.search(pattern, answer, re.IGNORECASE):
            reasons.append(f"forbidden match /{pattern}/")
    return reasons


# ---- live run ------------------------------------------------------------ #

_TRACE: contextvars.ContextVar[dict | None] = contextvars.ContextVar("bot_eval_trace", default=None)
_USAGE = {"calls": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0}


def _instrument() -> set[str]:
    """Wrap the tool registry, the intent validator and the HTTP seam to record calls.

    Returns the degrade lines. Imported here so the offline tests never load the app.
    """
    from app.bot import llm_client, qa, qa_open

    def wrap_tool(tool):
        async def run(*args, **kwargs):
            trace = _TRACE.get()
            if trace is not None:
                trace["tools"].append(tool.name)
            return await tool.run(*args, **kwargs)

        return replace(tool, run=run)

    qa_open.TOOLS = tuple(wrap_tool(t) for t in qa_open.TOOLS)

    validate = qa.validate_classification

    def recording_validate(raw, **kwargs):
        result = validate(raw, **kwargs)
        trace = _TRACE.get()
        if trace is not None and trace["intent"] is None:
            trace["intent"] = result.intent.value
            trace["open_path"] = result.intent in qa._OPEN_INTENTS
        return result

    qa.validate_classification = recording_validate

    open_answer = qa_open.answer_open

    async def recording_open(*args, **kwargs):
        trace = _TRACE.get()
        if trace is not None:
            trace["open_path"] = True
        return await open_answer(*args, **kwargs)

    qa.qa_open.answer_open = recording_open  # type: ignore[method-assign]

    post = llm_client._post_chat

    async def recording_post(*args, **kwargs):
        payload = await post(*args, **kwargs)
        usage = payload.get("usage") if isinstance(payload, dict) else None
        if isinstance(usage, dict):
            _USAGE["calls"] += 1
            _USAGE["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            _USAGE["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            details = usage.get("prompt_tokens_details") or {}
            _USAGE["cached_tokens"] += int(details.get("cached_tokens") or 0)
        return payload

    llm_client._post_chat = recording_post
    return {getattr(qa, name) for name in _DEGRADE_NAMES}


async def _run_sample(case: Case, index: int, context: dict, degrade: set[str]) -> dict:
    from app.bot import qa

    key = f"bot-eval:{case.id}:{index}:{time.time()}"
    history = [(role, fill(text, context) or text) for role, text in case.history]
    for step in (fill(t, context) or t for t in case.setup):
        _TRACE.set({"tools": [], "intent": None, "open_path": None})
        reply = await qa.answer_question(
            step,
            discord_id=case.discord_id,
            history=history,
            conversation_key=key,
            asker_name=case.asker_name,
        )
        history += [("user", f"{case.asker_name}: {step}"), ("assistant", reply)]
    question = fill(case.question, context) or case.question
    trace = {"tools": [], "intent": None, "open_path": None}
    _TRACE.set(trace)
    started = time.monotonic()
    answer = await qa.answer_question(
        question,
        discord_id=case.discord_id,
        history=history,
        conversation_key=key,
        asker_name=case.asker_name,
    )
    if trace["open_path"] is None:
        trace["open_path"] = False
    reasons = score(
        case,
        answer=answer,
        intent=trace["intent"],
        open_path=trace["open_path"],
        tools=trace["tools"],
        degrade_lines=degrade,
    )
    return {
        "case": case.id,
        "sample": index,
        "question": question,
        "answer": answer,
        "intent": trace["intent"],
        "open_path": trace["open_path"],
        "tools": trace["tools"],
        "seconds": round(time.monotonic() - started, 1),
        "passed": not reasons,
        "reasons": reasons,
    }


async def _context() -> dict[str, Any]:
    """Values a case may name as ``{placeholder}``, read from the local DB."""
    from app.bot import db_bridge

    records = await db_bridge.get_league_records_async()
    slate = await db_bridge.get_lines_slate_async()
    week = slate.get("week")
    return {
        "open_week": records.get("open_week"),
        "current_week": week,
        "last_week": week - 1 if isinstance(week, int) and week > 1 else None,
        "later_week": week + 2 if isinstance(week, int) and week <= 16 else None,
    }


def _skip_reason(case: Case, context: dict, env: dict[str, bool]) -> str | None:
    for need in case.requires:
        if not env.get(need, False):
            return f"needs {need}"
    texts = [case.question, *case.setup, *(text for _, text in case.history)]
    if any(fill(text, context) is None for text in texts):
        return "placeholder has no value in this DB"
    return None


async def run(cases: list[Case], *, samples: int | None, concurrency: int) -> dict:
    from app.config import settings

    degrade = _instrument()
    context = await _context()
    env = {"SEARXNG_URL": bool(settings.searxng_url)}
    gate = asyncio.Semaphore(concurrency)
    results: list[dict] = []
    skipped: dict[str, str] = {}

    async def one(case: Case, index: int) -> None:
        async with gate:
            try:
                results.append(await _run_sample(case, index, context, degrade))
            except Exception as exc:  # the report must still land
                results.append(
                    {"case": case.id, "sample": index, "passed": False, "reasons": [repr(exc)]}
                )

    jobs = []
    for case in cases:
        reason = _skip_reason(case, context, env)
        if reason:
            skipped[case.id] = reason
            continue
        jobs += [one(case, i) for i in range(samples or case.samples)]
    started = time.monotonic()
    await asyncio.gather(*jobs)
    results.sort(key=lambda r: (r["case"], r["sample"]))
    return {
        "context": context,
        "seconds": round(time.monotonic() - started, 1),
        "usage": dict(_USAGE),
        "vendor": settings.llm_api_vendor,
        "model": settings.llm_api_model,
        "open_model": settings.llm_api_open_model,
        "skipped": skipped,
        "results": results,
    }


def summarize(report: dict) -> tuple[str, float]:
    """The printed table and the overall pass rate over every sample."""
    by_case: dict[str, list[dict]] = {}
    for r in report["results"]:
        by_case.setdefault(r["case"], []).append(r)
    groups: dict[str, list[int]] = {}
    lines = []
    for case_id, rows in by_case.items():
        passed = sum(r["passed"] for r in rows)
        g = groups.setdefault(case_id.split(".", 1)[0], [0, 0])
        g[0] += passed
        g[1] += len(rows)
        mark = "ok " if passed == len(rows) else "FAIL"
        lines.append(f"{mark} {passed}/{len(rows)}  {case_id}")
        for r in rows:
            for reason in r["reasons"]:
                lines.append(f"       [{r['sample']}] {reason}")
    lines.append("")
    for name, (passed, total) in sorted(groups.items()):
        lines.append(f"group {name:<12} {passed}/{total}")
    total_pass = sum(g[0] for g in groups.values())
    total = sum(g[1] for g in groups.values())
    rate = total_pass / total if total else 0.0
    lines.append(f"TOTAL {total_pass}/{total} = {rate:.1%}   ({report['seconds']} s)")
    for case_id, reason in report["skipped"].items():
        lines.append(f"skipped {case_id}: {reason}")
    return "\n".join(lines), rate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--only", default="", help="run only case ids with these comma-separated prefixes"
    )
    parser.add_argument("--samples", type=int, default=None, help="samples per case")
    parser.add_argument("--min-pass", type=float, default=0.9, help="exit 1 below this rate")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--vendor-note", default="", help="free text stored in the report")
    args = parser.parse_args(argv)

    from app.config import settings

    if not settings.llm_api_server or not settings.llm_api_model:
        print("LLM_API_SERVER / LLM_API_MODEL are not set, so every call returns None.")
        print(ENV_HELP)
        return 2
    print(ENV_HELP, file=sys.stderr)

    prefixes = tuple(p.strip() for p in args.only.split(","))
    cases = [c for c in load_cases() if c.id.startswith(prefixes)]
    report = asyncio.run(run(cases, samples=args.samples, concurrency=args.concurrency))
    report["vendor_note"] = args.vendor_note
    table, rate = summarize(report)
    report["pass_rate"] = rate
    args.report.write_text(json.dumps(report, indent=1, default=str))
    print(table)
    print(f"usage: {report['usage']}")
    print(f"report: {args.report}")
    return 0 if rate >= args.min_pass else 1


if __name__ == "__main__":
    sys.exit(main())
