"""Discord-free OPEN NFL answer path (Path C, 260820-lw6).

The classifier in :mod:`app.bot.qa` offers a fixed menu of grounded intents. Before
this module existed, an off-menu football question had nowhere legal to go, so the
model routed it to the nearest-looking fixed intent — "who is the starting QB for the
Bears" came back as the injury report, because ``injuries`` is the only intent
described as "a team plus a player". This module is that question's legal
destination.

Posture, which is DELIBERATELY different from every other answer path:

* **This is the ONE path allowed to be wrong.** A wrong open answer is colour; a
  wrong spread is a defect. The ten grounded intents keep their current strictness
  and their DB-owned facts are never guessed here — ``OPEN_GUARD`` forbids stating a
  spread, total, score, standing, close time, or any member's pick, and this module
  makes NO ``db_bridge`` call at all, so it cannot read anyone's picks.
* **Best-effort, ``None`` by contract.** :func:`answer_open` NEVER raises; the caller
  falls back to a deterministic degrade line.
* **No ``discord`` import** — the cog :mod:`app.bot.commands.mention_qa` stays the
  only discord-importing module for this feature.
* **Untrusted text crosses the model boundary ONLY via
  :func:`app.bot.chat_personality._fence_untrusted`** — the question and every
  history turn alike.

Tool calling (the 260820 probe measured the served gemma selecting the right tool out
of two, and resolving "Chicago Bears" to ``CHI`` unaided): the model selects from the
FIXED :data:`TOOLS` whitelist BY NAME and NEVER builds a URL — no tool spec may
declare a ``url`` / ``endpoint`` / ``path`` / ``host`` parameter. Every tool's ``run``
must hold the Path B adapter contract: never raises, fails open on Redis, degrades to
``None`` on any HTTP error. The registry ships EMPTY; issue #179 appends to it.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

import structlog

from app.bot import chat_personality, llm_client
from app.bot.personality import compose_prompt

logger = structlog.get_logger(__name__)

# --------------------------------------------------------------------------- #
# The OPEN prompt. NEW constants — the guard text is deliberately NOT reused from
# (nor spliced into) ``personality.py`` / ``chat_personality.py``, whose guard
# constants are asserted byte-identical by ``tests/test_personality.py``.
#
# Every clause is a CONCRETE FULL SENTENCE, never a terse fragment: the local Gemma
# has a recorded habit of inverting or swallowing terse plain-string instructions
# (memory: qa-phrasing-inversion).
# --------------------------------------------------------------------------- #

OPEN_ROLE = (
    "You are answering a league member's open question about the NFL — a question the "
    "app's own data does not cover — using your own football knowledge rather than any "
    "figure read from the app's database."
)

# (a) FORMAT. The 2026-08-20 probe measured the model answering open questions with
# ### headings and bullet lists, and truncating mid-sentence even at a 300-token cap.
# The cap is NOT the fix; THIS clause is. Exposed as its own constant so the
# per-voice prompt test can assert it survives into every composed prompt.
OPEN_FORMAT_CLAUSE = (
    "Write your answer as plain Discord chat prose in a few short sentences, the way a "
    "person types a message into a chat window. Never write a markdown heading line, "
    "never write a bullet list or a numbered list, and never write a bold section "
    "label, because a heading or a bullet inside a Discord message is always wrong "
    "here. Finish the thought you started and then stop, so your answer never breaks "
    "off in the middle of a sentence."
)

# (b) SCOPE. Defense in depth OVER the validator's deterministic topic guard: the
# same probe measured the model writing out a FULL LASAGNA RECIPE under an
# "NFL expert" system prompt, so the persona alone provably does not scope it.
OPEN_SCOPE_CLAUSE = (
    "The only subject you talk about is the NFL, American football, or this pick'em "
    "league. If the member asks you about anything else — cooking, recipes, homework, "
    "code, politics, or any other topic — say plainly that football is the only thing "
    "you talk about, and answer nothing else about that topic."
)

# (c) DB-OWNERSHIP and (d) HONESTY. The numbers the app owns are answered by the
# grounded intents on their own untouched read paths; this path must never compete
# with them, and must never invent a specific figure to fill a gap.
OPEN_OWNERSHIP_CLAUSE = (
    "Never state a point spread, an over/under total, a game score, a standings "
    "position, a pick deadline or close time, or any league member's pick, because "
    "every one of those comes from the app's own data and other parts of the bot "
    "answer them."
)

OPEN_HONESTY_CLAUSE = (
    "When you are not certain of something, say so plainly instead of inventing a "
    "specific statistic, date, or number."
)

OPEN_GUARD = (
    f"{OPEN_FORMAT_CLAUSE} {OPEN_SCOPE_CLAUSE} {OPEN_OWNERSHIP_CLAUSE} "
    f"{OPEN_HONESTY_CLAUSE} Use at most one emoji."
)

# --------------------------------------------------------------------------- #
# Deterministic output scrub — the belt-and-suspenders backstop behind
# OPEN_FORMAT_CLAUSE. The format instruction is the PRIMARY fix; this is what
# catches the round where the model reaches for a heading anyway.
# --------------------------------------------------------------------------- #

# Leading list markers the scrub drops (ASCII dash/asterisk/plus plus the common
# unicode bullets). Only stripped when followed by whitespace, so inline emphasis
# (``**the** guy``) and interior punctuation (``3-1``) are left alone.
_BULLET_MARKERS = "-*+•‣▪·"
_NUMBERED_LIST_RE = re.compile(r"\d+\.\s+")


def _strip_markdown_structure(text: str) -> str:
    """Strip markdown STRUCTURE (headings / list markers) from ``text``. Pure.

    Per line: drops a leading run of ``#`` plus any following whitespace, then a
    leading bullet marker plus its following space (a line that is nothing BUT a
    marker is dropped whole), then a leading ``<digits>.`` plus its following space. Inline emphasis and interior punctuation are untouched. Runs
    of blank lines collapse to one and the result is stripped, so a reply that was
    nothing but structure markers scrubs down to ``""`` (which the caller treats as a
    miss).
    """
    scrubbed: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        without_hashes = line.lstrip("#")
        if without_hashes != line:
            line = without_hashes.lstrip()
        if line and line[0] in _BULLET_MARKERS and (len(line) == 1 or line[1].isspace()):
            # A lone marker (len 1) is pure structure with no content — drop it whole.
            line = line[1:].lstrip()
        else:
            match = _NUMBERED_LIST_RE.match(line)
            if match is not None:
                line = line[match.end() :].lstrip()
        scrubbed.append(line)

    collapsed: list[str] = []
    for line in scrubbed:
        if not line and (not collapsed or not collapsed[-1]):
            continue  # drop leading blanks and collapse blank runs to one
        collapsed.append(line)
    return "\n".join(collapsed).strip()


def _message_content(message: dict) -> str | None:
    """Return the stripped text content of an assistant ``message``, or ``None``.

    Defensive by design: the message came off the wire, so ``content`` may be absent,
    ``None`` (the normal shape when the model emitted only tool calls), or a non-str.
    """
    content = message.get("content")
    if not isinstance(content, str):
        return None
    stripped = content.strip()
    return stripped or None


# --------------------------------------------------------------------------- #
# The TOOL WHITELIST. The model selects from this fixed registry BY NAME and NEVER
# builds a URL — no spec may declare a ``url`` / ``endpoint`` / ``path`` / ``host``
# parameter, so a model-chosen call can never become a model-chosen request target
# (T-lw6-02). Every ``run`` must hold the Path B adapter contract (see
# ``app.services.espn_extra.fetch_injuries``): NEVER raises, fails open on Redis,
# degrades to ``None`` on any HTTP error.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Tool:
    """One whitelisted data call the model may name.

    ``name`` is the EXACT function name the model must emit (lookup is an exact
    string match — never a prefix or fuzzy match). ``spec`` is the OpenAI-style
    function schema with TYPED parameters; only the names it declares are ever passed
    through to ``run``. ``run`` is the awaitable adapter that performs the call.
    """

    name: str
    spec: dict
    run: Callable[..., Awaitable[object]]


# EMPTY as shipped: Path C answers from model knowledge first. Companion issue #179
# is the task that appends the first real tools (ESPN roster / nflverse stats).
TOOLS: tuple[_Tool, ...] = ()

# An unbounded loop on a quantized local model is the main NEW failure surface Path C
# introduces (the model can keep asking for one more call forever), so the loop is
# bounded twice over: by rounds AND by wall clock.
_MAX_TOOL_ROUNDS = 3
_TOOL_BUDGET_SECONDS = 20.0

# Each failure mode gets its OWN fixed payload so the model is TOLD what happened
# instead of being handed silence (silence reads as "the data says nothing", which is
# how an invented answer gets written). Concrete full sentences, never fragments.
_UNKNOWN_TOOL_PAYLOAD = (
    "That tool does not exist. Answer the question from your own football knowledge "
    "instead, and do not try to call it again."
)
_BAD_ARGUMENTS_PAYLOAD = (
    "The arguments for that tool could not be read as JSON. Answer the question from "
    "your own football knowledge instead."
)
_NO_DATA_PAYLOAD = (
    "That tool returned no data right now. Answer the question from your own football "
    "knowledge instead, and say plainly if you are not sure."
)


def _coerce_str(value: object) -> str:
    """Coerce a JSON scalar to ``str``; anything structural is UNCOERCIBLE."""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    raise ValueError("not a string")


def _coerce_int(value: object) -> int:
    """Coerce a JSON scalar to ``int``; a bool is deliberately NOT an int here."""
    if isinstance(value, bool):
        raise ValueError("a boolean is not an integer")
    if isinstance(value, (int, float, str)):
        return int(value)
    raise ValueError("not an integer")


def _coerce_float(value: object) -> float:
    """Coerce a JSON scalar to ``float``; a bool is deliberately NOT a number here."""
    if isinstance(value, bool):
        raise ValueError("a boolean is not a number")
    if isinstance(value, (int, float, str)):
        return float(value)
    raise ValueError("not a number")


def _coerce_bool(value: object) -> bool:
    """Strict bool coercion — anything that is not already a bool is UNCOERCIBLE."""
    if isinstance(value, bool):
        return value
    raise ValueError("not a boolean")


# JSON-schema type -> coercer. A declared parameter whose value will not coerce is
# DROPPED (never passed through raw), so a tool only ever sees what its spec promised.
# Each coercer NARROWS with isinstance before converting — a bare ``int(value)`` on an
# ``object`` both trips the type gate and raises on a dict.
_TYPE_COERCERS: dict[str, Callable[[object], object]] = {
    "string": _coerce_str,
    "integer": _coerce_int,
    "number": _coerce_float,
    "boolean": _coerce_bool,
}


def _lookup_tool(name: str) -> _Tool | None:
    """Return the whitelisted tool named EXACTLY ``name``, or ``None``. Pure."""
    for tool in TOOLS:
        if tool.name == name:
            return tool
    return None


def _declared_properties(tool: _Tool) -> dict:
    """Return the ``properties`` mapping the tool's own spec declares (or ``{}``)."""
    function = tool.spec.get("function")
    if not isinstance(function, dict):
        return {}
    parameters = function.get("parameters")
    if not isinstance(parameters, dict):
        return {}
    properties = parameters.get("properties")
    return properties if isinstance(properties, dict) else {}


def _filter_arguments(tool: _Tool, decoded: dict) -> dict:
    """Filter ``decoded`` down to ``tool``'s DECLARED parameters, type-coerced. Pure.

    Anything the spec does not declare is dropped (this is what stops the model from
    smuggling an undeclared ``url`` past a name-only whitelist), and a declared
    parameter whose value will not coerce to its declared type is dropped too. A
    declared parameter with no ``type`` passes through unchanged.
    """
    properties = _declared_properties(tool)
    filtered: dict = {}
    for key, value in decoded.items():
        declared = properties.get(key) if isinstance(key, str) else None
        if not isinstance(declared, dict):
            continue  # undeclared parameter name — dropped
        json_type = declared.get("type")
        coercer = _TYPE_COERCERS.get(json_type) if isinstance(json_type, str) else None
        if coercer is None:
            filtered[key] = value
            continue
        try:
            filtered[key] = coercer(value)
        except Exception:
            continue  # uncoercible — dropped
    return filtered


def _tool_message(call_id: str, name: str, result: object) -> dict:
    """Build the tool-role turn carrying the call id, the name and a JSON result."""
    try:
        content = json.dumps(result)
    except TypeError, ValueError:
        content = json.dumps(str(result))
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}


async def _resolve_tool_call(call: object, *, round_index: int) -> dict:
    """Resolve ONE model-emitted tool call into its tool-role result turn.

    Never raises: an unknown name, unreadable arguments, and a tool that blows up each
    produce their own fixed payload turn so the loop always has something to feed back.
    """
    call_id = ""
    name = ""
    arguments = ""
    if isinstance(call, dict):
        raw_id = call.get("id")
        call_id = raw_id if isinstance(raw_id, str) else ""
        function = call.get("function")
        if isinstance(function, dict):
            raw_name = function.get("name")
            name = raw_name if isinstance(raw_name, str) else ""
            raw_arguments = function.get("arguments")
            arguments = raw_arguments if isinstance(raw_arguments, str) else ""

    tool = _lookup_tool(name)
    if tool is None:
        logger.warning("qa_open_tool_unknown", tool=name, round=round_index)
        return _tool_message(call_id, name, _UNKNOWN_TOOL_PAYLOAD)

    try:
        decoded = json.loads(arguments) if arguments.strip() else {}
    except Exception:
        decoded = None
    if not isinstance(decoded, dict):
        logger.warning("qa_open_tool_bad_arguments", tool=name, round=round_index)
        return _tool_message(call_id, name, _BAD_ARGUMENTS_PAYLOAD)

    logger.info("qa_open_tool_call", tool=name, round=round_index)
    try:
        result = await tool.run(**_filter_arguments(tool, decoded))
    except Exception:
        # Belt-and-suspenders over the never-raise adapter contract.
        logger.warning("qa_open_tool_failed", tool=name, round=round_index, exc_info=True)
        result = None
    if result is None:
        return _tool_message(call_id, name, _NO_DATA_PAYLOAD)
    return _tool_message(call_id, name, result)


async def _run_tool_loop(messages: list[dict], *, system_prompt: str) -> str | None:
    """Drive the open-path model round(s) and return the final text, or ``None``.

    With the SHIPPED empty :data:`TOOLS` registry this is exactly ONE
    :func:`app.bot.llm_client.open_chat` call with ``tools=None`` — byte-identical to
    the zero-tool behavior, with no extra round and no latency cost.

    With a non-empty registry it loops at most :data:`_MAX_TOOL_ROUNDS` times against a
    :data:`_TOOL_BUDGET_SECONDS` wall clock (checked BEFORE each new round). A round
    whose message carries no tool calls returns its text immediately. A round WITH tool
    calls replays the model's own turn verbatim and appends one resolved tool-role turn
    per call. When the loop ends for ANY reason — round cap or budget — exactly ONE
    final ``open_chat`` call is made with ``tools=None``, which is what stops a capped
    loop from returning nothing. Returns ``None`` if any round's call returns ``None``.
    """
    deadline = time.monotonic() + _TOOL_BUDGET_SECONDS

    if not TOOLS:
        message = await llm_client.open_chat(messages, system_prompt=system_prompt, tools=None)
        if message is None:
            return None
        return _message_content(message)

    specs = [tool.spec for tool in TOOLS]
    working = list(messages)
    for round_index in range(_MAX_TOOL_ROUNDS):
        if time.monotonic() >= deadline:
            logger.warning("qa_open_tool_budget_exhausted", round=round_index)
            break
        message = await llm_client.open_chat(working, system_prompt=system_prompt, tools=specs)
        if message is None:
            return None
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            return _message_content(message)  # the model answered in text — done
        working.append(message)
        for call in tool_calls:
            working.append(await _resolve_tool_call(call, round_index=round_index))
    else:
        logger.info("qa_open_tool_round_cap_reached", rounds=_MAX_TOOL_ROUNDS)

    # The forced close: tools are withheld so the model MUST produce text.
    final = await llm_client.open_chat(working, system_prompt=system_prompt, tools=None)
    if final is None:
        return None
    return _message_content(final)


async def answer_open(
    question: str,
    *,
    voice: str,
    history: Sequence[tuple[str, str]] = (),
) -> str | None:
    """Answer an off-menu NFL ``question`` in ``voice`` as plain prose, or ``None``.

    Fences the question AND every ``history`` turn through
    :func:`app.bot.chat_personality._fence_untrusted` (the only way untrusted text is
    allowed across the model boundary), builds the message list as the fenced history
    followed by the fenced question, composes the system prompt as
    ``compose_prompt(voice, OPEN_ROLE, OPEN_GUARD)``, runs the bounded tool loop
    (:func:`_run_tool_loop`), and scrubs the reply through :func:`_strip_markdown_structure`.

    ``history`` is a sequence of ``(role, text)`` turns oldest-first; any role that is
    not exactly ``assistant`` is coerced to ``user``, so a smuggled ``system`` turn can
    never become an instruction. Returns ``None`` on any failure or an empty scrub —
    the caller falls back to its deterministic degrade line. NEVER raises.
    """
    try:
        messages: list[dict] = []
        for role, text in history:
            fenced_turn = chat_personality._fence_untrusted(text)
            if not fenced_turn:
                continue
            safe_role = "assistant" if role == "assistant" else "user"
            messages.append({"role": safe_role, "content": fenced_turn})
        messages.append({"role": "user", "content": chat_personality._fence_untrusted(question)})

        system_prompt = compose_prompt(voice, OPEN_ROLE, OPEN_GUARD)
        content = await _run_tool_loop(messages, system_prompt=system_prompt)
        if content is None:
            return None
        return _strip_markdown_structure(content) or None
    except Exception:
        # Best-effort by contract — a surprise raise degrades to the caller's
        # deterministic line and never escapes into the gateway loop.
        logger.warning("qa_open_answer_failed", exc_info=True)
        return None
