"""Discord-free "brain" for the inbound @mention Q&A feature (Path A v1, 260709-k5w).

A league member @mentions the bot with a free-text question; this module turns
that question into a public, in-character line built ONLY from stored DB data,
WITHOUT weakening the facts-first / leak-safe invariants the rest of the chat
layer holds. It mirrors the import posture of :mod:`app.bot.chat_personality`:
``structlog`` only, NO ``discord`` import — the ONLY module that imports ``discord``
for this feature is the cog :mod:`app.bot.commands.mention_qa`.

The pipeline is the two-Gemma-call shape from the locked design
(``.planning/notes/discord-query-bot-design.md``):

    question -> 1. CLASSIFY (Gemma, JSON-only via llm_client.classify)
             -> 2. VALIDATE (pure Python — the WHOLE safety story)
             -> 3. QUERY   (deterministic db_bridge read -> fact string)
             -> 4. PHRASE  (llm_client.phrase in the active voice)

Task 1 (this pass) owns steps 1 and 2: the fixed :class:`QaIntent` enum, the
best-effort :func:`classify_question` (routed through the DEDICATED deterministic
JSON extraction seam ``llm_client.classify`` — NOT ``phrase``), and the pure,
DB-free :func:`validate_classification` that coerces anything off-enum / invalid /
non-real-team to :attr:`QaIntent.unknown` (belt-and-suspenders over the model's own
``unknown``). Steps 3 and 4 (the intent handlers + phrasing orchestrator) are added
in Task 2.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

import structlog

from app.bot import chat_personality, llm_client

logger = structlog.get_logger(__name__)


class QaIntent(str, Enum):
    """The FIXED intent enum the classifier may emit (nothing else is legal).

    The two non-answer members are FIRST-CLASS values so the model has a legal way
    to say "planned" (``coming_soon``) or "I don't know" (``unknown``) instead of
    being forced into a wrong pick.
    """

    pick_status = "pick_status"
    standings = "standings"
    lines_slate = "lines_slate"
    scores = "scores"
    coming_soon = "coming_soon"
    unknown = "unknown"


# Which validated intents carry which optional params. A field irrelevant to the
# resolved intent is DROPPED (set to None), not treated as an error.
_TEAM_INTENTS = frozenset({QaIntent.lines_slate})
_WEEK_INTENTS = frozenset({QaIntent.pick_status, QaIntent.lines_slate, QaIntent.scores})
_SUBJECT_INTENTS = frozenset({QaIntent.lines_slate, QaIntent.unknown, QaIntent.coming_soon})

# Sane NFL week bounds for the coerced ``week`` param (regular season + playoffs);
# anything outside becomes None.
_MIN_NFL_WEEK = 1
_MAX_NFL_WEEK = 22


@dataclass(frozen=True)
class QaResult:
    """The validated, normalized classification — the safe output of the seam.

    ``intent`` is always a real :class:`QaIntent` member; the params are already
    scrubbed (team resolved to a real 32-team token or None, week in a sane range or
    None, irrelevant fields dropped).
    """

    intent: QaIntent
    team: str | None = None
    week: int | None = None
    subject: str | None = None


# The JSON-ONLY classifier system prompt. Distinct from the phrasing prompts: it
# instructs the model to emit ONLY a compact object and NOTHING else. Kept terse and
# deliberately free of any example team name that could be parroted back as a fact.
CLASSIFIER_SYSTEM_PROMPT = (
    "You classify a league member's NFL pick'em question into a fixed intent. "
    "Reply with ONLY a compact JSON object and NOTHING else — no prose, no code "
    "fence, no explanation. The object has exactly these keys: "
    '"intent", "team", "week", "subject". '
    '"intent" MUST be one of: pick_status (their own pick/lock status), standings '
    "(the leaderboard or someone's rank), lines_slate (the spread, total, this "
    "week's games, or when the window closes), scores (final or in-progress game "
    "scores), coming_soon (a recognized but unsupported topic: injuries, weather, "
    "news, line movement, or a who-will-win prediction), unknown (anything you are "
    'not sure about). "team" is a team name or abbreviation the question is about, '
    'or null. "week" is an integer week number, or null. "subject" is a short noun '
    "phrase describing what they asked, or null. When in doubt use unknown."
)


async def classify_question(question: str) -> dict | None:
    """Classify ``question`` into a raw intent dict, or ``None`` on any failure.

    Best-effort (mirrors ``llm_client`` returning ``None``): the untrusted question
    is FENCED via :func:`app.bot.chat_personality._fence_untrusted` before it crosses
    the model boundary, then handed to the DEDICATED deterministic extraction seam
    :func:`app.bot.llm_client.classify` — NOT ``phrase`` (which would append the
    closer-variety chat directive and sample with chat-variety knobs, sabotaging a
    JSON-only instruction). Parses the returned string as JSON and returns the parsed
    dict, or ``None`` when the client returned ``None`` or the content did not parse.
    NEVER raises — the caller (and the validator) treat ``None`` as ``unknown``.
    """
    fenced = chat_personality._fence_untrusted(question)
    try:
        raw = await llm_client.classify(fenced, system_prompt=CLASSIFIER_SYSTEM_PROMPT)
    except Exception:
        # The seam is best-effort None-by-contract, but guard anyway so a surprise
        # raise degrades to unknown rather than escaping into the caller.
        logger.warning("qa_classify_failed", exc_info=True)
        return None
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_team(value: object, known_team_tokens: set[str]) -> str | None:
    """Resolve ``value`` to a real 32-team token, or ``None`` if it is not real.

    Pure and case/whitespace insensitive: coerces the input to an upper-cased,
    stripped token and returns it ONLY when it is a member of ``known_team_tokens``
    (which the caller supplies with BOTH abbreviations and display-name tokens, all
    real teams). Anything that does not normalize to a real token — a made-up team, a
    non-string, blank — returns ``None``, which the validator turns into
    :attr:`QaIntent.unknown` for a team-bearing intent.
    """
    if not isinstance(value, str):
        return None
    token = value.strip().upper()
    if not token:
        return None
    real = {t.strip().upper() for t in known_team_tokens}
    return token if token in real else None


def _coerce_week(value: object) -> int | None:
    """Coerce ``value`` to an int in the sane NFL week range, else ``None``.

    Accepts an int or an all-digit string; anything outside ``1..22`` or non-numeric
    becomes ``None`` (the reader then resolves the current week itself).
    """
    if isinstance(value, bool):  # bool is an int subclass — reject it explicitly
        return None
    if isinstance(value, int):
        week = value
    elif isinstance(value, str) and value.strip().isdigit():
        week = int(value.strip())
    else:
        return None
    return week if _MIN_NFL_WEEK <= week <= _MAX_NFL_WEEK else None


def _coerce_subject(value: object) -> str | None:
    """Coerce ``value`` to a short stripped subject string, else ``None``."""
    if not isinstance(value, str):
        return None
    subject = value.strip()
    return subject or None


def validate_classification(raw: object, *, known_team_tokens: set[str]) -> QaResult:
    """Coerce an untrusted classifier output into a safe :class:`QaResult`.

    The WHOLE safety story (belt-and-suspenders over the model's own ``unknown``).
    Pure, synchronous, DB-free and network-free — ``known_team_tokens`` (the real
    32-team abbreviation + display-name token set) is supplied by the caller, so this
    stays unit-testable with a hand-built set. Coerces to
    ``QaResult(intent=QaIntent.unknown, ...)`` on ANY of:

    * ``raw`` is not a dict (absent / invalid JSON, non-dict input);
    * ``intent`` is missing / absent;
    * ``intent`` is not one of the :class:`QaIntent` values (off-enum);
    * a non-null ``team`` on a team-bearing intent does not normalize to a member of
      ``known_team_tokens`` (a non-real team).

    ``coming_soon`` is a legal enum value (recognized-but-planned) and is NEVER
    coerced. Params are scrubbed to the resolved intent: ``team`` is dropped for an
    intent that takes no team, ``week`` is coerced to a sane range or ``None``, and
    ``subject`` is kept only where relevant.
    """
    if not isinstance(raw, dict):
        return QaResult(intent=QaIntent.unknown)

    raw_intent = raw.get("intent")
    try:
        intent = QaIntent(raw_intent)
    except ValueError:
        return QaResult(intent=QaIntent.unknown)

    # Resolve the team for team-bearing intents. A present-but-non-real team is a
    # coercion trigger: the model named a game we cannot trust, so fall to unknown.
    team: str | None = None
    if intent in _TEAM_INTENTS:
        raw_team = raw.get("team")
        if raw_team is not None:
            team = _normalize_team(raw_team, known_team_tokens)
            if team is None:
                return QaResult(intent=QaIntent.unknown)

    week = _coerce_week(raw.get("week")) if intent in _WEEK_INTENTS else None
    subject = _coerce_subject(raw.get("subject")) if intent in _SUBJECT_INTENTS else None

    return QaResult(intent=intent, team=team, week=week, subject=subject)
