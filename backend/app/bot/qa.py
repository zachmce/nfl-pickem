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
from app.bot.personality import compose_prompt

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


# --------------------------------------------------------------------------- #
# Task 2 — deterministic intent handlers + the phrasing orchestrator.
#
# Facts-first, leak-safe, stateless: the bot owns a deterministic FACT string
# (built from a display-only db_bridge read) and the LLM only phrases it in the
# active voice. On any phrase failure the deterministic fact line is returned so
# exactly one line always lands; on any seam raise a deterministic error line is
# returned (answer_question NEVER raises into the gateway loop).
# --------------------------------------------------------------------------- #

# The Q&A ROLE (event-specific context) + the byte-distinct facts-first GUARD. The
# GUARD is QA-specific: it does NOT reuse the leak-token-avoidance wording from
# chat_personality verbatim — it states the facts-first / anti-invention / never
# reveal-another-player's-hidden-pick / one-line discipline in its own words.
QA_ROLE = (
    "You are answering a league member's question about the NFL pick'em game, using "
    "ONLY the facts supplied to you below. The facts were read from the app's own data."
)

QA_GUARD = (
    "State the supplied facts plainly and FIRST, then add a little personality — "
    "flavor must NEVER replace the answer. Invent NOTHING beyond the facts you are "
    "given: no stat, spread, total, score, standing, close time, or pick that is not "
    "written in the facts. NEVER reveal, guess, or hint at another player's hidden "
    "pick — you are only ever given the asker's own status. If the facts are a decline "
    "or a 'not yet supported' note, deliver that in character without inventing an "
    "answer. Reply with ONE short line and at most one emoji."
)

# Deterministic short-circuit line for an unregistered asker (no LLM call needed).
_REGISTER_LINE = "You need a pick'em account first — run /register to get set up."

# Deterministic error line — the best-effort fallback when a db seam raises. Never
# leaks anything; just keeps the listener from ever raising into the gateway loop.
_ERROR_LINE = "Something went sideways pulling that up — give it another shot in a bit."

# Tier-2 (coming_soon) wink: recognized-but-planned, NO capability menu, NO DB read.
_COMING_SOON_FACT = (
    "That's not something tracked yet — injuries, weather, news, line movement, and "
    "who-wins predictions are on the roadmap, not in the playbook today."
)

# Tier-3 (unknown) decline: the capability MENU (the four things it can answer) + a
# bug-the-developer nudge. The menu appears ONLY on unknown.
_UNKNOWN_FACT = (
    "Not sure how to answer that one. What can be answered: the asker's own pick / "
    "lock status, the season standings, this week's lines and slate (and when picks "
    "close), and game scores. For anything else, bug the developer to build it."
)

# Stateless soft-decline for a single-game lines question with no team resolved —
# no ask-and-wait, no pending-slot state.
_SOFT_DECLINE_FACT = "Name a team and the line for that game can be pulled up."

# Subject hints that a lines question is about a SINGLE game (so a missing team is a
# stateless soft-decline, not a whole-slate dump).
_SINGLE_GAME_HINTS = (
    "spread",
    "line",
    "total",
    "over",
    "under",
    "favorite",
    "underdog",
    "moneyline",
    "odds",
)


def _wants_single_game(subject: str | None) -> bool:
    """Whether a teamless lines question implies ONE game (-> soft-decline)."""
    if not subject:
        return False
    low = subject.lower()
    return any(hint in low for hint in _SINGLE_GAME_HINTS)


def _pick_status_fact(status: dict) -> str:
    """Build the asker's OWN pick-status fact (registered case)."""
    name = status.get("display_name")
    if status.get("complete"):
        return f"{name}'s standard card is complete — every pick is in for the week."
    remaining = status.get("remaining_labels") or []
    if remaining:
        return f"{name} still needs to make these picks this week: {', '.join(remaining)}."
    return f"{name}'s card is not complete yet."


def _standings_fact(ctx: dict) -> str:
    """Build the standings fact from the leaders context (display-only)."""
    leader = ctx.get("leader")
    if not leader:
        return "No standings yet — nobody has a graded pick."
    leader_total = ctx.get("leader_total")
    runner_up = ctx.get("runner_up")
    gap = ctx.get("gap")
    if runner_up and gap == 0:
        return f"{leader} and {runner_up} are tied for the lead with {leader_total}."
    parts = [f"{leader} leads the season with {leader_total}."]
    if runner_up:
        parts.append(f"{runner_up} is {gap} back in second.")
    return " ".join(parts)


def _one_game_line_fact(game: dict, week: int | None, close_at: object) -> str:
    """A single game's line fact (matchup + favorite/spread + total + close)."""
    away = game.get("away")
    home = game.get("home")
    parts = [f"Week {week}: {away} at {home}."]
    favorite = game.get("favorite")
    spread = game.get("spread")
    if favorite and spread:
        parts.append(f"{favorite} favored by {spread}.")
    total = game.get("total")
    if total:
        parts.append(f"Total is {total}.")
    if close_at:
        parts.append(f"Picks close {close_at}.")
    return " ".join(parts)


def _slate_fact(slate: dict) -> str:
    """Build the lines/slate fact — one game's line or a whole-week summary."""
    games = slate.get("games") or []
    week = slate.get("week")
    close_at = slate.get("close_at")
    if not games:
        return f"No games are posted for week {week} yet."
    if len(games) == 1:
        return _one_game_line_fact(games[0], week, close_at)

    summaries = []
    for game in games:
        matchup = f"{game.get('away')}@{game.get('home')}"
        favorite = game.get("favorite")
        spread = game.get("spread")
        if favorite and spread:
            matchup += f" ({favorite} -{spread})"
        summaries.append(matchup)
    fact = f"Week {week} slate: " + "; ".join(summaries) + "."
    if close_at:
        fact += f" Picks close {close_at}."
    return fact


def _scores_fact(scores: dict) -> str:
    """Build the scores fact (final + in-progress) for the week (display-only)."""
    games = scores.get("games") or []
    week = scores.get("week")
    if not games:
        return f"No scores yet for week {week}."
    parts = []
    for game in games:
        tag = "final" if game.get("status") == "FINAL" else "in progress"
        parts.append(
            f"{game.get('away')} {game.get('away_score')} at "
            f"{game.get('home')} {game.get('home_score')} ({tag})"
        )
    return f"Week {week} scores: " + "; ".join(parts) + "."


async def _build_fact(result: QaResult, *, discord_id: int) -> str | None:
    """Route a validated intent to its deterministic reader and build the FACT.

    Returns the fact string to phrase, or ``None`` for the pick_status
    unregistered short-circuit (the caller returns :data:`_REGISTER_LINE` directly,
    no LLM). LEAK INVARIANT: no branch here reads another user's picks —
    ``pick_status`` is asker-only (``get_pick_status_async`` takes ONLY the asker's
    ``discord_id``) and every other intent is already-public data.
    """
    from app.bot import db_bridge

    if result.intent is QaIntent.pick_status:
        status = await db_bridge.get_pick_status_async(discord_id)
        if not status.get("registered"):
            return None  # -> _REGISTER_LINE (deterministic, no LLM)
        return _pick_status_fact(status)

    if result.intent is QaIntent.standings:
        return _standings_fact(await db_bridge.get_leaders_context_async())

    if result.intent is QaIntent.lines_slate:
        # Stateless missing-param: a single-game line question with no team resolved
        # gets a soft decline — never an ask-and-wait / pending slot.
        if result.team is None and _wants_single_game(result.subject):
            return _SOFT_DECLINE_FACT
        return _slate_fact(await db_bridge.get_lines_slate_async(team_abbr=result.team))

    if result.intent is QaIntent.scores:
        return _scores_fact(await db_bridge.get_week_scores_async())

    if result.intent is QaIntent.coming_soon:
        return _COMING_SOON_FACT  # Tier 2 — no DB read

    return _UNKNOWN_FACT  # Tier 3 — decline + capability menu


async def answer_question(question: str, *, discord_id: int) -> str:
    """Answer a league member's @mention ``question`` as one public in-voice line.

    The best-effort orchestrator (mirrors ``embellish_chat``'s guarded posture;
    NEVER raises): classify -> validate (with the real-team token set) -> route to
    a deterministic reader and build a FACT -> phrase the fact in the active voice.
    On the pick_status unregistered path returns a deterministic /register line with
    no LLM call. When ``llm_client.phrase`` returns ``None`` returns the
    deterministic FACT string itself so exactly one line always lands. On ANY seam
    raise returns a deterministic error line — the on_message listener that calls
    this never sees an exception.
    """
    try:
        from app.bot import db_bridge

        raw = await classify_question(question)
        known_team_tokens = await db_bridge.get_real_team_tokens_async()
        result = validate_classification(raw, known_team_tokens=known_team_tokens)

        fact = await _build_fact(result, discord_id=discord_id)
        if fact is None:
            # pick_status, unregistered asker — deterministic, no phrasing.
            return _REGISTER_LINE

        voice = await db_bridge.resolve_active_voice_async()
        system_prompt = compose_prompt(voice, QA_ROLE, QA_GUARD)
        phrased = await llm_client.phrase(fact, system_prompt=system_prompt)
        return phrased if phrased is not None else fact
    except Exception:
        # A classify / db / phrase hiccup must never escape into the gateway loop.
        logger.warning("answer_question_failed", exc_info=True)
        return _ERROR_LINE
