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
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import TYPE_CHECKING

import structlog

from app.bot import chat_personality, llm_client
from app.bot.personality import compose_prompt

if TYPE_CHECKING:
    from app.scoreboard.types import ScoreboardOdds
    from app.services.weather import Stadium

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
    injuries = "injuries"
    weather = "weather"
    news = "news"
    prediction = "prediction"
    coming_soon = "coming_soon"
    unknown = "unknown"


# Which validated intents carry which optional params. A field irrelevant to the
# resolved intent is DROPPED (set to None), not treated as an error. ``injuries`` and
# ``weather`` are team-BEARING and team-REQUIRED (a teamless question soft-declines) —
# they reuse the same real-team validation + coercion path as ``lines_slate``.
# ``news`` is team-BEARING but team-OPTIONAL: a present team is coerced to a real token
# (non-real -> unknown), a null team stays valid and yields the LEAGUE answer downstream.
_TEAM_INTENTS = frozenset(
    {
        QaIntent.lines_slate,
        QaIntent.injuries,
        QaIntent.weather,
        QaIntent.news,
        QaIntent.prediction,
    }
)
_WEEK_INTENTS = frozenset({QaIntent.pick_status, QaIntent.lines_slate, QaIntent.scores})
_SUBJECT_INTENTS = frozenset(
    {QaIntent.lines_slate, QaIntent.unknown, QaIntent.coming_soon, QaIntent.news}
)

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
    "scores), injuries (a team's injury report — who is hurt, out, doubtful, or "
    "questionable), weather (the game-time forecast or conditions for a team's "
    "game), news (recent ESPN headlines about a specific team or the league), "
    "prediction (who will win a specific team's game — the pick, the cover or "
    "margin read, who covers the spread), "
    "coming_soon (a recognized but unsupported topic: line movement), "
    "unknown (anything you are "
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
    except ValueError, TypeError:
        return None
    return parsed if isinstance(parsed, dict) else None


# Curated deterministic slang -> canonical-abbreviation aliases (DATA, not logic).
# Keys are LOWERCASE common nicknames the classifier's ``known_team_tokens`` set does
# NOT already carry (abbreviations + display-name words are covered by the real set);
# values are the UPPERCASE canonical 32-team abbreviation. This is a pure FALLBACK
# folded into :func:`_normalize_team` so a nickname resolves DETERMINISTICALLY across
# every team-bearing intent (injuries/weather/news/lines_slate) instead of depending on
# whether the local model happens to translate the slang that call.
#
# AMBIGUITY DISCIPLINE (mirrors ``team_emoji._ABBR_TO_MARKET_PHRASES``): any slang that
# maps to more than one team is DELIBERATELY EXCLUDED — e.g. "birds" (Eagles / Cardinals
# / Ravens / Seahawks) is absent, so it never resolves. Every entry is a single,
# well-known, unambiguous nickname. The real-set guard in :func:`_normalize_team` is the
# belt-and-suspenders backstop: an alias only resolves if its target is itself a real
# token, so a typo here can never emit a non-real team.
_TEAM_ALIASES: dict[str, str] = {
    # KEYS ARE NORMALIZED to lowercase alphanumerics ONLY (no spaces/hyphens/
    # apostrophes) — _normalize_team strips the input the same way, so "big blue",
    # "gang-green" and "'boys" all match. AMBIGUOUS slang that maps to >1 team is
    # DELIBERATELY OMITTED (e.g. "birds" = ARI/ATL/BAL/PHI/SEA; "cats" = CAR/CIN/DET/
    # JAX; bare "purple" = MIN/BAL), mirroring team_emoji's ambiguity discipline. City/
    # display-name WORDS that already resolve as real tokens ("vegas", "philadelphia")
    # are not repeated here. Crude/derogatory fan slang is intentional — these are
    # inputs a real NFL fan types; the deterministic map just routes them to the team.
    # ARI
    "cards": "ARI",
    "zona": "ARI",
    "redbirds": "ARI",
    "birdgang": "ARI",
    # ATL
    "falcs": "ATL",
    "dirtybirds": "ATL",
    # BUF
    "mafia": "BUF",
    "billsmafia": "BUF",
    # CHI
    "dabears": "CHI",
    # CIN
    "bungles": "CIN",
    "whodey": "CIN",
    # CLE
    "dawgs": "CLE",
    "dawgpound": "CLE",
    "factoryofsadness": "CLE",
    # DAL
    "boys": "DAL",
    "cowgirls": "DAL",
    "jerryworld": "DAL",
    "americasteam": "DAL",
    # DEN
    "donkeys": "DEN",
    "orangecrush": "DEN",
    # GB
    "pack": "GB",
    "cheeseheads": "GB",
    # IND
    "dolts": "IND",
    # JAX
    "jags": "JAX",
    "jagoffs": "JAX",
    "sacksonville": "JAX",
    "duval": "JAX",
    # KC
    "chefs": "KC",
    "qweefs": "KC",
    "kingdom": "KC",
    "chiefskingdom": "KC",
    # LV
    "faiders": "LV",
    "raidernation": "LV",
    "silverandblack": "LV",
    # LAC
    "bolts": "LAC",
    "chargas": "LAC",
    # MIA
    "fins": "MIA",
    "phins": "MIA",
    "tunas": "MIA",
    # MIN
    "vikes": "MIN",
    "skol": "MIN",
    "purplepeople": "MIN",
    "minny": "MIN",
    # NE
    "pats": "NE",
    "cheatriots": "NE",
    # NO
    "whodat": "NO",
    "aints": "NO",
    "geaux": "NO",
    # NYG
    "gmen": "NYG",
    "bigblue": "NYG",
    "jints": "NYG",
    # NYJ
    "ganggreen": "NYJ",
    "sackexchange": "NYJ",
    # PHI
    "iggles": "PHI",
    "philly": "PHI",
    "gobirds": "PHI",
    "tushpush": "PHI",
    # PIT
    "stillers": "PIT",
    "yinz": "PIT",
    "yinzers": "PIT",
    "blitzburgh": "PIT",
    "sixburgh": "PIT",
    # SF
    "niners": "SF",
    "9ers": "SF",
    "fortyniners": "SF",
    "faithful": "SF",
    "frisco": "SF",
    # SEA
    "hawks": "SEA",
    "12s": "SEA",
    "twelves": "SEA",
    "legionofboom": "SEA",
    # TB
    "bucs": "TB",
    "pewter": "TB",
    "tompa": "TB",
    # TEN
    "tits": "TEN",
    "flamingthumbtack": "TEN",
    "twotoneblue": "TEN",
    # WSH
    "commies": "WSH",
    "footballteam": "WSH",
}


def _normalize_team(value: object, known_team_tokens: set[str]) -> str | None:
    """Resolve ``value`` to a real 32-team token, or ``None`` if it is not real.

    Pure and case/whitespace insensitive: coerces the input to an upper-cased,
    stripped token and returns it when it is a member of ``known_team_tokens`` (which
    the caller supplies with BOTH abbreviations and display-name tokens, all real
    teams). This real-token match is checked FIRST and ALWAYS wins — the alias path
    never shadows it.

    Only on a real-set MISS does a pure FALLBACK consult the curated
    :data:`_TEAM_ALIASES` slang map (case-insensitive lowercase key), returning the
    mapped canonical abbreviation ONLY IF that abbreviation is ITSELF a member of the
    real set (the defensive real-set guard — never emit a non-real team on a map typo
    or an unseeded / partial DB). Anything that is neither a real token nor a curated,
    real-targeting alias — a made-up team, ambiguous excluded slang, a non-string,
    blank — returns ``None``, which the validator turns into :attr:`QaIntent.unknown`
    for a team-bearing intent. Stays pure, synchronous and DB-free (the alias map is a
    static module constant).
    """
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    real = {t.strip().upper() for t in known_team_tokens}
    token = stripped.upper()
    if token in real:
        return token
    # Pure fallback: a curated, unambiguous slang nickname resolves to its canonical
    # abbreviation, but ONLY when that abbreviation is a real token (never emit a fake).
    # The lookup key is normalized to lowercase alphanumerics so punctuation/spacing
    # variants ("big blue", "gang-green", "'boys") all hit the same alias entry.
    alias_key = "".join(ch for ch in stripped.lower() if ch.isalnum())
    alias = _TEAM_ALIASES.get(alias_key)
    if alias is not None and alias in real:
        return alias
    return None


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

# Prediction leads phrase through a DIFFERENT prompt: the game-prediction intent has no
# asker-pick concept at all, so it must NOT inherit QA_ROLE/QA_GUARD's pick-status framing
# (which primes the model to narrate "you have/haven't picked …" onto an asker who made no
# pick — observed twice in live testing). This role frames the bot as an analyst calling a
# game and never mentions picks/accounts, so there is nothing to misattribute; the actual
# call + all numbers live in the deterministic _ListAnswer body, so this prompt only ever
# re-voices a pick-free one-line intro.
PREDICTION_ROLE = (
    "You are the league's wise-cracking NFL analyst, delivering YOUR OWN prediction for a "
    "game a member asked you to call. The member is only asking for your read — they have "
    "not made any pick, and picks are not part of this at all."
)
PREDICTION_GUARD = (
    "Re-voice the supplied intro in character — it sets up your prediction of a specific "
    "game. You may needle the teams or the matchup, but do NOT declare who wins, loses, or "
    "covers, and do not name your pick — that verdict lands verbatim on the very next line, "
    "so your intro only teases that your call is coming. Say NOTHING about the member's "
    "picks, choices, account, or pick'em status (there are none here), and add NO stat, "
    "spread, or number that is not in the intro. Keep it to one or two short lines with at "
    "most one emoji."
)

# Deterministic short-circuit line for an unregistered asker (no LLM call needed).
_REGISTER_LINE = "You need a pick'em account first — run /register to get set up."

# Deterministic error line — the best-effort fallback when a db seam raises. Never
# leaks anything; just keeps the listener from ever raising into the gateway loop.
_ERROR_LINE = "Something went sideways pulling that up — give it another shot in a bit."

# Tier-2 (coming_soon) wink: recognized-but-planned, NO capability menu, NO DB read.
# Injuries + weather + who-wins predictions are now LIVE intents, so they are dropped
# from the wink text to keep it honest (only line movement remains unsupported).
_COMING_SOON_FACT = (
    "That's not something tracked yet — live line movement is on the roadmap, "
    "not in the playbook today."
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

# Stateless soft-decline for a teamless injuries question — the report is always
# team-scoped, so a whole-league dump is never the answer. No HTTP, no DB.
# NOTE: deliberately worded WITHOUT the word "injury". The small local phrasing
# model carries a hard prior that any terse injury-flavored decline means "not yet
# supported" (injuries used to be a coming_soon topic) and inverts the meaning when
# it phrases it — e.g. "Name a team and I'll pull that team's injury report" came
# out as "That feature is not yet supported. 🙄". Framing it as a neutral
# "you didn't name a team, ask again" phrases faithfully; the injury context is
# obvious from the member's own question. The longer degrade line below survives
# phrasing (it reads as an unambiguous transient "couldn't do X right now").
_INJURIES_NO_TEAM_FACT = "No team in that question — name one and ask again."

# Best-effort degrade line when the team's game/event can't be resolved OR the ESPN
# fetch/parse fails. NEVER an invented injury (T-u0z-02) — a fixed, honest miss.
_INJURIES_DEGRADE_FACT = (
    "Couldn't pull the injury report right now — give it another shot in a bit."
)

# Stateless soft-decline for a teamless weather question — the forecast is always
# game-scoped (a specific stadium), so there is no whole-league answer. No HTTP, no DB.
# REUSES the proven-neutral injuries wording VERBATIM (PHRASING-INVERSION lesson): the
# small local phrasing model inverts terse topic-flavored declines, so a neutral
# "you didn't name a team, ask again" line phrases faithfully where a weather-flavored
# terse line would flip to "not supported". The weather context is obvious from the
# member's own question.
_WEATHER_NO_TEAM_FACT = "No team in that question — name one and ask again."

# Best-effort degrade line when the team's game can't be resolved, the stadium is
# missing from the table, OR the Open-Meteo fetch/parse fails. NEVER an invented
# forecast (T-29v-01) — an unambiguous transient miss that survives phrasing.
_WEATHER_DEGRADE_FACT = "Couldn't pull the forecast right now — give it another shot in a bit."

# Best-effort degrade line when the ESPN news fetch/parse fails OR a named team can't
# be resolved. NEVER an invented/paraphrased headline (T-ikf-01) — a fixed, honest miss.
# REUSES the exact locked wording proven to survive the local phrasing model (the
# PHRASING-INVERSION lesson from injuries PR #107 / weather PR #108): a terse
# news-flavored decline could be flipped to "not supported", so this reads as an
# unambiguous transient "couldn't do X right now" sentence.
_NEWS_DEGRADE_FACT = "Couldn't pull ESPN news right now — give it another shot in a bit."

# Concrete empty line for a teamless (league) ask that returns no articles — a
# transient, unambiguous "nothing right now", never an invented headline.
_NEWS_EMPTY_LEAGUE_FACT = "No fresh NFL headlines from ESPN right now — check back in a bit."

# Top N headlines shown for a news answer.
NEWS_DISPLAY_LIMIT = 5

# --- prediction intent (260710-mpw) --------------------------------------- #
# A DERIVED-FACTS intent: code owns the pick + ALL cover arithmetic; the LLM only
# voices the short lead. Every constant below is a CONCRETE, game-anchored sentence
# (never a terse topic-flavored fragment) so the phrasing model can't invert it —
# the degrade notes live in the _ListAnswer BODY (never phrased), and the plain-string
# facts are phrased, so both must read as unambiguous statements (qa-phrasing-inversion).

# Teamless prediction soft-decline — REUSES the proven-neutral injuries/weather wording
# VERBATIM (a terse "who wins?" decline would flip to "not supported"; this neutral
# "you didn't name a team" line phrases faithfully).
_PREDICTION_NO_TEAM_FACT = "No team in that question — name one and ask again."

# The team's current-week game couldn't be pinned down (unknown / bye / ambiguous).
# A concrete transient miss — never an invented pick.
_PREDICTION_UNRESOLVED_FACT = (
    "I couldn't pin down that team's game this week — double-check the team name and ask again."
)

# Neither the live market nor the frozen sheet has a spread posted yet, so there is no
# cover to call. Never an invented line/pick.
_PREDICTION_NO_LINE_FACT = (
    "There's no line posted on that game yet, so I can't call a cover for you — "
    "check back once the spread is up."
)

# Per-factor degrade notes — each a full, game-anchored transient sentence that reads
# as "couldn't check X this time", NOT a terse fragment. These sit in the verbatim body.
_PREDICTION_INJURIES_DEGRADE_NOTE = (
    "Couldn't pull the injury report this time, so this read leaves injuries out of it."
)
_PREDICTION_WEATHER_DEGRADE_NOTE = (
    "Couldn't pull the game-time forecast this time, so this read leaves weather out of it."
)
# Live-line-missing relabel: fall back to the frozen pick'em spread, clearly relabelled.
_PREDICTION_FROZEN_FALLBACK_NOTE = "Working off the line we've got locked here — couldn't reach the live market for a fresh number."

# A material live-vs-frozen divergence (favorite flip OR magnitude delta >= this) fires
# the conflict callout.
_PREDICTION_CONFLICT_THRESHOLD = Decimal("1.0")

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


@dataclass(frozen=True)
class _ListAnswer:
    """A multi-item answer (whole slate / whole-week scores).

    A one-line phrasing guard would make the LLM DROP a long list, so list answers
    are split: ``header_fact`` is the in-character lead and ``body`` — the deterministic
    per-game block — is appended verbatim by the orchestrator so the games/scores can
    never be summarized away.

    ``phrase_header`` controls whether the header is run through the LLM voice. The
    default (``True``) phrases it (slate/scores/injuries/weather leads). News sets it
    ``False``: the wrapper is a FIXED deterministic line (the design's "personality
    lives only in a fixed wrapper line"), because the small phrasing model INVERTS a
    terse news wrapper — "Latest on KC (ESPN …):" came out as "The data for KC is not
    yet supported 🙄" / "I have no facts to report." (the same inversion class as the
    injuries no-team line PR #107 and the weather dome line PR #108). Not phrasing it
    also makes the verbatim-relay guarantee absolute — the headlines never reach the LLM.
    """

    header_fact: str
    body: str
    phrase_header: bool = True


def _fmt_when(when: object) -> str | None:
    """Format a tz-aware close time as a short, human string, else ``None``.

    Turns ``2026-07-06 12:22:31.079408+00:00`` into ``Mon Jul 6, 12:22 PM UTC`` —
    no microseconds, no offset noise. Built without ``strftime`` day/hour padding
    quirks so it renders identically on any libc.
    """
    if not isinstance(when, datetime):
        return None
    hour = when.hour % 12 or 12
    ampm = "AM" if when.hour < 12 else "PM"
    return f"{when.strftime('%a %b')} {when.day}, {hour}:{when.minute:02d} {ampm} UTC"


def _close_clause(when: str | None, pick_open: bool) -> str | None:
    """The 'Picks close/closed <when>' clause — tense-correct, or ``None``."""
    if not when:
        return None
    return f"Picks {'close' if pick_open else 'closed'} {when}."


def _pick_status_fact(status: dict) -> str:
    """Build the asker's OWN pick-status fact (registered case).

    Window-aware. While the window is OPEN the fact is an actionable to-do (which
    slots still need a pick). Once the window has CLOSED the card can no longer be
    filled, so enumerating the unmade slots isn't actionable — the fact is just the
    verdict (locked in full vs. locked but incomplete), kept short so the one-line
    phrasing guard can't trim away the meaning.
    """
    name = status.get("display_name")
    complete = bool(status.get("complete"))
    if not status.get("pick_open"):
        if complete:
            return f"{name} was locked in for the week — a full card before the deadline."
        return f"Picks are locked for the week and {name}'s card was incomplete."
    if complete:
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


def _spread_clause(favorite: str, underdog: str | None, spread: str, asked_team: str | None) -> str:
    """The spread sentence, framed from the asked team's side when one was asked.

    When the user asked about a specific team (single-game path, ``asked_team`` set),
    phrase the line relative to THAT team and name the opponent — "FAV favored by N
    vs. DOG" or "DOG are N-point underdogs vs. FAV". Otherwise (teamless / whole-slate,
    or the opponent abbr is missing) keep the neutral favorite-anchored phrasing.
    Deterministic FACT only — invents nothing beyond the two team abbrs + the spread.
    """
    if underdog:
        if asked_team == favorite:
            return f"{favorite} favored by {spread} vs. {underdog}."
        if asked_team == underdog:
            return f"{underdog} are {spread}-point underdogs vs. {favorite}."
    return f"{favorite} favored by {spread}."


def _one_game_line_fact(
    game: dict, week: int | None, when: str | None, pick_open: bool, asked_team: str | None = None
) -> str:
    """A single game's line fact (matchup + favorite/spread + total + close).

    ``asked_team`` (the canonical abbr of the team the user asked about, if any) frames
    the spread from that team's perspective — see :func:`_spread_clause`.
    """
    away = game.get("away")
    home = game.get("home")
    parts = [f"Week {week}: {away} at {home}."]
    favorite = game.get("favorite")
    spread = game.get("spread")
    if favorite and spread:
        parts.append(_spread_clause(favorite, game.get("underdog"), spread, asked_team))
    total = game.get("total")
    if total:
        parts.append(f"Total is {total}.")
    close_clause = _close_clause(when, pick_open)
    if close_clause:
        parts.append(close_clause)
    return " ".join(parts)


def _slate_fact(slate: dict) -> str | _ListAnswer:
    """Build the lines/slate answer.

    A single game (or a team-narrowed slate) stays a one-liner. A whole multi-game
    slate becomes a :class:`_ListAnswer`: a short in-character header + a
    deterministic one-line-per-game block, so the full slate always lands.
    """
    games = slate.get("games") or []
    week = slate.get("week")
    when = _fmt_when(slate.get("close_at"))
    pick_open = bool(slate.get("pick_open"))
    asked_team = slate.get("asked_team")
    if not games:
        return f"No games are posted for week {week} yet."
    if len(games) == 1:
        return _one_game_line_fact(games[0], week, when, pick_open, asked_team)

    body_lines = []
    for game in games:
        line = f"{game.get('away')} @ {game.get('home')}"
        favorite = game.get("favorite")
        spread = game.get("spread")
        if favorite and spread:
            line += f" — {favorite} -{spread}"
        total = game.get("total")
        if total:
            line += f" (O/U {total})"
        body_lines.append(line)
    header = f"Week {week}'s full slate — {len(games)} games."
    close_clause = _close_clause(when, pick_open)
    if close_clause:
        header += f" {close_clause}"
    return _ListAnswer(header_fact=header, body="\n".join(body_lines))


def _scores_one_line(game: dict, week: int | None) -> str:
    """A single game's score fact (display-only)."""
    tag = "final" if game.get("status") == "FINAL" else "in progress"
    return (
        f"Week {week}: {game.get('away')} {game.get('away_score')} at "
        f"{game.get('home')} {game.get('home_score')} ({tag})."
    )


def _scores_fact(scores: dict) -> str | _ListAnswer:
    """Build the scores answer (final + in-progress) for the week.

    One scored game is a one-liner; a full scoreboard becomes a :class:`_ListAnswer`
    (a short header + a deterministic per-game score block) so no score is dropped.
    """
    games = scores.get("games") or []
    week = scores.get("week")
    if not games:
        return f"No scores yet for week {week}."
    if len(games) == 1:
        return _scores_one_line(games[0], week)

    n_final = sum(1 for game in games if game.get("status") == "FINAL")
    n_live = len(games) - n_final
    body_lines = []
    for game in games:
        tag = "final" if game.get("status") == "FINAL" else "in progress"
        body_lines.append(
            f"{game.get('away')} {game.get('away_score')} @ "
            f"{game.get('home')} {game.get('home_score')} ({tag})"
        )
    header = f"Week {week} scoreboard — {n_final} final, {n_live} in progress."
    return _ListAnswer(header_fact=header, body="\n".join(body_lines))


def _injuries_as_of(players: list[dict]) -> str | None:
    """The first available per-player ``date`` stamp — the report's as-of, or ``None``."""
    for player in players:
        as_of = player.get("date")
        if as_of:
            return as_of
    return None


def _injury_player_line(player: dict) -> str:
    """One deterministic per-player injury line, built ONLY from parsed fields.

    Invents nothing: a missing field is simply omitted (name/position/status/body
    part/return date each degrade out of the line rather than being fabricated).
    """
    name = player.get("display_name") or "Unknown player"
    position = player.get("position")
    subject = f"{name} ({position})" if position else name
    status = player.get("status") or "status unknown"
    body_part = player.get("body_part")
    detail = f"{status} — {body_part}" if body_part else status
    line = f"{subject}: {detail}"
    return_date = player.get("return_date")
    if return_date:
        line += f", expected back {return_date}"
    return line


def _injuries_fact(team_abbr: str, players: list[dict]) -> str | _ListAnswer:
    """Build the deterministic injuries answer for ``team_abbr``.

    * No injuries listed -> a single clean line (no fabricated as-of when there is
      no injury to stamp).
    * One player -> a one-liner naming status / name / position / body part / return
      date + the as-of stamp.
    * Multiple players -> a :class:`_ListAnswer` whose ``header_fact`` is a short
      phrasable lead and whose ``body`` is a deterministic one-line-per-player block,
      so the one-line phrasing guard can NEVER trim the roster away.

    Everything is derived from the parsed ``players`` — nothing beyond those fields
    is invented (T-u0z-02).
    """
    if not players:
        return f"No injuries listed for {team_abbr} right now."

    as_of = _injuries_as_of(players)
    as_of_clause = f" (as of {as_of})" if as_of else ""

    if len(players) == 1:
        return f"{team_abbr} injury report — {_injury_player_line(players[0])}.{as_of_clause}"

    header = f"{team_abbr} injury report — {len(players)} listed{as_of_clause}."
    body = "\n".join(_injury_player_line(player) for player in players)
    return _ListAnswer(header_fact=header, body=body)


def _indoor_fact(stadium: Stadium) -> str:
    """The deterministic dome/indoor line — weather is a non-factor, NO fetch needed.

    NOTE: worded as a concrete game-anchored statement ("This game is at {name}, a
    covered dome … the forecast does not apply") rather than a terse "is indoors —
    weather's a non-factor." The small local phrasing model INVERTS the terse form —
    it rewrote "{name} is indoors — weather's a non-factor." into "I don't have any
    data on stadium roofs 🙄", denying the deterministic fact instead of relaying it
    (same class of inversion as the injuries no-team line, PR #107). The concrete
    "This game is at … a covered dome … the forecast does not apply." form phrases
    faithfully (verified live).
    """
    return (
        f"This game is at {stadium.name}, a covered dome — indoor conditions at "
        f"kickoff, so the forecast does not apply."
    )


def _weather_fact(home_abbr: str, stadium: Stadium, forecast: dict) -> str:
    """Build the deterministic kickoff-time weather fact, ONLY from parsed fields.

    Invents nothing: a missing metric is simply OMITTED from the line rather than
    fabricated (T-29v-01). Anchored by the matched forecast hour so the reader can see
    the line is a kickoff-hour reading, not an invented current condition. A 0 (or
    absent) precip reads as "no precip expected".
    """
    parts: list[str] = []
    temp = forecast.get("temperature_f")
    if temp is not None:
        parts.append(f"{temp}°F")
    wind = forecast.get("wind_mph")
    if wind is not None:
        parts.append(f"wind {wind} mph")
    precip = forecast.get("precip_in")
    if precip is not None and precip > 0:
        parts.append(f"{precip} in precip")
    else:
        parts.append("no precip expected")

    hour = forecast.get("hour")
    anchor = f" ({hour} GMT)" if hour else ""
    return f"{stadium.name} at kickoff{anchor}: {', '.join(parts)}."


def _news_as_of(articles: list[dict]) -> str | None:
    """The first non-empty ``published`` stamp — the block's as-of, or ``None``."""
    for article in articles:
        as_of = article.get("published")
        if as_of:
            return as_of
    return None


def _news_headline_line(article: dict) -> str:
    """One deterministic headline line — the VERBATIM headline linked to its source.

    The raw headline string MUST survive unchanged (no truncation, no re-casing): this
    line is appended via the :class:`_ListAnswer` body and is NEVER handed to the LLM,
    so the headline can never be summarized or reinvented (T-ikf-01).

    When a source link is present, the headline is rendered as a Discord masked link
    ``[headline](url)`` so it is clickable (bot messages render masked links) while the
    headline text stays verbatim inside the brackets. A headline containing ``[`` or
    ``]`` would break the mask, so those fall back to the headline + a bare ``<url>``
    (angle brackets suppress the auto-embed). No link -> the bare headline.
    """
    headline = article["headline"]
    link = article.get("link")
    if not link:
        return headline
    if "[" in headline or "]" in headline:
        return f"{headline} — <{link}>"
    return f"[{headline}]({link})"


def _clean_subject(subject: str) -> str:
    """A short, single-line rendering of the classifier ``subject`` for the wrapper.

    The wrapper is deterministic (never phrased), so this user-derived text is inserted
    verbatim — collapse whitespace and cap the length so a noisy subject can't blow up
    the line. Not a security boundary (the subject already passed the classifier); this
    is purely cosmetic tidying.
    """
    collapsed = " ".join(subject.split())
    return collapsed[:60].strip()


def _news_fact(
    team_abbr: str | None,
    articles: list[dict],
    *,
    subject: str | None = None,
    subject_missed: bool = False,
) -> str | _ListAnswer:
    """Build the deterministic news answer — verbatim headlines under a FIXED wrapper.

    * No articles -> a concrete transient empty line (team-scoped or the league line);
      never an invented headline.
    * Otherwise ALWAYS a :class:`_ListAnswer` (even for a single headline) so the
      headline block is NEVER phrased: ``header_fact`` is the deterministic wrapper and
      ``body`` is the VERBATIM headline block (one line per article).
    * ``subject`` narrows the wrapper: when given and matched, the wrapper names it
      ("Latest on KC — Patrick Mahomes …:"); ``subject_missed`` (a subject was asked but
      no article matched, so we fell back to the general feed) says so honestly
      ("No Patrick Mahomes-specific headlines — latest on KC …:").
    """
    if not articles:
        if team_abbr is not None:
            return f"No fresh ESPN headlines on {team_abbr} right now — check back in a bit."
        return _NEWS_EMPTY_LEAGUE_FACT

    as_of = _news_as_of(articles)
    as_of_clause = f", as of {as_of}" if as_of else ""
    scope = team_abbr if team_abbr is not None else "the NFL"
    subj = _clean_subject(subject) if subject else ""
    if subj and subject_missed:
        header = f"No {subj}-specific headlines — latest on {scope} (ESPN{as_of_clause}):"
    elif subj:
        header = f"Latest on {scope} — {subj} (ESPN{as_of_clause}):"
    elif team_abbr is not None:
        header = f"Latest on {team_abbr} (ESPN{as_of_clause}):"
    else:
        header = f"Latest NFL headlines (ESPN{as_of_clause}):"
    body = "\n".join(_news_headline_line(article) for article in articles)
    # phrase_header=False: the wrapper is a FIXED deterministic line. The small phrasing
    # model inverts a terse news wrapper into a "not supported / no facts" decline, and
    # keeping it out of the LLM makes the verbatim-relay guarantee absolute.
    return _ListAnswer(header_fact=header, body=body, phrase_header=False)


def _to_decimal(value: object) -> Decimal | None:
    """Coerce a stringified/number spread to a positive-or-any ``Decimal``, else ``None``.

    Pure: the frozen spread arrives as a stringified magnitude (like ``"3.0"``); the live
    spread as a float. Any unusable value degrades to ``None`` (the caller then treats the
    line as absent) — never raises.
    """
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation, ValueError, TypeError:
        return None


def _fmt_num(value: Decimal) -> str:
    """Render a spread magnitude without trailing zeros (``3.0`` -> ``3``, ``3.5`` -> ``3.5``).

    Uses fixed-point formatting (never scientific) so a whole number stays readable in the
    cover read.
    """
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _prediction_injury_note(injuries: list[dict] | None) -> str:
    """A concrete injury note for the briefing body, derived ONLY from parsed fields.

    * ``None`` (couldn't fetch/parse) -> the fixed degrade note (never invents "healthy").
    * ``[]`` (report present, nobody listed) -> a clean "nobody flagged" line.
    * otherwise names up to three players (with status when known) + a "+N more" tail.
    """
    if injuries is None:
        return _PREDICTION_INJURIES_DEGRADE_NOTE
    if not injuries:
        return "Injury report is clean on that side right now — nobody flagged."
    named: list[str] = []
    for player in injuries[:3]:
        name = player.get("display_name") or "an unnamed player"
        status = player.get("status")
        named.append(f"{name} ({status})" if status else name)
    extra = "" if len(injuries) <= 3 else f", plus {len(injuries) - 3} more"
    return f"Injury watch: {', '.join(named)}{extra}."


def _prediction_fact(
    inputs: dict,
    *,
    live_odds: ScoreboardOdds | None,
    injuries: list[dict] | None,
    weather_note: str | None,
) -> str | _ListAnswer:
    """Build the DERIVED-FACTS prediction briefing — code owns the pick + ALL math.

    Pure and network-free. The pick is the EFFECTIVE line's favorite and every number
    (cover read, record/ATS, conflict delta) is computed HERE, never by the LLM. Returns:

    * a plain string for the no-line case (no pick can be called), or
    * a :class:`_ListAnswer` whose ``header_fact`` is a short in-voice lead naming the
      pick (phrased) and whose ``body`` is the deterministic VERBATIM factor block (cover
      read, record/ATS, injury note, weather note, conflict callout). The list-answer
      shape is what keeps the arithmetic out of the LLM's hands — the one-line ``QA_GUARD``
      phrases ONLY the lead; the body reaches Discord byte-for-byte (T-mpw-02).

    The EFFECTIVE line prefers the live market (labelled "current market"); when the live
    line is absent it falls back to the FROZEN spread, relabelled. When neither carries a
    usable spread it returns the no-line line — never an invented pick.
    """
    asked_team = inputs.get("asked_team")
    home = inputs.get("home")
    away = inputs.get("away")
    frozen_fav = inputs.get("favorite")
    frozen_dog = inputs.get("underdog")
    frozen_spread = _to_decimal(inputs.get("spread"))
    record = inputs.get("record") or "0-0"
    ats = inputs.get("ats") or "0-0"

    # Map the live signed home-relative spread back to favorite/underdog abbrs.
    live_fav: str | None = None
    live_dog: str | None = None
    live_mag: Decimal | None = None
    if live_odds is not None and live_odds.spread is not None and home and away:
        live_mag = Decimal(str(abs(live_odds.spread)))
        if live_odds.spread < 0:
            live_fav, live_dog = home, away
        elif live_odds.spread > 0:
            live_fav, live_dog = away, home
        # spread == 0 is a true pick'em — no favorite from the live line.

    using_live = live_fav is not None and live_mag is not None and live_mag > 0
    if using_live:
        assert live_fav is not None and live_mag is not None
        eff_fav, eff_dog, eff_mag = live_fav, live_dog, live_mag
    elif frozen_fav is not None and frozen_spread is not None and frozen_spread > 0:
        eff_fav, eff_dog, eff_mag = frozen_fav, frozen_dog, frozen_spread
    else:
        return _PREDICTION_NO_LINE_FACT

    dog_label = eff_dog or "the other side"
    lines: list[str] = []
    if using_live:
        lines.append(
            f"**My call: {eff_fav} to cover — {eff_fav} -{_fmt_num(eff_mag)} (current market line).**"
        )
        lines.append(
            f"{eff_fav} has to win by more than {_fmt_num(eff_mag)} for it to cash; "
            f"{dog_label} covers otherwise."
        )
    else:
        lines.append(f"**My call: {eff_fav} to cover — {eff_fav} -{_fmt_num(eff_mag)}.**")
        lines.append(
            f"{eff_fav} has to win by more than {_fmt_num(eff_mag)} for it to cash; "
            f"{dog_label} covers otherwise. {_PREDICTION_FROZEN_FALLBACK_NOTE}"
        )

    # Conflict callout — only when the live line is in play AND materially differs from the
    # frozen sheet (favorite flip OR magnitude delta >= threshold).
    if using_live and frozen_fav is not None and frozen_spread is not None and frozen_spread > 0:
        assert live_mag is not None
        favorite_flip = live_fav != frozen_fav
        mag_delta = abs(live_mag - frozen_spread)
        if favorite_flip or mag_delta >= _PREDICTION_CONFLICT_THRESHOLD:
            lines.append(
                f"Heads up: you locked this at {frozen_fav} -{_fmt_num(frozen_spread)}, "
                f"but the current market has {eff_fav} -{_fmt_num(eff_mag)}."
            )

    lines.append(
        f"{asked_team or 'They'} are {record} straight up and {ats} against the spread this season."
    )
    lines.append(_prediction_injury_note(injuries))
    lines.append(weather_note if weather_note is not None else _PREDICTION_WEATHER_DEGRADE_NOTE)

    # The phrased lead is a pick-FREE flavor intro: the pick lives only in the bold,
    # verbatim body line above, so the LLM can never misattribute it to the asker as a
    # pick'em selection (a game read is the bot's own forecast, not the asker's pick).
    header = f"Here's my read on the {asked_team or 'that'} game."
    return _ListAnswer(header_fact=header, body="\n".join(lines))


async def _build_fact(result: QaResult, *, discord_id: int) -> str | _ListAnswer | None:
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

    if result.intent is QaIntent.injuries:
        # Team-scoped by construction: a teamless injuries question stateless-soft-
        # declines (no HTTP, no DB lookup) — never a whole-league dump.
        if result.team is None:
            return _INJURIES_NO_TEAM_FACT
        # Resolve the asked team's current-week game -> stored espn_event_id (+ the
        # canonical ESPN abbreviation to filter the parse by). None -> degrade.
        resolved = await db_bridge.get_injuries_event_id_async(result.team)
        if resolved is None:
            return _INJURIES_DEGRADE_FACT
        event_id, canonical_abbr = resolved
        # espn_extra owns ALL HTTP + Redis; qa.py imports the seam, never httpx.
        from app.services import espn_extra

        payload = await espn_extra.fetch_injuries(event_id)
        if payload is None:
            return _INJURIES_DEGRADE_FACT  # best-effort fetch failed — never invent
        players = espn_extra.parse_injuries(payload, canonical_abbr)
        if players is None:
            return _INJURIES_DEGRADE_FACT  # unusable shape — never invent
        return _injuries_fact(canonical_abbr, players)

    if result.intent is QaIntent.weather:
        # Team-scoped by construction: a teamless weather question stateless-soft-
        # declines (no HTTP, no DB lookup) — the forecast is always game-scoped.
        if result.team is None:
            return _WEATHER_NO_TEAM_FACT
        # Resolve the asked team's current-week game -> HOME abbr + kickoff. None ->
        # degrade (unresolvable game — never invent a forecast).
        resolved = await db_bridge.get_weather_target_async(result.team)
        if resolved is None:
            return _WEATHER_DEGRADE_FACT
        home_abbr, kickoff_at = resolved
        # weather owns the stadium table + ALL HTTP + Redis; qa.py imports the seam.
        from app.services import weather

        stadium = weather.lookup_stadium(home_abbr)
        if stadium is None:
            return _WEATHER_DEGRADE_FACT  # no table row — never invent
        if stadium.indoor:
            return _indoor_fact(stadium)  # dome short-circuit — NO fetch
        payload = await weather.fetch_forecast(stadium.lat, stadium.lon)
        if payload is None:
            return _WEATHER_DEGRADE_FACT  # best-effort fetch failed — never invent
        forecast = weather.parse_forecast(payload, kickoff_at)
        if forecast is None:
            return _WEATHER_DEGRADE_FACT  # unusable / hour absent — never invent
        return _weather_fact(home_abbr, stadium, forecast)

    if result.intent is QaIntent.news:
        # Team is OPTIONAL: a named team filters the league page client-side; a teamless
        # ask returns the top LEAGUE headlines (a null team is a VALID answer here, NOT
        # a soft-decline). Headlines are relayed VERBATIM — never sent to the LLM.
        team_filter: tuple[str, str] | None = None
        team_abbr: str | None = None
        if result.team is not None:
            resolved = await db_bridge.get_news_team_filter_async(result.team)
            if resolved is None:
                return _NEWS_DEGRADE_FACT  # un-resolvable team — never invent
            team_filter = (resolved[0].upper(), resolved[1].upper())
            team_abbr = resolved[0]
        # espn_extra owns ALL HTTP + Redis; qa.py imports the seam, never httpx.
        from app.services import espn_extra

        payload = await espn_extra.fetch_news()
        if payload is None:
            return _NEWS_DEGRADE_FACT  # best-effort fetch failed — never invent
        # Parse the full candidate pool (team-filtered) so a subject filter has something
        # to narrow; the display cap is applied AFTER narrowing.
        pool = espn_extra.parse_news(
            payload, team_filter=team_filter, limit=espn_extra.NEWS_FETCH_LIMIT
        )
        if pool is None:
            return _NEWS_DEGRADE_FACT  # unusable shape — never invent
        # Subject/player narrowing: "any news about Patrick Mahomes?" keeps only articles
        # about that subject. None -> no meaningful subject (keep the whole feed); [] -> a
        # subject was asked but nothing matched -> FALL BACK to the feed with an honest note
        # (never an empty/invented answer).
        narrowed = espn_extra.filter_news_by_subject(pool, result.subject)
        if narrowed is None:
            return _news_fact(team_abbr, pool[:NEWS_DISPLAY_LIMIT])
        if narrowed:
            return _news_fact(team_abbr, narrowed[:NEWS_DISPLAY_LIMIT], subject=result.subject)
        return _news_fact(
            team_abbr, pool[:NEWS_DISPLAY_LIMIT], subject=result.subject, subject_missed=True
        )

    if result.intent is QaIntent.prediction:
        # DERIVED-FACTS: code computes the pick + ALL cover math; the LLM only voices the
        # lead. Team-required by construction — a teamless prediction soft-declines.
        if result.team is None:
            return _PREDICTION_NO_TEAM_FACT
        inputs = await db_bridge.get_prediction_inputs_async(result.team)
        if inputs is None:
            return _PREDICTION_UNRESOLVED_FACT  # no single game this week — never invent

        # The independent live factors run CONCURRENTLY; each degrades on its own without
        # aborting the briefing (degrade-never-bail). qa.py imports the seams; it never
        # touches httpx / redis itself.
        import asyncio

        from app.services import espn_extra, live_odds, weather

        season = inputs.get("season")
        week = inputs.get("week")
        event_id = inputs.get("espn_event_id")
        home_abbr = inputs.get("home")
        asked_abbr = inputs.get("asked_team")
        kickoff_at = inputs.get("kickoff_at")

        # Resolve the home stadium synchronously so a DOME short-circuits the forecast
        # fetch (weather is a non-factor indoors — reuse the existing indoor line).
        stadium = weather.lookup_stadium(home_abbr) if home_abbr else None
        fetch_weather = stadium is not None and not stadium.indoor

        async def _none_result() -> None:
            return None

        odds_task = (
            live_odds.fetch_live_odds(season, week, event_id)
            if event_id is not None and season is not None and week is not None
            else _none_result()
        )
        injuries_task = (
            espn_extra.fetch_injuries(event_id) if event_id is not None else _none_result()
        )
        weather_task = (
            weather.fetch_forecast(stadium.lat, stadium.lon)
            if fetch_weather and stadium is not None
            else _none_result()
        )

        odds_res, injuries_res, weather_res = await asyncio.gather(
            odds_task, injuries_task, weather_task, return_exceptions=True
        )

        # Live line: fetch_live_odds already returns the target event's ScoreboardOdds or
        # None; any raised exception degrades to None (fall back to the frozen spread).
        live = odds_res if not isinstance(odds_res, BaseException) else None

        # Injuries: parse the asked team's block defensively; any failure -> None note.
        injuries: list[dict] | None = None
        injuries_payload = injuries_res if not isinstance(injuries_res, BaseException) else None
        if injuries_payload is not None and asked_abbr:
            injuries = espn_extra.parse_injuries(injuries_payload, asked_abbr)

        # Weather: a dome is a concrete non-factor line (no fetch); an outdoor game uses the
        # matched-hour forecast; any miss leaves the note None -> the degrade note.
        weather_note: str | None = None
        if stadium is not None and stadium.indoor:
            weather_note = _indoor_fact(stadium)
        elif fetch_weather and stadium is not None:
            weather_payload = weather_res if not isinstance(weather_res, BaseException) else None
            if weather_payload is not None and kickoff_at is not None:
                forecast = weather.parse_forecast(weather_payload, kickoff_at)
                if forecast is not None and home_abbr is not None:
                    weather_note = _weather_fact(home_abbr, stadium, forecast)

        return _prediction_fact(
            inputs, live_odds=live, injuries=injuries, weather_note=weather_note
        )

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
        # A prediction lead uses the analyst role (no pick-status framing); every other
        # intent uses the facts-first QA role/guard.
        if result.intent is QaIntent.prediction:
            system_prompt = compose_prompt(voice, PREDICTION_ROLE, PREDICTION_GUARD)
        else:
            system_prompt = compose_prompt(voice, QA_ROLE, QA_GUARD)

        if isinstance(fact, _ListAnswer):
            # List answer: phrase ONLY the short header in voice, then append the
            # deterministic block verbatim so the full slate/scoreboard always lands
            # (a one-line phrasing guard would otherwise summarize the list away). A
            # header with phrase_header=False (news) is a FIXED deterministic wrapper —
            # never sent to the LLM — so the verbatim relay is absolute.
            if fact.phrase_header:
                phrased_header = await llm_client.phrase(
                    fact.header_fact, system_prompt=system_prompt
                )
                header = phrased_header if phrased_header is not None else fact.header_fact
            else:
                header = fact.header_fact
            return f"{header}\n{fact.body}"

        phrased = await llm_client.phrase(fact, system_prompt=system_prompt)
        return phrased if phrased is not None else fact
    except Exception:
        # A classify / db / phrase hiccup must never escape into the gateway loop.
        logger.warning("answer_question_failed", exc_info=True)
        return _ERROR_LINE
