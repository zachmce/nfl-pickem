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
``None`` on any HTTP error. The registry ships with TWO tools — ``lookup_team_roster``
(issue #179) and ``lookup_player_season_stats`` (issue #183) — each grounding its
question in current ESPN data instead of the model's training cutoff; an EMPTY registry
stays a supported fallback branch.
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


async def _lookup_team_roster(team: str = "", position: str | None = None) -> object | None:
    """Look up ``team``'s CURRENT ESPN roster, narrowed to ``position`` when given.

    The ``espn_extra`` import is deferred to the call, mirroring ``qa.py``: espn_extra
    owns ALL HTTP + Redis and the brain modules import the seam, never httpx. ``team``
    defaults to empty so a model that forgets the argument degrades through the
    32-team allowlist to ``None`` rather than raising a TypeError into the loop.
    """
    from app.services import espn_extra

    payload = await espn_extra.fetch_team_roster(team)
    if payload is None:
        return None
    return espn_extra.parse_team_roster(payload, position=position)


# Concrete full sentences, never terse fragments — the phrasing hazard applies to what
# the model READS as much as to what it says (memory: qa-phrasing-inversion).
#
# MEASURED 2026-08-20, and the reason this text is shaped the way it is: a description
# that ONLY disclaimed the starter ("this tool does not know who starts") suppressed the
# call entirely — the model read "not relevant" and answered a starter question from its
# own stale memory instead, 5/5. So the description must do BOTH jobs, in this order:
# instruct the model to CALL for a starter question, then forbid it naming a starter from
# what comes back. The payload's own caveat is the second barrier (T-oym-05).
_ROSTER_TOOL_DESCRIPTION = (
    "Look up the players currently on one NFL team's roster this season. The team "
    "argument is that team's standard abbreviation, for example CHI for the Chicago "
    "Bears, LV for the Las Vegas Raiders, or KC for the Kansas City Chiefs. Pass a "
    "position abbreviation such as QB, WR or CB in the position argument to get the "
    "names of the players at that position; if you leave the position argument out you "
    "get only a count of how many players the team carries at each position, so ask "
    "again with a position when you need names. Call this tool for a question about who "
    "STARTS at a position as well, because the players it lists are the only players who "
    "could be starting, and your own memory of a team's quarterback is often a year or "
    "more out of date. This tool does not know who starts at any position, and it does "
    "not know any depth-chart order, because ESPN does not publish one. So when you are "
    "asked who starts, name the players this tool lists at that position and say plainly "
    "that the roster does not show which of them starts. Never call any player a starter "
    "on the strength of this tool. It reports each player's roster status, such as Active "
    "or Day-To-Day, but it carries no injury detail at all — no body part and no return "
    "date."
)


async def _lookup_player_season_stats(
    team: str = "", player: str = "", season: int | None = None
) -> object | None:
    """Look up ONE player's ESPN totals for ONE season.

    Two hops, both cached and both through the ``espn_extra`` seam (deferred import,
    same as :func:`_lookup_team_roster`): the roster payload resolves the name to an
    athlete id, then that id fetches the career table. The roster hop reuses the
    32-team allowlist as the ``team`` guard rather than rewriting it, and it reads the
    RAW payload because :func:`~app.services.espn_extra.parse_team_roster` drops the id
    on purpose (D-7). No id ever comes from the model. A roster MISS adds a third,
    also-cached hop through :func:`_resolve_off_roster`, which is how a player who
    changed teams still resolves.

    Every argument defaults so a model that forgets one degrades rather than raising a
    TypeError into the loop. A resolution miss returns a NOTE dict, never ``None``,
    because ``None`` becomes :data:`_NO_DATA_PAYLOAD`, which tells the model to answer
    from its own stale memory — exactly the failure this tool exists to remove (D-5).
    """
    from app.services import espn_extra

    asked_for = player.strip() if isinstance(player, str) else ""
    if not asked_for:
        # There is no name to resolve, so no fetch is worth making. Returning before the
        # roster hop keeps a forgotten argument from costing a live GET, and stops a
        # not-on-roster note that names nobody.
        return None

    roster = await espn_extra.fetch_team_roster(team)
    if roster is None:
        return None
    matches = espn_extra.find_roster_athletes(roster, asked_for)
    if matches is None:
        return None

    team_abbr = team.strip().upper() if isinstance(team, str) else ""
    # The roster hop carries ESPN's OWN current season, and it carries it even when the
    # hop MISSED, which is the only source for it on either path: the stats payload has
    # no current year and D-1 forbids reading the app's own tables from here.
    current_season = espn_extra.roster_season_year(roster)

    if len(matches) > 1:
        candidates = [str(match["display_name"]) for match in matches]
        return {
            "note": _AMBIGUOUS_PLAYER_NOTE.format(team=team_abbr, candidates=", ".join(candidates)),
            "candidates": candidates,
        }

    if matches:
        match = matches[0]
        name = str(match["display_name"])
        athlete_id = str(match["athlete_id"])
        identity: dict[str, object] = {
            "player": name,
            "team": team_abbr,
            "position": match["position"],
        }
    else:
        resolved = await _resolve_off_roster(asked_for, team_abbr)
        found_id = resolved.pop("athlete_id", None)
        if not isinstance(found_id, str):
            return resolved  # a terminal note: unfound, ambiguous, or search unavailable
        athlete_id = found_id
        name = str(resolved["player"])
        # Popped rather than carried: the model never sees an athlete id, so it can never
        # learn to send one back (D-4).
        identity = resolved

    # Past this point the resolution already PROVED who this player is, so a stats miss
    # must keep that identity (D-5) — bare ``None`` here made the model deny him.
    on_team = str(identity["team"])
    payload = await espn_extra.fetch_athlete_stats(athlete_id)
    if payload is None:
        return {**identity, "note": _STATS_FETCH_FAILED_NOTE.format(player=name, team=on_team)}
    facts = espn_extra.parse_athlete_stats(payload, season=season)
    if facts is None:
        return {**identity, "note": _NO_STATS_PUBLISHED_NOTE.format(player=name, team=on_team)}

    # The stats payload carries no name of its own, so identity comes from the resolution.
    facts = {**facts, **identity}
    selected = facts["season"]
    if selected is None:
        seasons = ", ".join(str(year) for year in facts["available_seasons"])
        facts["note"] = _SEASON_UNAVAILABLE_NOTE.format(
            player=name, seasons=seasons or "no seasons at all"
        )
        return facts

    if current_season is not None and selected >= current_season:
        # ``>=`` and not ``==``: a row ahead of the roster's own year is still a season
        # nobody has finished, so it must not be called an official total either.
        statement = _CURRENT_SEASON_STATEMENT.format(player=name, season=selected)
        games = espn_extra.games_played(facts["stats"])
        if games is not None:
            statement += _GAMES_PLAYED_CLAUSE.format(games=games, season=selected)
        facts["season_statement"] = statement
    else:
        facts["season_statement"] = _SEASON_STATEMENT.format(player=name, season=selected)
        if current_season is not None and current_season not in facts["available_seasons"]:
            facts["current_season_statement"] = _NO_CURRENT_SEASON_STATEMENT.format(
                player=name, current=current_season, season=selected
            )
    return facts


async def _resolve_off_roster(player: str, team_abbr: str) -> dict[str, object]:
    """Resolve a player who is NOT on the asked roster, through ESPN's player search.

    Reached only after the roster hop misses, which it does for every player who changed
    teams — the roster is the CURRENT one and a stats question is about a PAST season.
    Always returns a dict: one carrying ``athlete_id`` when EXACTLY one NFL player
    matches, otherwise a terminal ``note`` (D-5 — never bare ``None``, which would send
    the model back to the stale memory this tool exists to replace). More than one NFL
    match returns the candidates, never a silently picked one.
    """
    from app.services import espn_extra

    payload = await espn_extra.fetch_athlete_search(player)
    found = espn_extra.parse_athlete_search(payload) if payload is not None else None
    if found is None:
        return {"note": _SEARCH_UNAVAILABLE_NOTE.format(player=player, team=team_abbr)}
    if not found:
        return {"note": _NOT_ON_ROSTER_NOTE.format(player=player, team=team_abbr)}
    if len(found) > 1:
        candidates = [f"{one['display_name']} of the {one['team_name']}" for one in found]
        return {
            "note": _AMBIGUOUS_SEARCH_NOTE.format(
                player=player, team=team_abbr, candidates=", ".join(candidates)
            ),
            "candidates": candidates,
        }

    one = found[0]
    name = str(one["display_name"])
    found_team = str(one["team_name"])
    return {
        "athlete_id": str(one["athlete_id"]),
        "player": name,
        # The team the SEARCH reported, not the one that was asked about — it is the
        # grounded one, and the identity has to carry the team the figures belong to.
        "team": found_team,
        "position": None,
        "team_change_statement": _PLAYER_CHANGED_TEAMS_STATEMENT.format(
            player=name, asked_team=team_abbr, found_team=found_team
        ),
    }


# D-1b: the model narrates a year of its own choosing unless it can SEE the one it was
# given, and a bare integer in a dict body is readable but not voiceable. This sentence
# is the voiceable form; the integer ``season`` field stays alongside it as the
# machine-readable one. Built here and not in the parser, which never phrases and does
# not know the player's name.
_SEASON_STATEMENT = (
    "Every figure below is {player}'s official total for the {season} NFL season, so "
    "say {season} when you report any of them."
)

# Gap 2, measured 2026-08-20: ESPN's newest row for Mahomes was 2025 while the season
# being played was 2026, so "how many yards has he thrown this year" silently answered
# about a different season — and mid-season the trap inverts, because a partial current
# season row would be narrated as an official total. The current year comes from the
# roster payload (the stats payload carries none), so both cases can be told apart.
_CURRENT_SEASON_STATEMENT = (
    "Every figure below is {player}'s total SO FAR in the {season} NFL season, which is "
    "the season being played right now and is not finished, so say {season} when you "
    "report any of them and say that they are his figures so far rather than a finished "
    "season's total."
)
_GAMES_PLAYED_CLAUSE = " He has played {games} games in the {season} season so far."
_NO_CURRENT_SEASON_STATEMENT = (
    "ESPN publishes no figures at all for {player} in the {current} NFL season so far, "
    "so if the member was asking about this season you must tell him plainly that there "
    "are no {current} figures for {player} yet, and you must never report the {season} "
    "figures below as though they were this season's."
)

# D-5: each resolution miss is a concrete full sentence telling the model what to do
# next, returned in a dict body because a bare string fact gets voiced or swallowed
# (memory: qa-phrasing-inversion).
# The roster hop anchors on the CURRENT roster while the question is about a PAST season,
# so it misses every player who changed teams. Only reached once ESPN's own player search
# has ALSO failed to place him, which is why this no longer asks the model to supply the
# team — its team knowledge is the stale thing this tool replaces.
_NOT_ON_ROSTER_NOTE = (
    "No player named {player} is on the {team} roster, and ESPN's own player search "
    "found nobody by that name on any NFL team either, so this tool has no figures for "
    "him at all. Tell the member plainly that you could not find that player in ESPN's "
    "data, never give a figure from your own memory instead, and never guess which team "
    "he plays for."
)
_SEARCH_UNAVAILABLE_NOTE = (
    "No player named {player} is on the {team} roster, and the search that would have "
    "found which team he is on failed just now, so this tool has no figures for him this "
    "time. Say that you could not look him up, and never give a figure from your own "
    "memory instead."
)
_AMBIGUOUS_SEARCH_NOTE = (
    "No player named {player} is on the {team} roster, and ESPN's player search found "
    "more than one NFL player by that name: {candidates}. Ask the member which one of "
    "them he means, and do not report any figure until he answers."
)
# The team name here came from ESPN's search, so it is GROUNDED and the model may say it.
_PLAYER_CHANGED_TEAMS_STATEMENT = (
    "{player} does not play for {asked_team} any more. ESPN lists him with the "
    "{found_team} now, so say that he plays for the {found_team}, and the figures below "
    "are his own for the season this answer names."
)
_AMBIGUOUS_PLAYER_NOTE = (
    "More than one player on the {team} roster matches that name: {candidates}. Ask the "
    "member which one of them he means, and do not report any figure until he answers."
)
# Live-measured 2026-08-20: ESPN answers 200 with NO ``categories`` key at all for a
# rostered player who has no recorded stats (Mario Williams, LAR WR). Returning bare
# ``None`` there sent the model to _NO_DATA_PAYLOAD and it DENIED a player it had just
# resolved — "I don't recall a Mario Williams playing receiver for the Rams". D-5 says a
# miss returns a note; these two carry the roster identity so the denial cannot recur.
_NO_STATS_PUBLISHED_NOTE = (
    "{player} is on the {team} roster right now, but ESPN publishes no season "
    "statistics for him at all, so this tool has no figures for him. Say that he is on "
    "the roster and that you have no statistics for him, and never say that he does not "
    "play for that team."
)
_STATS_FETCH_FAILED_NOTE = (
    "{player} is on the {team} roster right now, but the statistics lookup for him "
    "failed just now, so this tool has no figures for him this time. Say that you could "
    "not retrieve his statistics, and never give a figure from your own memory instead."
)
_SEASON_UNAVAILABLE_NOTE = (
    "ESPN's table does not carry the season you asked about for {player}, so this tool "
    "has no figures for that season. The only seasons ESPN carries for {player} are "
    "{seasons}. Tell the member plainly that you do not have the season he asked "
    "about, and never give him a figure from a different season as if it were that one."
)

# INSTRUCT first, CONSTRAIN second — measured, not stylistic. In the predecessor task a
# description that only disclaimed a limitation suppressed the call 5/5 and the model
# fell back to stale memory. The season clause is D-1a: the model resolves "last year"
# against its TRAINING CUTOFF, which is the measured bug (Caleb Williams narrated as a
# rookie in 2026), so the year is taken out of its hands entirely.
_STATS_TOOL_DESCRIPTION = (
    "Look up one NFL player's official ESPN statistics for a single season, such as how "
    "many yards he threw or rushed for, how many touchdowns he scored, or how many "
    "games he played. Call this tool for ANY question that asks what a player did in a "
    "season, INCLUDING a question phrased as last year, last season or this season, "
    "because your own memory of which season is the most recent one, and of what a "
    "player did in it, is often a year or more out of date. The team argument is the "
    "standard abbreviation of the team the player is on right now, for example LAR for "
    "the Los Angeles Rams or PHI for the Philadelphia Eagles. The player argument is "
    "the player's name exactly as the member wrote it. Pass the season argument ONLY "
    "when the member named a specific year such as 2024. LEAVE THE SEASON ARGUMENT OUT "
    "for every other phrasing, including last year, last season and this season, "
    "because this tool already knows which season is the most recent one ESPN has and "
    "you do not. Never work out a year number for yourself from a phrase like last "
    "year. This tool finds the player even when he has changed teams, so pass the team "
    "the member named and let the tool tell you which team he is on now; when it says he "
    "plays for a different team, say so and use the figures it returns. If it reports "
    "that more than one player matches, ask the member which one he means instead of "
    "guessing. Every figure it returns belongs to the one season the answer names, so "
    "say that year when you report a figure and never describe it as this year's or "
    "last year's. When it says the season it reports is still being played, say that "
    "those are his figures so far and never call them a final total. When it says ESPN "
    "has no figures yet for the season being played now, tell the member that plainly "
    "instead of giving him an earlier season's figures as if they were this season's."
)

TOOLS: tuple[_Tool, ...] = (
    _Tool(
        name="lookup_team_roster",
        spec={
            "type": "function",
            "function": {
                "name": "lookup_team_roster",
                "description": _ROSTER_TOOL_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "team": {
                            "type": "string",
                            "description": "The team's standard abbreviation, such as CHI.",
                        },
                        "position": {
                            "type": "string",
                            "description": (
                                "Optional position abbreviation, such as QB. Leave it out "
                                "to get per-position counts instead of names."
                            ),
                        },
                    },
                    "required": ["team"],
                },
            },
        },
        run=_lookup_team_roster,
    ),
    _Tool(
        name="lookup_player_season_stats",
        spec={
            "type": "function",
            "function": {
                "name": "lookup_player_season_stats",
                "description": _STATS_TOOL_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "team": {
                            "type": "string",
                            "description": (
                                "The standard abbreviation of the team the player is on "
                                "right now, such as LAR."
                            ),
                        },
                        "player": {
                            "type": "string",
                            "description": "The player's name as the member wrote it.",
                        },
                        "season": {
                            "type": "integer",
                            "description": (
                                "The four-digit year, and ONLY when the member named "
                                "one. Leave it out for last year, last season or this "
                                "season."
                            ),
                        },
                    },
                    "required": ["team", "player"],
                },
            },
        },
        run=_lookup_player_season_stats,
    ),
)

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

    With an EMPTY :data:`TOOLS` registry (the fallback branch) this is exactly ONE
    :func:`app.bot.llm_client.open_chat` call with ``tools=None`` — byte-identical to
    the zero-tool behavior, with no extra round and no latency cost.

    With the shipped non-empty registry it loops at most :data:`_MAX_TOOL_ROUNDS` times against a
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
