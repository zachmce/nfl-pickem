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
``None`` on any HTTP error. The registry ships with FOUR tools —
``lookup_team_roster`` (issue #179), ``lookup_player_season_stats`` (issue #183),
``lookup_player_current_team`` and ``lookup_game_leaders`` (issue #183 Route D) — each
grounding its question in current ESPN data instead of the model's training cutoff; an
EMPTY registry stays a supported fallback branch. ``lookup_playoff_results`` is the
FIFTH, and it ships alongside the calendar preamble below rather than on its own: the
model was measured answering a finished season's Super Bowl from memory, and once it knew
what year it was that memory turned an honest hedge into a confident falsehood.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

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
# The CALENDAR preamble. NOTHING in this path told the model what day it is, so it
# resolved every relative date against its TRAINING CUTOFF and told members live that a
# finished season "hasn't happened yet" — twice, about the 2025 season and about the
# Super Bowl played in February 2026. This block is the fix, and it is deliberately in
# the ROLE half rather than in OPEN_GUARD: the guard constants stay byte-identical, and
# these sentences cost no tool round, so they ground even a question no tool covers.
#
# Date grounding ALONE is NOT enough and was measured to be worse: with only the date,
# the model answered that Kansas City won that Super Bowl, which is false (Seattle beat
# New England). An honest hedge became a confident falsehood. ``lookup_playoff_results``
# is the other half of this fix and the two ship together.
# --------------------------------------------------------------------------- #

_TODAY_STATEMENT = (
    "Today's date is {today}, and every day, every month and every year earlier than "
    "today is in the past."
)
_SEASON_NOW_STATEMENT = "The {season} NFL season is the season being played right now."
_SEASON_PHASE_CLAUSE = " The {season} season is in its {phase} at the moment."
# THE mapping the model got wrong both times, stated as its own sentence with the years
# filled in, because a rule it has to apply to work out a year is a rule it drops.
_SEASON_SPAN_STATEMENT = (
    "An NFL season spans two calendar years: a season named for a year starts in "
    "September of that year and ends with its Super Bowl in February of the following "
    "year. So the {previous} NFL season ran from September {previous} to its Super Bowl "
    "in February {season}, and the {season} NFL season will end with its Super Bowl in "
    "February {following}."
)
_MOST_RECENT_FINISHED_STATEMENT = (
    "The most recent NFL season to have finished is the {finished} season, and every "
    "game of it, including all of its playoff games and its Super Bowl in February "
    "{after}, has already been played."
)
# The same fact WITHOUT the superlative, for the round where the check below could not
# run: a season one behind the one ESPN names is finished under every calendar, but it is
# only the most recent finished one while the current season's Super Bowl is still ahead.
_FINISHED_SEASON_STATEMENT = (
    "The {finished} NFL season is over and finished, and every game of it, including all "
    "of its playoff games and its Super Bowl in February {after}, has already been played."
)
_UNKNOWN_SEASON_STATEMENT = (
    "You could not be told which NFL season is being played right now, so say nothing at "
    "all about which season that is and never name a year for it."
)
_SEASON_SPAN_RULE = (
    "An NFL season spans two calendar years: a season named for a year starts in "
    "September of that year and ends with its Super Bowl in February of the following "
    "year."
)
# Unconditional, and last so it is the most recent thing read: this is the exact sentence
# the live defect produced, and a caveat the model has to decide whether to apply is a
# caveat it drops (measured 3/3 on the predecessor task).
_NEVER_NOT_HAPPENED_YET_STATEMENT = (
    "Never tell the member that an NFL season in the past has not happened yet, and never "
    "tell him that a game that has already been played has not happened yet. Never tell "
    "him that a date earlier than today is still in the future. Your own sense of what "
    "year it is comes from your training and it is out of date, so the date given to you "
    "here is the one you use."
)


def _spoken_date(moment: datetime) -> str:
    """``moment`` as the date a person would say — "Friday 21 August 2026". Pure."""
    return f"{moment:%A} {moment.day} {moment:%B} {moment.year}"


async def _most_recently_finished_season(current: int) -> tuple[int, bool]:
    """The newest NFL season whose Super Bowl has been played, and whether it was CHECKED.

    ``current - 1`` is true under every calendar, but it is the MOST RECENT finished
    season only while the current season's own Super Bowl is still ahead of today. The
    check covers the weeks between a Super Bowl and ESPN rolling its season year over,
    which is the window the live defect was reported in. It costs one cached call, and it
    is the SAME cache entry ``lookup_playoff_results`` reads for a Super Bowl question, so
    whichever asks first warms the other. A failed check degrades to the true statement
    without the superlative rather than to a guessed year.
    """
    from app.services import espn_extra

    payload = await espn_extra.fetch_postseason_scoreboard(current, espn_extra.SUPER_BOWL_WEEK)
    facts = espn_extra.parse_postseason_round(payload) if payload is not None else None
    if facts is None:
        return current - 1, False
    return (current if facts["any_completed"] else current - 1), True


async def _calendar_facts() -> str:
    """The date-and-season sentences the OPEN role carries. Never raises, never empty.

    The date comes from the system clock, which is not a database read, so D-1 holds —
    this module still makes no ``db_bridge`` call. The season year and the phase come from
    ESPN's league root, the source the rest of this module already reads the season from,
    so there is no second derived answer to disagree with it; working a season year out
    from the month would be a guess about ESPN's own rollover calendar, and this is the
    value that must never be guessed. Every hop degrades to saying LESS rather than to
    naming a year it could not read.
    """
    from app.services import espn_extra

    sentences = [_TODAY_STATEMENT.format(today=_spoken_date(datetime.now(UTC)))]
    league = await espn_extra.fetch_league()
    season = espn_extra.league_season_year(league)
    if season is None:
        sentences.append(_SEASON_SPAN_RULE)
        sentences.append(_UNKNOWN_SEASON_STATEMENT)
    else:
        playing = _SEASON_NOW_STATEMENT.format(season=season)
        phase = espn_extra.league_season_phase(league)
        if phase is not None:
            playing += _SEASON_PHASE_CLAUSE.format(season=season, phase=phase)
        sentences.append(playing)
        sentences.append(
            _SEASON_SPAN_STATEMENT.format(season=season, previous=season - 1, following=season + 1)
        )
        finished, checked = await _most_recently_finished_season(season)
        template = _MOST_RECENT_FINISHED_STATEMENT if checked else _FINISHED_SEASON_STATEMENT
        sentences.append(template.format(finished=finished, after=finished + 1))
    sentences.append(_NEVER_NOT_HAPPENED_YET_STATEMENT)
    return " ".join(sentences)


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
    "date. This tool answers a question about a team the member has named. When the "
    "member names a player instead and asks which team that player is on now, "
    "lookup_player_current_team is the tool for that question and this one is not."
)


async def _lookup_player_season_stats(
    player: str = "", team: str = "", season: int | None = None
) -> object | None:
    """Look up ONE player's ESPN totals for ONE season.

    ``team`` is a HINT, not a requirement, and that is a MEASURED decision. While it was
    required, the served model spent a whole extra round filling it in — it called
    ``lookup_player_current_team`` first and this tool second, 3/3, on the commonest
    question there is, burning two of the three rounds the loop allows. So with no team
    in hand the name goes STRAIGHT to ESPN's player search: one cached hop to resolve the
    athlete id, one to fetch the career table, and no roster fetch merely to obtain an id.

    With a team in hand the roster hop runs first, unchanged: it is cached and often
    already warm from a roster question, and it reads the RAW payload because
    :func:`~app.services.espn_extra.parse_team_roster` drops the id on purpose (D-7). A
    roster miss still falls through to :func:`_resolve_off_roster`, which is how a player
    asked about on the team he PLAYED that season for still resolves. No id ever comes
    from the model on either path.

    The season being played comes from neither hop. It is read from ESPN's league root
    (:func:`~app.services.espn_extra.fetch_league`), which needs no team and so answers
    both paths identically — the fix for the live 2026-08-21 defect, where a team-less
    question read no roster, got no year, and the model invented one.

    Every argument defaults so a model that forgets one degrades rather than raising a
    TypeError into the loop. A resolution miss returns a NOTE dict, never ``None``,
    because ``None`` becomes :data:`_NO_DATA_PAYLOAD`, which tells the model to answer
    from its own stale memory — exactly the failure this tool exists to remove (D-5).
    """
    from app.services import espn_extra

    asked_for = player.strip() if isinstance(player, str) else ""
    if not asked_for:
        # There is no name to resolve, so no fetch is worth making. Returning before the
        # first hop keeps a forgotten argument from costing a live GET, and stops a
        # not-found note that names nobody.
        return None

    team_abbr = team.strip().upper() if isinstance(team, str) else ""
    if team_abbr:
        roster = await espn_extra.fetch_team_roster(team_abbr)
        if roster is None:
            return None
        matches = espn_extra.find_roster_athletes(roster, asked_for)
        if matches is None:
            return None
    else:
        matches = []

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
        identity: dict[str, object] = {"player": name, "position": match["position"]}
        # Kept OUT of ``identity``: the team a player is on today is not the team a past
        # season's figures belong to, and a team name the model can see is one it may
        # attach to the season. It reaches the model only inside the two roster miss
        # notes below, where there is no season for it to be attached to.
        on_roster = team_abbr
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
        on_roster = None

    # Past this point the resolution already PROVED who this player is, so a stats miss
    # must keep that identity (D-5) — bare ``None`` here made the model deny him.
    payload = await espn_extra.fetch_athlete_stats(athlete_id)
    if payload is None:
        note = _note(
            _STATS_FETCH_FAILED_NOTE, _SEARCHED_STATS_FETCH_FAILED_NOTE, team=on_roster, player=name
        )
        return {**identity, "note": note}
    facts = espn_extra.parse_athlete_stats(payload, season=season)
    if facts is None:
        note = _note(
            _NO_STATS_PUBLISHED_NOTE, _SEARCHED_NO_STATS_PUBLISHED_NOTE, team=on_roster, player=name
        )
        return {**identity, "note": note}

    # The stats payload carries no name of its own, so identity comes from the resolution.
    facts = {**facts, **identity}
    selected = facts["season"]
    if selected is None:
        seasons = ", ".join(str(year) for year in facts["available_seasons"])
        facts["note"] = _SEASON_UNAVAILABLE_NOTE.format(
            player=name, seasons=seasons or "no seasons at all"
        )
        return facts

    # ONE source for the season being played, read on BOTH paths so neither can produce a
    # statement the other cannot. It came off the ROSTER until 2026-08-21, so a team-less
    # question had no year, both statements below were skipped, and the model filled that
    # silence with "since that season is still being played" about a finished season.
    # Read here, after the figures are in hand, so no resolution miss costs a request.
    current_season = espn_extra.league_season_year(await espn_extra.fetch_league())

    if current_season is not None and selected >= current_season:
        # ``>=`` and not ``==``: a row ahead of the league's own year is still a season
        # nobody has finished, so it must not be called an official total either.
        statement = _CURRENT_SEASON_STATEMENT.format(player=name, season=selected)
        games = espn_extra.games_played(facts["stats"])
        if games is not None:
            statement += _GAMES_PLAYED_CLAUSE.format(games=games, season=selected)
    else:
        statement = _SEASON_STATEMENT.format(player=name, season=selected)
        if current_season is not None:
            statement += _FINISHED_SEASON_CLAUSE.format(season=selected, current=current_season)
            if current_season not in facts["available_seasons"]:
                facts["current_season_statement"] = _NO_CURRENT_SEASON_STATEMENT.format(
                    player=name, current=current_season, season=selected
                )
    facts["season_statement"] = statement + _season_team_clause(
        name, selected, facts["season_teams"]
    )
    return facts


def _join_teams(teams: list[str]) -> str:
    """Join club names into one readable phrase — "the A and the B", "the A, the B and…"."""
    articled = [f"the {team}" for team in teams]
    if len(articled) < 3:
        return " and ".join(articled)
    return f"{', '.join(articled[:-1])} and {articled[-1]}"


def _season_team_clause(player: str, season: int, teams: object) -> str:
    """The sentence naming the team ``player`` played for IN ``season``.

    The whole point of the tool's team reporting: the club the FIGURES belong to is the
    only club the model is told about, so it cannot borrow another team name in the
    payload and hang it on a past season (the live Pacheco defect).
    """
    names = [str(team) for team in teams] if isinstance(teams, list) else []
    if not names:
        return _UNKNOWN_SEASON_TEAM_CLAUSE.format(player=player, season=season)
    if len(names) == 1:
        return _SEASON_TEAM_CLAUSE.format(player=player, season=season, team=names[0])
    return _SPLIT_SEASON_TEAM_CLAUSE.format(player=player, season=season, teams=_join_teams(names))


def _note(with_team: str, without_team: str, *, team: str | None, **fields: str) -> str:
    """Pick a D-5 note's team-naming wording, or its team-less twin. Pure.

    Every note past the point identity is proven comes in a pair, because half of them
    affirm a club ("he IS on the KC roster") and that affirmation is only true when a
    roster was actually read. Since ``team`` became optional the same rule covers the
    resolution misses too: a note formatted with an empty ``team`` reads as a dangling
    "on the  roster", which is precisely the sentence this makes impossible.
    """
    if team:
        return with_team.format(team=team, **fields)
    return without_team.format(**fields)


async def _resolve_off_roster(player: str, team_abbr: str) -> dict[str, object]:
    """Resolve a player through ESPN's player search. ``team_abbr`` may be empty.

    The FIRST hop when no team was asked about, and the fallback hop after a roster miss
    — which the roster takes for every player asked about on the team he played that
    season for, since the roster is the CURRENT one. Always returns a dict: one carrying
    ``athlete_id`` when EXACTLY one NFL player matches, otherwise a terminal ``note``
    (D-5 — never bare ``None``, which would send the model back to the stale memory this
    tool exists to replace). More than one NFL match returns the candidates, never a
    silently picked one, on either path.

    The team the search placed him on is used to FIND him and is then dropped. It is
    where he plays today, which is not the team a past season's figures belong to, and
    narrating it is what produced the live "he played for the Detroit Lions in 2025".
    """
    from app.services import espn_extra

    payload = await espn_extra.fetch_athlete_search(player)
    found = espn_extra.parse_athlete_search(payload) if payload is not None else None
    if found is None:
        note = _note(
            _SEARCH_UNAVAILABLE_NOTE, _NAME_SEARCH_FAILED_NOTE, team=team_abbr, player=player
        )
        return {"note": note}
    if not found:
        note = _note(_NOT_ON_ROSTER_NOTE, _NAME_NOT_IN_NFL_NOTE, team=team_abbr, player=player)
        return {"note": note}
    if len(found) > 1:
        candidates = [f"{one['display_name']} of the {one['team_name']}" for one in found]
        return {
            "note": _note(
                _AMBIGUOUS_SEARCH_NOTE,
                _AMBIGUOUS_NAME_NOTE,
                team=team_abbr,
                player=player,
                candidates=", ".join(candidates),
            ),
            "candidates": candidates,
        }

    one = found[0]
    return {
        "athlete_id": str(one["athlete_id"]),
        "player": str(one["display_name"]),
        "position": None,
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

# THE FIX for the live 2026-08-20 defect. Asked what Pacheco averaged per carry last year,
# the payload named only his CURRENT club and the model answered that he played for the
# Detroit Lions in 2025 — he played for Kansas City. ESPN's own per-season row carries the
# season's team, so the season's team is now the one and only club the model is handed.
_SEASON_TEAM_CLAUSE = (
    " {player} played for the {team} in the {season} season, so say the {team} whenever "
    "you say which team he was playing for while he put up any of these figures. The "
    "{team} is the only team you may name anywhere in your answer about the {season} "
    "season."
)
# Measured live: a split season yields one row per club PLUS a combined row, so the
# figures below are the whole year's and belong to no single club.
_SPLIT_SEASON_TEAM_CLAUSE = (
    " {player} played for more than one team during the {season} season: he played for "
    "{teams} that year, and every figure below is his combined total across all of them. "
    "Say that he split the {season} season between {teams}, and never name just one of "
    "them as the team he played for that season."
)
_UNKNOWN_SEASON_TEAM_CLAUSE = (
    " ESPN does not say here which team {player} played for in the {season} season, so "
    "say nothing at all about which team he was playing for that year and never name one."
)

# Gap 2, measured 2026-08-20: ESPN's newest row for Mahomes was 2025 while the season
# being played was 2026, so "how many yards has he thrown this year" silently answered
# about a different season — and mid-season the trap inverts, because a partial current
# season row would be narrated as an official total. The current year comes from ESPN's
# league root (the stats payload carries none), so both cases can be told apart.
_CURRENT_SEASON_STATEMENT = (
    "Every figure below is {player}'s total SO FAR in the {season} NFL season, which is "
    "the season being played right now and is not finished, so say {season} when you "
    "report any of them and say that they are his figures so far rather than a finished "
    "season's total."
)
_GAMES_PLAYED_CLAUSE = " He has played {games} games in the {season} season so far."
# THE FIX for the live 2026-08-21 defect: handed a finished season and no statement about
# which season is being played, the model wrote its own — "since that season is still
# being played" about 2025. _SEASON_STATEMENT already called those figures an official
# total and that did not stop it, so the finish is now SAID, on every past season and not
# only the ones _NO_CURRENT_SEASON_STATEMENT covers. Unconditional in wording, because a
# caveat the model has to decide whether to apply is a caveat it drops (measured 3/3).
_FINISHED_SEASON_CLAUSE = (
    " The {season} NFL season is over and finished, and the {current} NFL season is the "
    "one being played now. Never say that the {season} season is still being played, and "
    'never say the words "so far" about any figure below.'
)
# Live-measured 2026-08-20: the FIRST wording made this conditional ("if the member was
# asking about this season..."), and the model did not evaluate the condition — 3/3 it
# answered "3,587 yards SO FAR in the 2025 season" to a "this year" question and never
# said 2026 had no figures. State the fact unconditionally and ban the phrasing outright;
# a caveat the model has to decide whether to apply is a caveat it drops.
_NO_CURRENT_SEASON_STATEMENT = (
    "The {current} NFL season is the season happening now, and ESPN publishes no figures "
    "at all for {player} in it yet. The figures below are from the {season} season, which "
    "is over and finished. Say the year {season} every time you report any of them. Never "
    "call them this season's figures, never call them this year's figures, and never say "
    'the words "so far" about them, because a finished season has no so far. If the '
    "member asked about this season or this year, tell him plainly that ESPN has no "
    "{current} figures for {player} yet, and give him the {season} figures only after you "
    "have told him that."
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
_AMBIGUOUS_PLAYER_NOTE = (
    "More than one player on the {team} roster matches that name: {candidates}. Ask the "
    "member which one of them he means, and do not report any figure until he answers."
)
# The team-less twins of the three notes above, for the path where no team was asked
# about and there is therefore no roster to say he is missing from. Each one carries the
# same instruction as its twin; only the club disappears, because naming an empty team is
# the dangling sentence :func:`_note` exists to prevent.
_NAME_NOT_IN_NFL_NOTE = (
    "ESPN's own player search found nobody in the NFL named {player}, so this tool has "
    "no figures for him at all. Tell the member plainly that you could not find that "
    "player in ESPN's data, never give a figure from your own memory instead, and never "
    "guess which team he plays for."
)
_NAME_SEARCH_FAILED_NOTE = (
    "The player search that would have found {player} failed just now, so this tool has "
    "no figures for him this time. Say that you could not look him up, and never give a "
    "figure from your own memory instead."
)
_AMBIGUOUS_NAME_NOTE = (
    "ESPN's player search found more than one NFL player whose name matches {player}, "
    "and here is each of them with the team he is on right now: {candidates}. Ask the "
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
# The search-fallback twins of the two notes above. They affirm the identity the search
# proved WITHOUT naming a club, because the only club the search knows is the one he is on
# today and this branch has no season to attach it to.
_SEARCHED_NO_STATS_PUBLISHED_NOTE = (
    "ESPN's own data does list {player} as a current NFL player, but it publishes no "
    "season statistics for him at all, so this tool has no figures for him. Say that you "
    "found him and that you have no statistics for him, and never say that he is not an "
    "NFL player."
)
_SEARCHED_STATS_FETCH_FAILED_NOTE = (
    "ESPN's own data does list {player} as a current NFL player, but the statistics "
    "lookup for him failed just now, so this tool has no figures for him this time. Say "
    "that you could not retrieve his statistics, and never give a figure from your own "
    "memory instead."
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
    "player did in it, is often a year or more out of date. The player argument is the "
    "player's name exactly as the member wrote it, and it is the only argument this tool "
    "needs, because it finds the player even when he has changed teams. The team "
    "argument is optional. Pass the team argument only when the member's own question "
    "names a team, and then it is that team's standard abbreviation, for example LAR for "
    "the Los Angeles Rams or PHI for the Philadelphia Eagles. When his question names no "
    "team, leave the team argument out and call this tool with the player's name alone. "
    "Never call lookup_player_current_team first so that you can fill in the team "
    "argument here, because this tool finds the player without it and calling another "
    "tool first only spends a turn you need for the answer. Pass the season argument ONLY "
    "when the member named a specific year such as 2024. LEAVE THE SEASON ARGUMENT OUT "
    "for every other phrasing, including last year, last season and this season, "
    "because this tool already knows which season is the most recent one ESPN has and "
    "you do not. Never work out a year number for yourself from a phrase like last "
    "year. This tool tells you which team he played for in the season it reports, and "
    "that team is the only team you ever name in your answer, "
    "because the team a player is on today is not the team a past season's figures "
    "belong to. If it reports "
    "that more than one player matches, ask the member which one he means instead of "
    "guessing. Every figure it returns belongs to the one season the answer names, so "
    "say that year when you report a figure and never describe it as this year's or "
    "last year's. When it says the season it reports is still being played, say that "
    "those are his figures so far and never call them a final total. When it says ESPN "
    "has no figures yet for the season being played now, tell the member that plainly "
    "instead of giving him an earlier season's figures as if they were this season's. "
    "This tool answers what a player DID in a season. When the member asks only which "
    "team a player is on now, lookup_player_current_team is the tool for that question "
    "and this one is not."
)


# --------------------------------------------------------------------------- #
# The CURRENT-TEAM tool. Its own tool rather than a field on the stats payload, for a
# LIVE-MEASURED reason: the stats payload used to carry the player's current club and
# the model glued that club to a PAST season — it said Pacheco played for Detroit in
# 2025, when he played for Kansas City. A field the model can see is a field it may
# voice, so the club a player is on today now reaches it only when the question asked
# for it.
# --------------------------------------------------------------------------- #


async def _lookup_player_current_team(player: str = "") -> object | None:
    """Look up which NFL club ``player`` is on RIGHT NOW, through ESPN's player search.

    ONE hop, cached, through the ``espn_extra`` seam (deferred import, as in the other
    two adapters). There is deliberately no team argument and no roster hop: the search
    payload already carries the club, and the whole point of this tool is that the asker
    does not know the team. The athlete id the search carries is dropped here and never
    reaches the model (D-4).

    ``player`` defaults so a model that forgets it degrades rather than raising a
    TypeError into the loop, and an empty name returns before the fetch — there is
    nothing to resolve, so no live GET is worth making. Past that point EVERY outcome is
    a note, never bare ``None``, because ``None`` becomes :data:`_NO_DATA_PAYLOAD`,
    which sends the model to the stale memory this tool exists to replace (D-5).
    """
    from app.services import espn_extra

    asked_for = player.strip() if isinstance(player, str) else ""
    if not asked_for:
        return None

    payload = await espn_extra.fetch_athlete_search(asked_for)
    found = espn_extra.parse_athlete_search(payload) if payload is not None else None
    if found is None:
        return {"note": _CURRENT_TEAM_SEARCH_FAILED_NOTE.format(player=asked_for)}
    if not found:
        return {"note": _NO_SUCH_NFL_PLAYER_NOTE.format(player=asked_for)}
    if len(found) > 1:
        # Measured live 2026-08-20: "josh allen" is THREE NFL players, and the club is
        # the only thing that tells them apart — so each candidate is named with his.
        candidates = [f"{one['display_name']} of the {one['team_name']}" for one in found]
        return {
            "note": _AMBIGUOUS_CURRENT_TEAM_NOTE.format(
                player=asked_for, candidates=", ".join(candidates)
            ),
            "candidates": candidates,
        }

    one = found[0]
    name, team = str(one["display_name"]), str(one["team_name"])
    return {
        "player": name,
        "current_team": team,
        "current_team_statement": _CURRENT_TEAM_STATEMENT.format(player=name, team=team),
    }


# The voiceable form of the two fields above. A dict field is readable but not
# voiceable, and the phrasing hazard applies to what the model READS as much as to what
# it says (memory: qa-phrasing-inversion). The past-season ban is stated
# UNCONDITIONALLY, because a caveat the model has to decide whether to apply is a caveat
# it drops (measured 3/3 on this branch).
_CURRENT_TEAM_STATEMENT = (
    "{player} plays for the {team} right now, because that is the team ESPN lists him "
    "on today. Say the {team} when you tell the member which team he plays for now. "
    "This is the team he is on today and it is not the team he played for in any earlier "
    "season, so never say that he played for the {team} in a past season and never "
    "attach the {team} to a year."
)
_AMBIGUOUS_CURRENT_TEAM_NOTE = (
    "ESPN's player search found more than one NFL player whose name matches {player}, "
    "and here is each of them with the team he is on right now: {candidates}. Ask the "
    "member which one of them he means, name none of them as the answer yet, and never "
    "pick one of them yourself."
)
_NO_SUCH_NFL_PLAYER_NOTE = (
    "ESPN lists no NFL player named {player} at all, so this tool cannot tell you which "
    "team he is on. Tell the member plainly that you could not find that player in "
    "ESPN's data, and never name a team for him from your own memory."
)
_CURRENT_TEAM_SEARCH_FAILED_NOTE = (
    "The player search that would have found which NFL team {player} is on failed just "
    "now, so this tool has no answer for him this time. Say that you could not look him "
    "up, and never name a team for him from your own memory."
)

# INSTRUCT first, CONSTRAIN second — measured twice on this branch, not stylistic: a
# disclaimer-only description suppressed the call 5/5, and a conditional caveat was
# ignored 3/3. The opening sentence is also what keeps this tool distinct from the other
# two at selection time (roster = who is on a team, stats = what a player did in a
# season, this = which team a player is on now).
_CURRENT_TEAM_TOOL_DESCRIPTION = (
    "Look up which NFL team one player is on RIGHT NOW. Call this tool every time you "
    "are asked which team a player plays for now, who he plays for, where he plays, or "
    "which team he is on this season, because players change teams every year and your "
    "own memory of where a player plays is often a year or more out of date. The player "
    "argument is the player's name exactly as the member wrote it. There is no team "
    "argument, because this tool finds the player without being told where he is, which "
    "is the whole reason to call it. This tool knows only the team he is on today. It "
    "has no statistics of any kind, it does not know what any player did in any season, "
    "and it does not know which team he played for in any earlier season, so never "
    "attach the team it names to a past year. If it reports that more than one NFL "
    "player matches the name, ask the member which one of them he means instead of "
    "guessing. If it reports that ESPN lists no NFL player by that name, tell the member "
    "that plainly and never name a team for him from your own memory."
)


# --------------------------------------------------------------------------- #
# The GAME tool (Route D of issue #183). It owns the game as a WHOLE — who led it and
# who won it — off the SAME ``summary`` payload the injuries path already caches. What a
# named player did in a named game is deliberately NOT here: ``athletes/{id}/gamelog``
# answers that for every game in one cached payload, where this path would cost a
# schedule fetch plus a summary fetch per game asked about (D-1).
# --------------------------------------------------------------------------- #


async def _lookup_game_leaders(
    team: str = "", week: int | None = None, season: int | None = None, playoff_round: str = ""
) -> object | None:
    """Look up who led ONE NFL game, regular season or postseason, and who won it.

    Two cached hops on either happy path: one hop resolves WHICH game is meant, then the
    game summary yields the leaders. ``playoff_round`` picks the resolver — with it the
    postseason scoreboard resolves the game (:func:`_playoff_game_leaders`), without it the
    team's regular-season schedule does. The summary is NOT fetched on any miss branch —
    the measured unplayed payload is 109 KB and carries no ``leaders`` key at all, so
    fetching it would cost that much to learn nothing.

    The postseason branch is the fix for the live 2026-08-21 defect: asked who led the
    Super Bowl in rushing, this tool could reach only the regular-season schedule, so it
    answered about a week-18 game against a different opponent with a different rushing
    total and never said it had changed games. ``playoff_round`` reuses
    ``lookup_playoff_results``' round vocabulary EXACTLY, so the model learns one set of
    round names rather than two, and an unreachable game now returns a note that says which
    game could not be found instead of a reachable game's figures.

    ``week`` and ``season`` never reach a URL as the model wrote them: ``week`` is a parser
    argument only (and is ignored entirely on the postseason branch, where the round
    already fixes the week), and ``season`` passes an integer range check inside the seam
    before the format string runs. With no ``week`` the most recent COMPLETED game is
    selected and the payload says which week that was (D-6); with no ``season`` ESPN's own
    ``requestedSeason`` echo names the year, so the model never works one out from "last
    year" (D-5).

    Every argument defaults so a model that forgets one degrades rather than raising a
    TypeError into the loop, and a question naming neither a team nor a round returns
    before the first hop — there is nothing to resolve, so no live GET is worth making.
    EVERY outcome is a NOTE, never bare ``None``, which becomes :data:`_NO_DATA_PAYLOAD`
    and sends the model back to the stale memory this tool exists to replace (D-5 of the
    predecessor task).
    """
    from app.services import espn_extra

    team_abbr = team.strip().upper() if isinstance(team, str) else ""
    asked_round = playoff_round.strip() if isinstance(playoff_round, str) else ""
    asked_week = week if isinstance(week, int) and not isinstance(week, bool) else None
    asked_season = season if isinstance(season, int) and not isinstance(season, bool) else None

    if asked_round:
        return await _playoff_game_leaders(team_abbr, asked_round, asked_season)
    if not team_abbr:
        return {"note": _NO_GAME_TO_LOOK_UP_NOTE}

    payload = await espn_extra.fetch_team_schedule(team_abbr, season=asked_season)
    if payload is None:
        return None
    schedule = espn_extra.parse_team_schedule(payload, week=asked_week)
    if schedule is None:
        return None

    # D-7, checked ONCE and on a deliberately narrow predicate: "the season has not
    # started" falls back, "that week has not been played yet" does not. Measured
    # 2026-08-21, the current regular season had 0 completed games, so without this every
    # default-path question declines for the whole offseason — and a decline is silence.
    unstarted_season: int | None = None
    if asked_season is None and not schedule["any_completed"]:
        newer = schedule["season"]
        if isinstance(newer, int):
            older = await espn_extra.fetch_team_schedule(team_abbr, season=newer - 1)
            fallback = (
                espn_extra.parse_team_schedule(older, week=asked_week)
                if older is not None
                else None
            )
            if fallback is not None and fallback["any_completed"]:
                schedule, unstarted_season = fallback, newer

    club = schedule["team"] or team_abbr
    year = schedule["season"]
    game = schedule["game"]

    if game is None:
        if asked_week is not None and asked_week == schedule["bye_week"]:
            return {"note": _BYE_WEEK_NOTE.format(team=club, week=asked_week, season=year)}
        if asked_week is not None:
            return {"note": _NO_GAME_THAT_WEEK_NOTE.format(team=club, week=asked_week, season=year)}
        return {"note": _NO_COMPLETED_GAMES_NOTE.format(team=club, season=year)}

    fixture = game["name"] if isinstance(game["name"], str) else f"the {club} game"
    if not game["completed"]:
        when = game["date"] if isinstance(game["date"], str) else "a date ESPN does not give"
        return {
            "note": _NOT_YET_PLAYED_NOTE.format(
                game=fixture, date=when, week=game["week"], season=year
            )
        }

    summary = await espn_extra.fetch_game_summary(game["event_id"])
    facts = espn_extra.parse_game_leaders(summary) if summary is not None else None
    if facts is None:
        # The game identity was already PROVED, so the miss keeps it: a bare miss after a
        # successful resolution made the model deny what it had just found (260820-s5y).
        return {"note": _NO_LEADERS_NOTE.format(game=fixture, week=game["week"], season=year)}

    # Phrased HERE and not in the parser, which never phrases: a bare integer in a dict
    # body is readable but not voiceable, and the sentence is what the model repeats.
    statement = _GAME_STATEMENT.format(game=fixture, week=game["week"], season=year)
    winner = facts["winner"]
    if isinstance(winner, str):
        statement += _GAME_WINNER_CLAUSE.format(winner=winner)
    statement += _NO_SCORE_CLAUSE + _REGULAR_SEASON_ONLY_CLAUSE
    if unstarted_season is not None:
        statement = (
            _UNSTARTED_SEASON_STATEMENT.format(team=club, current=unstarted_season, season=year)
            + " "
            + statement
        )

    return {
        "leaders": facts["leaders"],
        "winner": winner,
        "season": year,
        "week": game["week"],
        "game": fixture,
        "game_statement": statement,
        "caveat": espn_extra.GAME_LEADERS_CAVEAT,
    }


async def _playoff_game_leaders(team: str, asked_round: str, season: int | None) -> dict:
    """Look up who led ONE POSTSEASON game of ``asked_round``, and who won it.

    The 2026-08-21 defect's fix. The round resolves to a LITERAL week through
    :func:`~app.services.espn_extra.postseason_round_week` exactly as
    ``lookup_playoff_results`` does, so the Pro Bowl's week 4 stays unreachable and no
    model-written number ever reaches a URL. ``team`` selects WHICH game of a multi-game
    round is meant and never reaches a URL at all on this path — the scoreboard URL carries
    only the season and the week — so it needs no abbreviation allowlist here.

    Never substitutes a game it could not reach: a team that did not play in the round, a
    round with several games and no team to pick one, and a game whose summary carries no
    leaders each return a note naming what is missing, and no note ever carries another
    game's figures. Always returns a dict, never bare ``None`` (D-5).
    """
    from app.services import espn_extra

    if espn_extra.asked_for_the_pro_bowl(asked_round):
        return {"note": _PRO_BOWL_LEADERS_NOTE}
    week = espn_extra.postseason_round_week(asked_round)
    if week is None:
        return {"note": _UNKNOWN_ROUND_NOTE}

    if season is None:
        current = espn_extra.league_season_year(await espn_extra.fetch_league())
        if current is None:
            return {"note": _NO_SEASON_TO_ASK_ABOUT_NOTE}
        season, _checked = await _most_recently_finished_season(current)

    label = espn_extra.POSTSEASON_ROUND_LABELS[week]
    payload = await espn_extra.fetch_postseason_scoreboard(season, week)
    games = espn_extra.find_postseason_games(payload, week=week) if payload is not None else None
    if not games:
        return {"note": _NO_POSTSEASON_RESULTS_NOTE.format(season=season, round=label)}
    if not any(game["completed"] for game in games):
        return {"note": _POSTSEASON_NOT_PLAYED_NOTE.format(season=season, round=label)}

    matches = espn_extra.postseason_games_for_team(games, team) if team else games
    if not matches:
        return {
            "note": _TEAM_NOT_IN_ROUND_NOTE.format(
                round=label, season=season, matchups=_matchups(games)
            )
        }
    if len(matches) > 1:
        return {
            "note": _WHICH_PLAYOFF_GAME_NOTE.format(
                round=label, season=season, matchups=_matchups(matches)
            )
        }

    game = matches[0]
    fixture = _game_phrase(game["game"]) if isinstance(game["game"], str) else "that game"
    if not game["completed"]:
        return {
            "note": _PLAYOFF_GAME_NOT_PLAYED_NOTE.format(game=fixture, round=label, season=season)
        }

    event_id = game["event_id"]
    summary = await espn_extra.fetch_game_summary(event_id) if event_id is not None else None
    facts = espn_extra.parse_game_leaders(summary) if summary is not None else None
    if facts is None:
        # The game identity was already PROVED, so the miss keeps it: a bare miss after a
        # successful resolution made the model deny what it had just found (260820-s5y).
        return {"note": _NO_PLAYOFF_LEADERS_NOTE.format(game=fixture, round=label, season=season)}

    statement = _PLAYOFF_GAME_STATEMENT.format(
        game=fixture,
        teams=_join_teams([str(club) for club in game["teams"]]),
        round=label,
        season=season,
    )
    winner = facts["winner"]
    if isinstance(winner, str):
        statement += _GAME_WINNER_CLAUSE.format(winner=winner)
    statement += _NO_SCORE_CLAUSE

    return {
        "leaders": facts["leaders"],
        "winner": winner,
        "season": season,
        "round": label,
        "game": fixture,
        "game_statement": statement,
        "caveat": espn_extra.GAME_LEADERS_CAVEAT,
    }


def _matchups(games: list[dict]) -> str:
    """The clubs of each of a round's games, as a phrase a person would say. Pure.

    Names WHICH games a round holds without carrying a single figure out of any of them —
    the distinction requirement 3 of the defect report turns on.
    """
    return "; ".join(
        _join_teams([str(club) for club in game["teams"]])
        for game in games
        if isinstance(game.get("teams"), list) and game["teams"]
    )


# D-3: the game is STATED, never implied. One concrete full sentence naming both clubs,
# the week and the season, so the model never has to work out which game it is holding —
# and so the residual D-10 hazard (the member names an opponent, the model omits the week,
# a different game comes back) is answered in the payload rather than assumed away.
_GAME_STATEMENT = (
    "Every figure below comes from ONE single NFL game: {game}, played in week {week} of "
    "the {season} NFL season. Name both of those teams and say week {week} of {season} "
    "whenever you report any figure from this answer, so the member knows exactly which "
    "game you are talking about."
)
# D-2: the winner IS returned. It is not on OPEN_OWNERSHIP_CLAUSE's list, and "who won" is
# the first thing anyone asks about a game — leaving it out leaves the biggest hole in the
# answer for the model to fill, and it has no real memory of this result to fall back on.
_GAME_WINNER_CLAUSE = " The {winner} won that game."
# D-2, and UNCONDITIONAL rather than an "if": the score is never read out of the payload at
# all, because a field the model can see is a field it may voice — and OPEN_OWNERSHIP_CLAUSE
# already forbids stating one, so a score here would contradict the system prompt.
_NO_SCORE_CLAUSE = (
    " The final score of that game is not in this answer at all. Never state the score of "
    "that game, never say how many points either team scored, and never work a score out "
    "from the figures below."
)
# D-7, in the shape _NO_CURRENT_SEASON_STATEMENT was measured working in: state the fact
# unconditionally and ban the wrong phrasing outright.
_UNSTARTED_SEASON_STATEMENT = (
    "The {current} NFL season has not started yet and the {team} have not played a game "
    "in it at all, so the game described below is from the {season} season instead, which "
    "is the most recent season they played. Say the year {season} when you talk about this "
    "game, and never call it a game from this season or from this year."
)

# Every miss is a concrete full sentence telling the model what to do next, returned in a
# dict body because a bare string fact gets voiced or swallowed (memory:
# qa-phrasing-inversion) and a bare ``None`` sends it back to its own stale memory.
_BYE_WEEK_NOTE = (
    "The {team} did not play at all in week {week} of the {season} NFL season, because "
    "that week was their bye week. Tell the member plainly that they were on their bye "
    "week that week and had no game, and never give him a different week's game instead."
)
_NO_GAME_THAT_WEEK_NOTE = (
    "ESPN's schedule shows no {team} regular-season game in week {week} of the {season} "
    "NFL season, so this tool has no game at all for that week. Tell the member plainly "
    "that you have no game for that week, and never give him a different week's game "
    "instead."
)
_NO_COMPLETED_GAMES_NOTE = (
    "The {team} have not finished a single regular-season game in the {season} NFL season, "
    "so this tool has no game to report for them. Tell the member plainly that ESPN has no "
    "finished {season} game for them yet, and never describe a game from your own memory "
    "instead."
)
_NOT_YET_PLAYED_NOTE = (
    "{game} is scheduled for {date}, in week {week} of the {season} NFL season, and it has "
    "not been played yet, so there are no figures from it at all. Tell the member plainly "
    "that the game has not been played yet and say when it is scheduled for, and never "
    "describe how it went or who led it."
)
_NO_LEADERS_NOTE = (
    "This tool did find the game the member asked about — {game}, in week {week} of the "
    "{season} NFL season — but ESPN publishes no game leaders for it, so this tool has no "
    "figures from it. Say that you found the game but have no figures from it, never say "
    "that the game did not happen, and never give a figure from your own memory instead."
)
# THE second barrier behind the postseason branch, for the round where the model asks about
# a Super Bowl without passing playoff_round and this path answers instead. Unconditional
# in every sentence: the live defect narrated a week-18 game as the Super Bowl, and a
# caveat the model has to decide whether to apply is a caveat it drops (measured 3/3).
_REGULAR_SEASON_ONLY_CLAUSE = (
    " The game described here is a regular-season game, and it is not a playoff game, not "
    "a conference championship game and not the Super Bowl. Never report any figure below "
    "as a figure from a playoff game or from the Super Bowl. Say that this was a "
    "regular-season game when you report any figure from it, so the member can tell it "
    "apart from a playoff game."
)


# --------------------------------------------------------------------------- #
# The POSTSEASON half of the game tool, added 2026-08-21 after a live member asked who led
# the Super Bowl in rushing and got a week-18 game against a different opponent. The
# statements below carry the round rather than a week number: a bare "week 5" is voiceable
# and would be read as a regular-season week.
# --------------------------------------------------------------------------- #

# The postseason twin of _GAME_STATEMENT, and it names BOTH CLUBS itself rather than
# leaning on the game's name: ESPN's own name for a playoff game is a headline like "Super
# Bowl LX" or "NFC Wild Card Playoffs", which names no team at all.
_PLAYOFF_GAME_STATEMENT = (
    "Every figure below comes from ONE single NFL game: {game}, in which {teams} played "
    "each other. That game was played in the {round} of the {season} NFL season, which "
    "means it belongs to the {season} season even though it was played in the year after "
    "{season}. Name both of those teams and say the {season} season whenever you report "
    "any figure from this answer, so the member knows exactly which game you are talking "
    "about."
)

# THE anti-substitution notes. Each says plainly WHICH game could not be found, names no
# figure from any other game, and tells the model what to say — a miss that returns silence
# or a reachable game's figures is the defect these exist to close.
_TEAM_NOT_IN_ROUND_NOTE = (
    "The NFL team you asked about did not play in the {round} of the {season} NFL season, "
    "so this tool has no figures at all for that team in that round. The clubs that did "
    "play in that round were: {matchups}. Tell the member plainly that the team he asked "
    "about was not in that round, never report any figure from one of those other games as "
    "though it were his team's, and never give a figure from your own memory instead."
)
_WHICH_PLAYOFF_GAME_NOTE = (
    "The {round} of the {season} NFL season was more than one game, so this tool cannot "
    "tell which of them the member means. The clubs that played in that round were: "
    "{matchups}. Ask the member which of those games he means, report no figure at all "
    "until he answers, and never pick one of those games yourself."
)
_PLAYOFF_GAME_NOT_PLAYED_NOTE = (
    "{game}, in the {round} of the {season} NFL season, has not been played yet, so there "
    "are no figures from it at all. Tell the member plainly that the game has not been "
    "played yet, never describe how it went or who led it, and never give him a different "
    "game's figures instead."
)
_NO_PLAYOFF_LEADERS_NOTE = (
    "This tool did find the game the member asked about — {game}, in the {round} of the "
    "{season} NFL season — but ESPN publishes no game leaders for it, so this tool has no "
    "figures from it. Say that you found the game but have no figures from it, never say "
    "that the game did not happen, and never give a figure from your own memory or from a "
    "different game instead."
)
_PRO_BOWL_LEADERS_NOTE = (
    "The Pro Bowl is an exhibition game rather than a playoff round, and this tool has no "
    "figures from a Pro Bowl at all — it cannot tell you who played in one, who led one, "
    "or how one went. Tell the member plainly that you have no Pro Bowl data, never name a "
    "player from your own memory as having played in one, and never give him figures from "
    "a playoff game as though they were a Pro Bowl's. The games this tool does cover in "
    "the postseason are the wild card round, the divisional round, the conference "
    "championship games and the Super Bowl."
)
_NO_GAME_TO_LOOK_UP_NOTE = (
    "The member's question named no NFL team and no playoff round, so this tool has no "
    "game at all to look up. Ask the member which team's game he means, and never describe "
    "a game from your own memory instead."
)

# INSTRUCT first, CONSTRAIN second — measured twice on this branch, not stylistic: a
# disclaimer-only description suppressed the call 5/5. The starter constraint is why the
# ordering matters most here (D-4): it is a disclaimer about a DIFFERENT question from the
# one the opener instructs on, so it constrains the answer without suppressing the call.
# The season wording is copied from _STATS_TOOL_DESCRIPTION on purpose, so the model learns
# ONE rule rather than two (D-5), and the round vocabulary is _PLAYOFF_ROUND_ENUM, shared
# byte-for-byte with lookup_playoff_results for the same reason.
_GAME_LEADERS_TOOL_DESCRIPTION = (
    "Look up which players led ONE single NFL game in passing, rushing, receiving, sacks "
    "and tackles, and which team won that one game, in the regular season or in the "
    "playoffs. Call this tool every time the member asks how a team did in a game, how "
    "their last game went, who led a game in yards, catches, sacks or tackles, how a named "
    "team did in a given week, or who led a playoff game or a Super Bowl, because your own "
    "memory of any individual game is often a year or more out of date. Pass the "
    "playoff_round argument every time the member asks about a playoff game, a conference "
    "championship game or a Super Bowl, and it is one of wild card, divisional, conference "
    "championships or super bowl; leave the playoff_round argument out for a regular-season "
    "game. The team argument is that team's standard abbreviation, for example KC for the "
    "Kansas City Chiefs, LV for the Las Vegas Raiders, or CHI for the Chicago Bears. Pass "
    "the team argument whenever the member's question names a team, and leave it out only "
    "when he asks about a Super Bowl and names no team, because a season has just one "
    "Super Bowl and this tool finds it without a team. Pass the week argument ONLY "
    "when the member named a week number, and leave the week argument out when he says "
    "their last game or their most recent game, because this tool finds the most recent "
    "finished game by itself and tells you which week it was. Pass the season argument "
    "ONLY when the member named a specific year such as 2024. LEAVE THE SEASON ARGUMENT "
    "OUT for every other phrasing, including last year, last season and this season, "
    "because this tool already knows which season is the most recent one ESPN has and you "
    "do not. Never work out a year number for yourself from a phrase like last year. An "
    "NFL season is named for the year it STARTED in, so the Super Bowl played in February "
    "2026 belongs to the 2025 season. When this tool tells you it could not find the game "
    "the member asked about, say that plainly and never report a different game's figures "
    "as though they were that game's. When the member asks only which teams WON a whole "
    "round of the playoffs rather than who led one game, lookup_playoff_results is the "
    "tool for that question and this one is not. It carries neither team's score, so never "
    "state the score of the game and never say how many points either team scored. The "
    "player who led a game in passing is not necessarily that team's starting quarterback, "
    "because teams rest their starters and give backups snaps, so never call any player "
    "this tool names a starter. When the member asks who STARTS at a position, "
    "lookup_team_roster is the tool for that question and this one is not. When he asks "
    "what a player did across a whole season rather than in one game, "
    "lookup_player_season_stats is the tool for that question and this one is not."
)


# --------------------------------------------------------------------------- #
# The PLAYOFF tool. The other half of the calendar fix above: no tool covered postseason
# results, so every Super Bowl question was answered from ungrounded memory, and once the
# model knew what year it was that memory produced a confident falsehood rather than a
# hedge. Regular-season games stay with ``lookup_game_leaders``; the two descriptions
# route to each other.
# --------------------------------------------------------------------------- #


async def _lookup_playoff_results(
    season: int | None = None, playoff_round: str = ""
) -> object | None:
    """Look up who won one ROUND of one NFL season's playoffs.

    One cached hop on every branch. The week is never model-written: ``playoff_round`` is
    resolved to a LITERAL by :func:`~app.services.espn_extra.postseason_round_week`, which
    can only return 1, 2, 3 or 5, and the seam rejects anything else — so the Pro Bowl's
    week 4 is unreachable and can never be relayed as a playoff result. The season, when
    the member named one, passes an integer range check inside the seam before the format
    string runs; when he named none it is resolved from ESPN's own league root rather than
    invented, which is why the season argument is optional (a required argument the model
    cannot fill invites it to chain tools instead, measured 3/3 on the predecessor task).

    Every outcome is a dict — a NOTE on every miss, never bare ``None``, which becomes
    :data:`_NO_DATA_PAYLOAD` and sends the model back to the stale memory this tool exists
    to replace. No score is read on any path (D-2).
    """
    from app.services import espn_extra

    asked_round = playoff_round.strip() if isinstance(playoff_round, str) else ""
    if espn_extra.asked_for_the_pro_bowl(asked_round):
        return {"note": _PRO_BOWL_NOTE}
    week = (
        espn_extra.postseason_round_week(asked_round) if asked_round else espn_extra.SUPER_BOWL_WEEK
    )
    if week is None:
        return {"note": _UNKNOWN_ROUND_NOTE}

    asked_season = season if isinstance(season, int) and not isinstance(season, bool) else None
    if asked_season is None:
        current = espn_extra.league_season_year(await espn_extra.fetch_league())
        if current is None:
            return {"note": _NO_SEASON_TO_ASK_ABOUT_NOTE}
        asked_season, _checked = await _most_recently_finished_season(current)

    label = espn_extra.POSTSEASON_ROUND_LABELS[week]
    payload = await espn_extra.fetch_postseason_scoreboard(asked_season, week)
    facts = espn_extra.parse_postseason_round(payload) if payload is not None else None
    if facts is None:
        return {"note": _NO_POSTSEASON_RESULTS_NOTE.format(season=asked_season, round=label)}

    year = facts["season"] if isinstance(facts["season"], int) else asked_season
    if not facts["any_completed"]:
        # The ONE case where "has not happened yet" is correct — and it is said about this
        # one named season, never as the blanket hedge the live defect produced.
        return {"note": _POSTSEASON_NOT_PLAYED_NOTE.format(season=year, round=label)}

    statement = _PLAYOFF_RESULTS_STATEMENT.format(season=year, round=label, after=year + 1)
    for game in facts["games"]:
        statement += _playoff_game_clause(game)
    statement += _PLAYOFF_NO_SCORE_CLAUSE

    return {
        "season": year,
        "round": label,
        "games": facts["games"],
        "results_statement": statement,
        "caveat": espn_extra.POSTSEASON_CAVEAT,
    }


def _playoff_game_clause(game: dict) -> str:
    """The sentence naming who won ONE playoff game, or what is known instead. Pure.

    Phrased here and not in the parser, which never phrases: a dict field is readable but
    not voiceable, and the winner is the one fact a playoff question is asked for.
    """
    name = _game_phrase(game["game"]) if isinstance(game["game"], str) else "that game"
    if not game["completed"]:
        return _PLAYOFF_GAME_UNPLAYED_CLAUSE.format(game=name)
    winner = game["winner"]
    if not isinstance(winner, str):
        return _PLAYOFF_GAME_NO_WINNER_CLAUSE.format(game=name)
    teams = [str(team) for team in game["teams"]] if isinstance(game["teams"], list) else []
    beaten = [team for team in teams if team != winner]
    if len(beaten) != 1:
        return _PLAYOFF_GAME_WINNER_CLAUSE.format(game=name, winner=winner)
    return _PLAYOFF_GAME_BEAT_CLAUSE.format(game=name, winner=winner, loser=beaten[0])


def _game_phrase(name: str) -> str:
    """ESPN's own name for a game with the article a person would say. Pure.

    "the AFC Championship", but "Super Bowl LX" with no article at all, because a Super
    Bowl's name is already a proper noun and the model repeats this wording verbatim.
    """
    return name if name.lower().startswith("super bowl") else f"the {name}"


# D-3 of the predecessor, applied to a round: the season and the round are STATED, never
# implied, and the naming rule is repeated here because the payload is the last thing the
# model reads before it answers.
_PLAYOFF_RESULTS_STATEMENT = (
    "Every result below comes from the {round} of the {season} NFL season. The {season} "
    "season's playoffs were played in January and February {after} and they are over and "
    "finished. Say the {season} season whenever you report any of these results, because "
    "an NFL season is named for the year it started in and not for the year its playoffs "
    "were played in."
)
_PLAYOFF_GAME_BEAT_CLAUSE = " The {winner} beat the {loser} in {game}."
_PLAYOFF_GAME_WINNER_CLAUSE = " The {winner} won {game}."
_PLAYOFF_GAME_NO_WINNER_CLAUSE = (
    " {game} was played, but ESPN records no winner for it, so name no winner for that one game."
)
_PLAYOFF_GAME_UNPLAYED_CLAUSE = (
    " {game} has not been played yet, so say that about that one game and name no winner for it."
)
# D-2, unconditional rather than an "if": the score is never read out of the payload at
# all, and OPEN_OWNERSHIP_CLAUSE already forbids stating one.
_PLAYOFF_NO_SCORE_CLAUSE = (
    " The final score of every one of these games is left out of this answer on purpose. "
    "Never state the score of any of them, never say how many points either team scored, "
    "and never work a score out for yourself."
)

# Every miss is a concrete full sentence telling the model what to do next, returned in a
# dict body because a bare string fact gets voiced or swallowed (memory:
# qa-phrasing-inversion) and a bare ``None`` sends it back to its own stale memory.
_POSTSEASON_NOT_PLAYED_NOTE = (
    "The {season} NFL season's postseason has not been played yet, so ESPN has no result "
    "for the {round} of it at all. Tell the member plainly that the {round} of the "
    "{season} season has not been played yet, name no winner for it, and never give a "
    "result from your own memory instead. Say this about the {season} season only, "
    "because every NFL season before it has already been played in full."
)
_PRO_BOWL_NOTE = (
    "The Pro Bowl is an exhibition game and it is not a playoff round, so this tool never "
    "reports it as a playoff result. Tell the member plainly that the Pro Bowl is not part "
    "of the playoffs, and never name a Pro Bowl team as a playoff winner. The rounds this "
    "tool does cover are the wild card round, the divisional round, the conference "
    "championship games and the Super Bowl."
)
_UNKNOWN_ROUND_NOTE = (
    "This tool has no playoff round by that name. The only rounds it covers are the wild "
    "card round, the divisional round, the conference championship games and the Super "
    "Bowl. Ask the member which of those rounds he means, and report no result until he "
    "answers."
)
_NO_SEASON_TO_ASK_ABOUT_NOTE = (
    "The member named no season, and the lookup that would have told you which NFL season "
    "is the most recent finished one failed just now, so this tool has no results this "
    "time. Ask the member which season he means, and never work a year out for yourself."
)
# Measured 2026-08-21: a season outside ESPN's record (1960, 1966, 2030) answers 200 with
# no events and no season echo, which is the same dead end a failed fetch reaches. The
# wording covers both, because the instruction that matters is identical either way.
_NO_POSTSEASON_RESULTS_NOTE = (
    "This tool has no results at all for the {round} of the {season} NFL season, either "
    "because ESPN's record does not carry that season or because the lookup of it failed "
    "just now. Tell the member plainly that you could not look that season's playoff "
    "results up, never give a result from your own memory instead, and never tell him "
    "that those games have not been played."
)

# INSTRUCT first, CONSTRAIN second — measured three times on this branch, not stylistic.
# The season sentence is the longest one here for a reason: naming a season by the year its
# Super Bowl was played in is the exact mistake the live defect made twice, so the rule is
# stated with the years filled in rather than left for the model to apply.
_PLAYOFF_TOOL_DESCRIPTION = (
    "Look up which teams won one round of one NFL season's playoffs, including the Super "
    "Bowl. Call this tool every time the member asks who won a Super Bowl, who won a "
    "playoff game, who won a conference championship, which teams reached or won any "
    "round of the playoffs, or how a season ended, because your own memory of a Super "
    "Bowl result or a playoff result is often a year or more out of date and this tool "
    "reads ESPN's own record of it. The season argument is the four-digit year the NFL "
    "season is NAMED for, and an NFL season is named for the year it STARTED in and never "
    "for the year its playoffs were played in: a season's playoffs are played in January "
    "and February of the year AFTER the year the season is named for, so the Super Bowl "
    "played in February 2026 belongs to the 2025 season and you pass 2025 for it. Leave "
    "the season argument out whenever the member named no season at all, because this tool "
    "then answers about the most recent season that has finished and you do not have to "
    "work out which season that is. The playoff_round argument is one of wild card, "
    "divisional, conference championships or super bowl; leave it out and this tool "
    "answers about the Super Bowl, which is the round members ask about most. This tool "
    "reports which TEAMS WON a whole round and it carries no player figures at all, so "
    "when the member asks about a regular-season game, about how a team did in a given "
    "week, or about who LED any one game in yards, catches, sacks or tackles, a playoff "
    "game and a Super Bowl included, lookup_game_leaders is the tool for that question "
    "and this one is not. It "
    "carries neither team's score, so never state the score of any game it reports and "
    "never say how many points either team scored. The Pro Bowl is an exhibition game "
    "rather than a playoff round and this tool never reports it, so never call a Pro Bowl "
    "result a playoff result. When it says a season's postseason has not been played yet, "
    "tell the member that plainly about that one season, and never say that about a season "
    "it did give you results for."
)


# ONE round vocabulary across both tools that take a round, so the model learns one set of
# names rather than two. The enum is a second bound on a model-written value; either
# adapter still resolves anything else through espn_extra's own keyword table.
_PLAYOFF_ROUND_ENUM = ["wild card", "divisional", "conference championships", "super bowl"]


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
                        "player": {
                            "type": "string",
                            "description": "The player's name as the member wrote it.",
                        },
                        "team": {
                            "type": "string",
                            "description": (
                                "Optional team abbreviation, such as LAR, and ONLY when "
                                "the member's own question names a team. Leave it out "
                                "when his question names no team."
                            ),
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
                    "required": ["player"],
                },
            },
        },
        run=_lookup_player_season_stats,
    ),
    _Tool(
        name="lookup_player_current_team",
        spec={
            "type": "function",
            "function": {
                "name": "lookup_player_current_team",
                "description": _CURRENT_TEAM_TOOL_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "player": {
                            "type": "string",
                            "description": "The player's name as the member wrote it.",
                        },
                    },
                    "required": ["player"],
                },
            },
        },
        run=_lookup_player_current_team,
    ),
    _Tool(
        name="lookup_game_leaders",
        spec={
            "type": "function",
            "function": {
                "name": "lookup_game_leaders",
                "description": _GAME_LEADERS_TOOL_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "team": {
                            "type": "string",
                            "description": "The team's standard abbreviation, such as KC.",
                        },
                        "week": {
                            "type": "integer",
                            "description": (
                                "The week number, and ONLY when the member named one. "
                                "Leave it out for their last game or their most recent "
                                "game."
                            ),
                        },
                        "season": {
                            "type": "integer",
                            "description": (
                                "The four-digit year, and ONLY when the member named "
                                "one. Leave it out for last year, last season or this "
                                "season."
                            ),
                        },
                        "playoff_round": {
                            "type": "string",
                            "enum": _PLAYOFF_ROUND_ENUM,
                            "description": (
                                "Which playoff round the game was in, and ONLY for a "
                                "playoff game. Leave it out for a regular-season game."
                            ),
                        },
                    },
                    # NOTHING is required. ``team`` was required until 2026-08-21, and a
                    # Super Bowl question is exactly the case the member's own words cannot
                    # fill it from — a required argument the model cannot fill invites it
                    # to chain tools or invent a value (measured 3/3).
                    "required": [],
                },
            },
        },
        run=_lookup_game_leaders,
    ),
    _Tool(
        name="lookup_playoff_results",
        spec={
            "type": "function",
            "function": {
                "name": "lookup_playoff_results",
                "description": _PLAYOFF_TOOL_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "season": {
                            "type": "integer",
                            "description": (
                                "The four-digit year the season is NAMED for, and ONLY "
                                "when the member named a season. Leave it out for the "
                                "most recent season that has finished."
                            ),
                        },
                        "playoff_round": {
                            "type": "string",
                            "enum": _PLAYOFF_ROUND_ENUM,
                            "description": (
                                "Which playoff round. Leave it out for the Super Bowl."
                            ),
                        },
                    },
                    # Neither argument is required: the member's question supplies them
                    # only sometimes, and a required argument the model cannot fill
                    # invites it to chain tools (measured 3/3).
                    "required": [],
                },
            },
        },
        run=_lookup_playoff_results,
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
    ``compose_prompt(voice, OPEN_ROLE + the calendar facts, OPEN_GUARD)`` — the guard
    constants are never edited or relocated — runs the bounded tool loop
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

        role = f"{OPEN_ROLE} {await _calendar_facts()}"
        system_prompt = compose_prompt(voice, role, OPEN_GUARD)
        content = await _run_tool_loop(messages, system_prompt=system_prompt)
        if content is None:
            return None
        return _strip_markdown_structure(content) or None
    except Exception:
        # Best-effort by contract — a surprise raise degrades to the caller's
        # deterministic line and never escapes into the gateway loop.
        logger.warning("qa_open_answer_failed", exc_info=True)
        return None
