"""On-demand ESPN "extras" adapter — one shared fetch-and-cache shell + pure parsers.

The Path-B seam (outside-intelligence): when a league member asks the bot something the
app's own database does not own — an injury report, a headline, a team roster — the bot fetches
the public ESPN payload RIGHT THEN, caches it briefly in Redis, and parses it into a
deterministic fact structure. NO new DB tables, NO Celery-beat poller — freshness matters
most here, so on-demand is always fresh (design:
``.planning/notes/discord-query-bot-path-b-design.md``).

Design — impure shell / pure never-raising core (mirrors :mod:`app.scoreboard.espn`):

* IMPURE: ONE shell, :func:`_fetch_cached`, serves EVERY endpoint by delegating to
  :func:`app.services.http_cache.fetch_cached`, which owns the contract (cache-first, one
  GET, never raises, fail-open Redis). :func:`fetch_game_summary`, :func:`fetch_news`,
  :func:`fetch_team_roster`, :func:`fetch_athlete_stats`, :func:`fetch_athlete_search`,
  :func:`fetch_league`, :func:`fetch_team_schedule` and :func:`fetch_postseason_scoreboard`
  are thin delegations supplying their own URL, cache key, TTL and log label.
  :func:`fetch_injuries` delegates one step further, to :func:`fetch_game_summary`, so the
  injuries path and the game-leaders path share ONE Redis entry rather than fetching the
  same 635 KB payload twice (D-8).
* PURE: one parser per endpoint (:func:`parse_injuries`, :func:`parse_news`,
  :func:`parse_team_roster`, :func:`parse_athlete_stats`, :func:`parse_athlete_search`,
  :func:`parse_team_schedule`, :func:`parse_game_leaders`,
  :func:`parse_postseason_round`), plus :func:`find_roster_athletes` resolving a name
  against a raw roster payload and :func:`find_postseason_games` /
  :func:`postseason_games_for_team` resolving one playoff game against a raw scoreboard
  payload, :func:`league_season_year` / :func:`league_season_phase`
  reading the season being played and the phase it is in, and
  :func:`postseason_round_week` turning a round name into the one week number that may
  reach a URL, turning an already-parsed payload into facts. Defensive on EVERY field (isinstance
  guards, ``.get``, degrade to ``None``); never raises — this is what the offline tests
  exercise.

This module imports NO ``discord`` and lives on the Discord-free side: the qa.py brain
imports THIS seam for the HTTP+cache, staying itself HTTP-free.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote

import structlog

from app.config import settings
from app.seeds.teams import NFL_TEAMS
from app.services import http_cache

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants (mirror the scoreboard adapter's endpoint/UA/timeout conventions)
# ---------------------------------------------------------------------------

# The public, no-auth ESPN game ``summary`` endpoint (SAME host family as the
# scoreboard we already poll). One call carries BOTH teams' injuries (+ game news).
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={event_id}"

# One source of truth for the timeout — the shared shell owns the value.
DEFAULT_TIMEOUT = http_cache.DEFAULT_TIMEOUT

# Short Redis cache: freshness matters for injuries, but ~10 min cushions repeat asks
# so a flurry of questions on the same game is ONE upstream call.
INJURIES_CACHE_TTL_SECONDS = 600


def _cache_key(event_id: int | str) -> str:
    """The Redis key for one event's cached ``summary`` payload.

    ``int | str`` because :func:`fetch_game_summary` normalises the id to a digit string
    before it gets here; the f-string output is identical either way (D-8).
    """
    return f"qa:injuries:summary:{event_id}"


# The public, no-auth ESPN league ``news`` endpoint (SAME host family as the
# scoreboard/summary). Carries ONLY a fixed integer ``limit`` — never user text
# (T-ikf-03: the SSRF surface is a constant). The team filter is applied AFTER the
# fetch, client-side (the ``?team=`` param is unreliable — design note).
NEWS_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/news?limit={limit}"

# Fetch a wide-ish page so client-side team filtering still yields enough headlines.
NEWS_FETCH_LIMIT = 25

# Short Redis cache: headlines move, but ~10 min cushions a flurry of asks into ONE
# upstream call (same rationale as injuries).
NEWS_CACHE_TTL_SECONDS = 600


def _news_cache_key(limit: int) -> str:
    """The Redis key for the cached league-news page.

    The league page is NOT team-scoped — the team filter is applied AFTER the fetch,
    so repeat asks for DIFFERENT teams reuse ONE cached page keyed only by ``limit``.
    """
    return f"qa:news:league:{limit}"


# The public, no-auth ESPN team ``roster`` endpoint (SAME host family). ``{team}`` is
# the ONLY model-influenced value anywhere in this module, which is why the allowlist
# below runs before the format string ever does (T-oym-01).
ROSTER_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team}/roster"

# A roster moves on a scale of days, not minutes, so it caches an order of magnitude
# longer than the ten-minute injuries/news endpoints.
ROSTER_CACHE_TTL_SECONDS = 3600

# The largest measured position group is 13 (CHI cornerbacks, 2026-08-20), so this is a
# defensive ceiling on the tool loop's token budget, not a routine truncation.
ROSTER_MAX_PLAYERS = 20

# DERIVED from the canonical seed table, never retyped: a drifted copy of the list that
# guards a URL is the failure worth preventing. Importing that seeder is documented
# side-effect-free (``app.seeds.historical_games`` imports it the same way).
NFL_TEAM_ABBRS = frozenset(abbr for _espn_id, abbr, _display_name in NFL_TEAMS)


def _roster_cache_key(team_abbr: str) -> str:
    """The Redis key for one team's cached ``roster`` payload."""
    return f"qa:roster:team:{team_abbr}"


# The public, no-auth ESPN per-athlete career-stats endpoint. A DIFFERENT subdomain from
# every other URL here, which is worth stating precisely because it changes nothing: it
# is still an ESPN edge, so it still sends NO custom User-Agent (see :func:`_fetch_cached`).
ATHLETE_STATS_URL = (
    "https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{athlete_id}/stats"
)

# A season total is settled until the player's next game, so it caches an order of
# magnitude longer than the hour-long roster — the same step the roster takes over the
# ten-minute injuries/news feeds.
ATHLETE_STATS_CACHE_TTL_SECONDS = 21600

# ``{athlete_id}`` is the ONLY value ever formatted into that URL, and ``fullmatch``
# against this runs BEFORE the format string does (T-s5y-01).
_ATHLETE_ID_RE = re.compile(r"[0-9]{1,12}")


def _athlete_stats_cache_key(athlete_id: str) -> str:
    """The Redis key for one athlete's cached career-stats payload."""
    return f"qa:stats:athlete:{athlete_id}"


# The largest measured category count is 6 (Saquon Barkley, 2026-08-20), so this is a
# defensive ceiling on the tool loop's token budget, not a routine truncation.
STATS_MAX_CATEGORIES = 8

# D-6: the values that make a whole category zero-noise the model could misread.
_ZEROISH_STAT_VALUES = frozenset({"0", "0.0", "-", ""})

# Games Played is excluded from the all-zero test, under any of the three spellings the
# payload can key it by.
_GAMES_PLAYED_KEYS = frozenset({"Games Played", "gamesPlayed", "GP"})

# ESPN puts the suffix INSIDE ``lastName`` (measured: 'McClendon Jr.'), so a surname
# comparison has to drop it or "McClendon" never matches.
_NAME_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})

# The public, no-auth ESPN player search, used ONLY after the roster hop misses: roster
# resolution anchors on the CURRENT roster while a stats question is about a PAST season,
# and those disagree for every player who changed teams (measured: Isiah Pacheco asked
# about on KC, rostered on DET).
ATHLETE_SEARCH_URL = (
    "https://site.web.api.espn.com/apis/search/v2?query={query}&limit={limit}&type=player"
)

# Six is enough to see the ambiguity: the measured "josh allen" page needs four rows
# before the NFL matches are all in hand.
ATHLETE_SEARCH_LIMIT = 6

# Which team a player is on moves on the scale of days, so this matches the roster's hour
# rather than the stats table's six.
ATHLETE_SEARCH_CACHE_TTL_SECONDS = 3600

# The search query is the ONLY model-influenced text this module ever puts in a URL.
# ``quote(safe="")`` percent-encodes EVERY reserved character, ``&`` and ``?`` and ``#``
# and ``/`` included, so the value cannot close its own query parameter, add another, or
# reach the path — and this cap bounds it before that ever runs (T-s5y-08).
SEARCH_QUERY_MAX_CHARS = 40

# The athlete id lives in ``uid`` as ``s:20~l:28~a:4361529``; the top-level ``id`` is a
# GUID and is never usable against the stats endpoint (issue #183, Route B).
_SEARCH_UID_ATHLETE_RE = re.compile(r"a:([0-9]{1,12})$")

# DERIVED from the canonical seed table, never retyped (same reason as NFL_TEAM_ABBRS).
# The search is cross-sport and cross-level — the measured "josh allen" page returns a
# Luton Town soccer player and two college programmes — so ``subtitle`` is matched against
# the 32 club display names to keep only NFL players.
NFL_TEAM_DISPLAY_NAMES_UPPER = frozenset(
    display_name.upper() for _espn_id, _abbr, display_name in NFL_TEAMS
)


def _athlete_search_cache_key(query: str) -> str:
    """The Redis key for one search query's cached result page."""
    return f"qa:search:athlete:{query}"


# The public, no-auth ESPN league root — the ONE source of the season being played, on
# every path (the stats payload carries no year, and D-1 forbids reading the app's own
# ``Week`` table from the open path). This URL is a CONSTANT: unlike the roster's team and
# the search's query, no segment of it is model-influenced, so there is nothing to
# allowlist, encode or cap before it is requested. A third ESPN subdomain, which changes
# nothing — still an ESPN edge, so still no custom User-Agent.
LEAGUE_URL = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"

# The longest TTL in this module by an order of magnitude, because the value behind it
# moves once a YEAR: six hours still refreshes it ~1,400 times per change, while capping
# what a stale entry can cost at the one moment it can cost anything — the August rollover,
# where a longer day-scale TTL would name last season as the one being played.
LEAGUE_CACHE_TTL_SECONDS = 21600

# A constant, not a builder: the league root takes no parameter, so its key takes none.
_LEAGUE_CACHE_KEY = "qa:league:nfl"


# The sentence the model is most likely to voice, so it is concrete and complete rather
# than a terse fragment (memory: qa-phrasing-inversion). The second clause exists because
# D-6 deliberately keeps a category ESPN publishes a rate for on zero attempts.
STATS_CAVEAT = (
    "These are ESPN's own published figures for the one NFL season this answer names, so "
    "state that season's year when you report any of them and never call them this "
    "year's or last year's figures. When the answer says that season is still being "
    "played, these are only the totals so far in it, so never call them a final or an "
    "official season total. When a category shows the player had no attempts in "
    "it, do not report that category's rate statistics such as a passer rating or a "
    "yards per attempt average, because a rate worked out on no attempts is meaningless."
)


# The public, no-auth ESPN team ``schedule`` endpoint (SAME host family as the roster).
# Measured 2026-08-21: the path accepts the standard ABBREVIATION, so the roster's
# 32-abbreviation allowlist guards this URL too and no numeric team-id table is needed.
REGULAR_SEASON_TYPE = 2

# ``seasontype`` is PINNED on BOTH forms: without it the endpoint serves the PRESEASON,
# whose ``week.number`` disagrees with its own week text ("Preseason Week 1" carries
# number 2), and a week number that lies is worse than no week number at all.
TEAM_SCHEDULE_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team}/schedule"
    f"?season={{season}}&seasontype={REGULAR_SEASON_TYPE}"
)

# The season-less form. Measured to return the CURRENT season's regular schedule AND to
# echo its year in ``requestedSeason`` — which is what makes the season year cost no extra
# hop and never be a guess (D-5).
CURRENT_TEAM_SCHEDULE_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team}/schedule"
    f"?seasontype={REGULAR_SEASON_TYPE}"
)

# D-9: a schedule's ``completed`` flags move on game days, the same freshness argument the
# ten-minute injuries feed carries. One TTL for every season, so there is no branch that
# could go stale the moment the model asks for the current year explicitly.
SCHEDULE_CACHE_TTL_SECONDS = 600

# A full NFL regular season is 17 games plus a bye, so this is a defensive ceiling on the
# tool loop's token budget rather than a routine truncation.
SCHEDULE_MAX_EVENTS = 22

# The bounds on ``season`` before it is formatted into a URL — the ``_ATHLETE_ID_RE``
# discipline (T-s5y-01) applied to an integer instead of a string.
_SCHEDULE_SEASON_MIN = 1920
_SCHEDULE_SEASON_MAX = 2100

# The event id comes from a schedule payload rather than the model, and is still
# ``fullmatch``ed before the summary URL is formatted (T-f0s-03).
_EVENT_ID_RE = re.compile(r"[0-9]{1,12}")


def _schedule_cache_key(team_abbr: str, season: int | None) -> str:
    """The Redis key for one team-season's cached ``schedule`` payload.

    A season-less request keys on the literal ``current`` rather than on a year worked
    out here — the year it will turn out to be is ESPN's to echo, not ours to predict.
    """
    return f"qa:schedule:team:{team_abbr}:{season if season is not None else 'current'}"


# The sentences the model is most likely to voice, so each is concrete and complete rather
# than a terse fragment (memory: qa-phrasing-inversion). The starter sentence is stated
# UNCONDITIONALLY (D-4): KC's measured week-18 passing leader was Shane Buechele, a backup,
# because teams rest starters in the final week — and a caveat the model has to decide
# whether to apply is a caveat it drops (measured 3/3 on this branch).
GAME_LEADERS_CAVEAT = (
    "Each list of leaders below belongs only to the club it is named under, so never say "
    "that a player led the other club in anything and never move a player from one club's "
    "list to the other's. Every figure here is from that ONE single game and none of them "
    "is any player's total for a season, so never call any of these figures a season "
    "total. The player who led a game in passing is not necessarily that team's starting "
    "quarterback, because teams rest their starters and give backups snaps, so never call "
    "any player named here a starter."
)


# The sentences the model is most likely to voice, so each is concrete and complete rather
# than a terse fragment (memory: qa-phrasing-inversion). The last one is unconditional
# because a caveat the model has to decide whether to apply is a caveat it drops.
TEAM_SCHEDULE_CAVEAT = (
    "Every game listed here is a regular-season game, and this list carries no playoff "
    "game at all, so never call any game on it a playoff game and never say that it "
    "shows how a team did in the playoffs. It carries no score and no result for any "
    "game on it, so never say who won any of these games and never say how many points "
    "either team scored. A game this list says has not been played yet has not been "
    "played, so never report it as a game that has already happened. Every kick-off time "
    "here is given in UTC, so never state one of them as a local time."
)


# The public, no-auth ESPN league-leaders endpoint. ``seasontype`` is baked in and spelled
# all-lowercase: a season without one answers the POSTSEASON, and the camel-cased spelling
# is ignored and falls back the same way. ``{sort}`` is a LEADER_SORTS literal (T-jbh-01).
LEADERS_URL = (
    "https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/statistics/byathlete"
    f"?limit={{limit}}&sort={{sort}}&seasontype={REGULAR_SEASON_TYPE}"
)

LEADERS_LIMIT = 5

# Measured 2026-08-21: every stat in LEADER_SORTS sits within the first eight ``names`` of
# its own category block, so this cap costs no allowlisted figure and keeps five leaders
# of the widest block (receiving, twelve names) inside the payload ceiling.
LEADERS_MAX_FACTS = 8

# Leaders move on game days, the same freshness argument the schedule carries.
LEADERS_CACHE_TTL_SECONDS = 600

# The CODE-OWNED allowlist: a model-facing name -> the (category, stat) the sort is BUILT
# from. Every pair verified live 2026-08-21; an unknown one answers HTTP 400. A SORT spells
# it ``defensiveInterceptions`` and the payload ``defensiveinterceptions``, hence the fold.
LEADER_SORTS: dict[str, tuple[str, str]] = {
    "passing yards": ("passing", "passingYards"),
    "passing touchdowns": ("passing", "passingTouchdowns"),
    "rushing yards": ("rushing", "rushingYards"),
    "rushing touchdowns": ("rushing", "rushingTouchdowns"),
    "receiving yards": ("receiving", "receivingYards"),
    "receptions": ("receiving", "receptions"),
    "receiving touchdowns": ("receiving", "receivingTouchdowns"),
    "sacks": ("defensive", "sacks"),
    "tackles": ("defensive", "totalTackles"),
    "interceptions": ("defensiveInterceptions", "interceptions"),
}

# Substring -> allowlist key, tried IN THIS ORDER so "touchdown passes" never falls
# through to the yards entry and "receptions" never reaches the receiving-yards one. The
# shape :data:`_ROUND_KEYWORD_WEEKS` already holds: model-written text selects a LITERAL.
_LEADER_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("passing touchdown", "passing touchdowns"),
    ("passing td", "passing touchdowns"),
    ("touchdown pass", "passing touchdowns"),
    ("rushing touchdown", "rushing touchdowns"),
    ("rushing td", "rushing touchdowns"),
    ("receiving touchdown", "receiving touchdowns"),
    ("receiving td", "receiving touchdowns"),
    ("reception", "receptions"),
    ("catch", "receptions"),
    ("passing", "passing yards"),
    ("passer", "passing yards"),
    ("pass", "passing yards"),
    ("rushing", "rushing yards"),
    ("rusher", "rushing yards"),
    ("rush", "rushing yards"),
    ("receiving", "receiving yards"),
    ("receiver", "receiving yards"),
    ("sack", "sacks"),
    ("tackle", "tackles"),
    ("interception", "interceptions"),
)

# The sentences the model is most likely to voice, so each is concrete and complete rather
# than a terse fragment (memory: qa-phrasing-inversion). The interceptions sentence is
# unconditional because the same word names two different statistics.
LEAGUE_LEADERS_CAVEAT = (
    "Every figure here is from the REGULAR season only and no playoff figure is in it, so "
    "never say that any of it includes a playoff game. This is a ranking of individual "
    "PLAYERS and it is not a team standing, so never report it as one. The order below is "
    "ESPN's own order, so never reorder it and never add a player to it from your own "
    "memory. Where this answer reports interceptions, those are interceptions CAUGHT by "
    "defenders and they are never interceptions thrown by a quarterback."
)


def _leaders_cache_key(category: str, season: int | None) -> str:
    """The Redis key for one category-and-season's cached leaders page.

    Keyed on the RESOLVED allowlist name, so every near-miss phrasing of one category
    shares one entry rather than opening a new one per wording.
    """
    return f"qa:leaders:{category}:{season if season is not None else 'current'}"


# The public, no-auth ESPN per-athlete GAME LOG — the same subdomain as the career-stats
# table above, so still an ESPN edge and still no custom User-Agent.
GAMELOG_URL = (
    "https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{athlete_id}/gamelog"
)

# A settled per-game line, so this matches the season table's six hours rather than the
# schedule's ten minutes.
GAMELOG_CACHE_TTL_SECONDS = 21600

# MEASURED 2026-08-21 on a real quarterback log: a sixteen-wide row is ~580 bytes a game,
# so six games alone was 3,492 against the 3,400-byte payload ceiling. The fact cap is the
# full row width because a lower one drops every rushing figure (index 12 of 16).
GAME_LOG_MAX_EVENTS = 3
GAME_LOG_MAX_FACTS = 16

# The ``categories[].splitType`` of each block. A member asking about "week 2" means the
# REGULAR season's week 2 and never a divisional playoff, so a week number selects only
# out of the regular-season block.
_GAMELOG_REGULAR_SPLIT = "2"

# ``@`` and ``vs`` are terse tokens, and a terse token handed to the model comes back out
# of its mouth verbatim (memory: qa-phrasing-inversion).
_GAMELOG_VENUES = {"@": "away", "vs": "at home"}

# A four-digit year at the START of a season-type name ("2024 Regular Season") — the
# fallback season source, because this endpoint's ``requestedSeason`` is null (M-1).
_GAMELOG_SEASON_RE = re.compile(r"^([0-9]{4})")

# The sentences the model is most likely to voice, so each is concrete and complete
# rather than a terse fragment (memory: qa-phrasing-inversion). The first one is the
# whole point of a game log: a single game's figure read as a season total is the defect.
GAME_LOG_CAVEAT = (
    "Every figure here belongs to the ONE game it is listed under and to no other game, "
    "so never add these figures together and never call any of them a season total. This "
    "answer carries no score for any of these games, so never state a score and never "
    "say how many points either team scored. A game listed under a postseason is a "
    "playoff game, so never call one of those a regular-season game and never count it "
    "as a week of the regular season."
)


def _gamelog_cache_key(athlete_id: str, season: int | None) -> str:
    """The Redis key for one athlete-season's cached game log.

    A season-less request keys on the literal ``current`` rather than on a year worked
    out here — the year it will turn out to be is ESPN's to answer, not ours to predict.
    """
    return f"qa:gamelog:athlete:{athlete_id}:{season if season is not None else 'current'}"


# The public, no-auth ESPN core-API standings. The season TYPE and the GROUP are module
# constants rather than parameters: M-3 measured group 9 as all 32 clubs in ONE fetch
# where group 8 is one conference, so no caller ever chooses either.
LEAGUE_STANDINGS_GROUP = 9

STANDINGS_URL = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season}"
    f"/types/{REGULAR_SEASON_TYPE}/groups/{LEAGUE_STANDINGS_GROUP}/standings/0"
)

# Records move on game days, the same freshness argument the schedule carries.
STANDINGS_CACHE_TTL_SECONDS = 600

# The trailing team id of a ``$ref`` path. The ``$ref`` is a URL supplied by a remote
# payload and it is NEVER fetched, joined or passed to httpx — only regexed (T-jbh-05).
_TEAM_REF_ID_RE = re.compile(r"/teams/([0-9]{1,4})(?:[?/]|$)")

# The season the standings payload says it is, out of its own ``$ref`` — the same
# echo-before-relay discipline ``requestedSeason`` carries on the schedule.
_STANDINGS_SEASON_RE = re.compile(r"/seasons/([0-9]{4})/")

# DERIVED from the canonical seed table, never retyped (same reason as NFL_TEAM_ABBRS).
NFL_TEAM_BY_ID: dict[str, tuple[str, str]] = {
    str(espn_id): (abbr, display_name) for espn_id, abbr, display_name in NFL_TEAMS
}

# The record TYPES relayed, mapped to the words a person would say. A camelCase or terse
# token handed to the model comes back out of its mouth verbatim (memory:
# qa-phrasing-inversion), and everything outside this table is never read at all.
_TEAM_RECORD_LABELS: dict[str, str] = {
    "total": "overall record",
    "home": "record at home",
    "road": "record on the road",
    "vsdiv": "record against teams in their own division",
}

# The sentences the model is most likely to voice, so each is concrete and complete
# (memory: qa-phrasing-inversion). The second answers the guard collision: a win-loss
# record is not the standings POSITION that OPEN_OWNERSHIP_CLAUSE bans.
TEAM_RECORD_CAVEAT = (
    "Every figure here is ESPN's own win-loss record for the one NFL season this answer "
    "names, so say that season's year when you report any of them. A win-loss record is "
    "not a standings position, it is not a league table place and it is not a game "
    "score, so none of the things you are told never to state applies to it. This is not "
    "this pick'em league's own standings and it is not any league member's standing, so "
    "never report it as either of those. It carries no score at all, so never say how "
    "many points any team scored. Where this answer says a season has not begun, never "
    "report a record for that season."
)


def _standings_cache_key(season: int) -> str:
    """The Redis key for one season's cached whole-league standings payload."""
    return f"qa:standings:{season}"


# The public, no-auth ESPN scoreboard, PINNED to the postseason (SAME host family as the
# schedule). Measured 2026-08-21 against the whole 2025 bracket: it resolves every playoff
# round, and only the postseason, which is what keeps it from competing with the
# regular-season schedule endpoint above.
POSTSEASON_SEASON_TYPE = 3

POSTSEASON_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    f"?dates={{season}}&seasontype={POSTSEASON_SEASON_TYPE}&week={{week}}"
)

# A bracket's ``winner`` flags move on game days in January and February, the same
# freshness argument the ten-minute schedule feed carries. One TTL for every season, so no
# branch can go stale in the one month it matters.
POSTSEASON_CACHE_TTL_SECONDS = 600

# The postseason week each PLAYOFF round is played in, and the human name of each — both
# measured 2026-08-21 (week 1 six games, week 2 four, week 3 two, week 5 one). Week 4 is
# ABSENT on purpose: it is the Pro Bowl, an exhibition game, so it is not a playoff round
# and no round name may ever select it (:data:`PRO_BOWL_WEEK`).
POSTSEASON_ROUND_LABELS: dict[int, str] = {
    1: "wild card round",
    2: "divisional round",
    3: "conference championship games",
    5: "Super Bowl",
}

PRO_BOWL_WEEK = 4
SUPER_BOWL_WEEK = 5

# The ONLY week numbers that may ever be formatted into POSTSEASON_SCOREBOARD_URL.
POSTSEASON_WEEKS = frozenset(POSTSEASON_ROUND_LABELS)

# Substring -> week, tried in this order, so "super bowl LX" never falls through to the
# "bowl"-less keywords and "conference championship" resolves the same as "championship".
_ROUND_KEYWORD_WEEKS: tuple[tuple[str, int], ...] = (
    ("superbowl", 5),
    ("wildcard", 1),
    ("division", 2),
    ("conference", 3),
    ("championship", 3),
)

# The sentences the model is most likely to voice, so each is concrete and complete rather
# than a terse fragment (memory: qa-phrasing-inversion). The first one is the whole point
# of this endpoint: the live 2026-08-21 defect was the model calling a finished season's
# Super Bowl a game that had not happened yet.
POSTSEASON_CAVEAT = (
    "Every result here is ESPN's own record of a playoff game that has already been "
    "played, so report it as a settled fact and never say that it has not happened yet. "
    "The final score of every one of these games is left out of this answer on purpose, "
    "so never state a score and never say how many points either team scored. The Pro "
    "Bowl is an exhibition game rather than a playoff round and is never reported here, "
    "so never call a Pro Bowl result a playoff result."
)


def _postseason_cache_key(season: int, week: int) -> str:
    """The Redis key for one season-and-round's cached postseason scoreboard."""
    return f"qa:postseason:{season}:{week}"


# ---------------------------------------------------------------------------
# Pure parsing (no network — unit-tested offline)
# ---------------------------------------------------------------------------


def _first_str(*values: Any) -> str | None:
    """Return the first non-empty string among ``values``, else ``None``."""
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _parse_one_injury(injury: Any) -> dict[str, str | None] | None:
    """Normalize one ``injuries[].injuries[]`` entry into a per-player fact dict.

    Defensive on every field (``isinstance`` guards + ``.get``); degrades a missing
    field to ``None`` and never raises. Returns ``None`` for an unusable (non-dict)
    entry so the caller can skip it. Reads status defensively from EITHER the
    top-level ``status`` OR ``type.name`` (whichever the unofficial shape carries),
    the athlete display name / position abbreviation, the body part + return date
    from ``details``, and the injury-level ``date`` as the "as-of" freshness stamp.
    """
    if not isinstance(injury, dict):
        return None

    athlete = injury.get("athlete")
    athlete = athlete if isinstance(athlete, dict) else {}
    position = athlete.get("position")
    position = position if isinstance(position, dict) else {}

    type_obj = injury.get("type")
    type_obj = type_obj if isinstance(type_obj, dict) else {}

    details = injury.get("details")
    details = details if isinstance(details, dict) else {}

    return {
        "status": _first_str(injury.get("status"), type_obj.get("name")),
        "display_name": _first_str(athlete.get("displayName")),
        "position": _first_str(position.get("abbreviation")),
        "body_part": _first_str(details.get("type")),
        "return_date": _first_str(details.get("returnDate")),
        "date": _first_str(injury.get("date")),
    }


def parse_injuries(payload: Any, team_abbr: str) -> list[dict[str, str | None]] | None:
    """Extract ONLY ``team_abbr``'s per-player injury facts from a ``summary`` payload.

    Pure and never-raising (mirrors ``normalize_event``/``normalize_odds``):

    * Returns a list of per-player fact dicts (status / display_name / position /
      body_part / return_date / as-of ``date``, each ``None`` when absent) for the
      matched team's injury block.
    * Returns ``[]`` when the team's block is present but lists no injuries.
    * Returns ``None`` when the top-level shape is unusable (non-dict payload, or
      ``injuries`` is not a list) OR the asked team's block is not present — a
      distinct signal so the caller degrades to "couldn't pull the report" rather
      than falsely announcing "no injuries" for a team it could not locate.

    One ``summary`` call carries BOTH teams, so this filters to the block whose
    ``team.abbreviation`` (upper-cased) equals ``team_abbr`` (upper-cased).
    """
    if not isinstance(payload, dict):
        return None
    blocks = payload.get("injuries")
    if not isinstance(blocks, list):
        return None

    needle = team_abbr.strip().upper()
    for block in blocks:
        if not isinstance(block, dict):
            continue
        team = block.get("team")
        team = team if isinstance(team, dict) else {}
        abbr = team.get("abbreviation")
        if not isinstance(abbr, str) or abbr.strip().upper() != needle:
            continue
        # Matched the asked team's block. Parse its (possibly empty) injury list.
        entries = block.get("injuries")
        entries = entries if isinstance(entries, list) else []
        players: list[dict[str, str | None]] = []
        for entry in entries:
            parsed = _parse_one_injury(entry)
            if parsed is not None:
                players.append(parsed)
        return players

    # The asked team's block was not present — degrade (never invent "no injuries").
    return None


def _article_teams(article: dict) -> list[str]:
    """Collect the upper-cased team descriptors an article is tagged with.

    From ``article["categories"]`` (a list), per category dict, gather the non-empty
    upper-cased strings among the category's ``team`` sub-dict (``abbreviation`` /
    ``displayName`` / ``description``) AND the category's own ``description``. These
    are matched against the ``(abbr, name)`` team filter — the ``?team=`` query param
    is UNRELIABLE and intentionally not used. Defensive: a non-list ``categories`` or
    a non-dict entry contributes nothing; never raises.
    """
    categories = article.get("categories")
    if not isinstance(categories, list):
        return []
    descriptors: list[str] = []
    for category in categories:
        if not isinstance(category, dict):
            continue
        team = category.get("team")
        team = team if isinstance(team, dict) else {}
        for value in (
            team.get("abbreviation"),
            team.get("displayName"),
            team.get("description"),
            category.get("description"),
        ):
            token = _first_str(value)
            if token is not None:
                descriptors.append(token.upper())
    return descriptors


def _parse_one_article(article: Any) -> dict | None:
    """Normalize one ``articles[]`` entry into a verbatim headline fact dict.

    Defensive on every field (``isinstance`` guards + ``.get``); degrades a missing
    field to ``None`` and never raises. Returns ``None`` for an unusable (non-dict)
    entry OR one with no headline (NEVER fabricates a headline). The ``headline`` is
    the EXACT ``_first_str`` result, unmodified (verbatim relay is the whole point).
    Captures ``description``, ``published`` (from ``published`` then ``lastModified``
    as the "as-of" stamp), the first usable ``link`` href, and the article's team
    ``teams`` descriptors for client-side filtering.
    """
    if not isinstance(article, dict):
        return None

    headline = _first_str(article.get("headline"))
    if headline is None:
        return None  # never fabricate a headline

    # First usable href for the article. ESPN's real shape is ``links`` as a DICT
    # keyed by surface — ``links.web.href`` (preferred), then ``links.mobile.href``.
    # Also tolerate a ``links`` LIST of {href} dicts and a singular ``link`` {href}
    # (defensive across the unofficial schema).
    link: str | None = None
    links = article.get("links")
    if isinstance(links, dict):
        for key in ("web", "mobile"):
            sub = links.get(key)
            if isinstance(sub, dict):
                href = _first_str(sub.get("href"))
                if href is not None:
                    link = href
                    break
    elif isinstance(links, list):
        for entry in links:
            if isinstance(entry, dict):
                href = _first_str(entry.get("href"))
                if href is not None:
                    link = href
                    break
    if link is None:
        link_obj = article.get("link")
        if isinstance(link_obj, dict):
            link = _first_str(link_obj.get("href"))

    return {
        "headline": headline,
        "description": _first_str(article.get("description")),
        "published": _first_str(article.get("published"), article.get("lastModified")),
        "link": link,
        "teams": _article_teams(article),
    }


def _team_filter_matches(descriptors: list[str], team_filter: tuple[str, str]) -> bool:
    """Whether an article's team ``descriptors`` match the ``(abbr, name)`` filter.

    A descriptor ``d`` matches when ``abbr == d`` OR ``name == d`` OR ``name in d``
    (both filter values pre-upper-cased by the caller) — so the canonical
    "KANSAS CITY CHIEFS" matches an ESPN category description regardless of exact shape.
    """
    abbr_upper, name_upper = team_filter
    for d in descriptors:
        if abbr_upper == d or name_upper == d or name_upper in d:
            return True
    return False


def parse_news(
    payload: Any, *, team_filter: tuple[str, str] | None = None, limit: int
) -> list[dict] | None:
    """Extract the top verbatim headline facts from a league ``news`` payload.

    Pure and never-raising (mirrors :func:`parse_injuries`):

    * Returns ``None`` when the top-level shape is unusable (non-dict payload, or
      ``articles`` is not a list) — the distinct failure signal.
    * Otherwise parses each ``articles[]`` entry via :func:`_parse_one_article`,
      skipping the ones that return ``None`` (non-dict / headline-less — never
      fabricated). When ``team_filter`` is given as ``(abbr_upper, name_upper)``, keeps
      ONLY articles whose captured ``teams`` descriptors match (client-side; the
      ``?team=`` param is never used).
    * Returns the first ``limit`` surviving articles (top-first in payload order), or
      ``[]`` when none survive (a VALID empty answer — distinct from the ``None``
      failure signal).
    """
    if not isinstance(payload, dict):
        return None
    entries = payload.get("articles")
    if not isinstance(entries, list):
        return None

    articles: list[dict] = []
    for entry in entries:
        parsed = _parse_one_article(entry)
        if parsed is None:
            continue
        if team_filter is not None and not _team_filter_matches(parsed["teams"], team_filter):
            continue
        articles.append(parsed)
        if len(articles) >= limit:
            break
    return articles


# Generic query/news words that carry no subject signal — dropped before matching so
# "recent news"/"latest update" narrow to nothing (i.e. no subject filter is applied).
_SUBJECT_STOPWORDS = frozenset(
    {
        "news",
        "latest",
        "recent",
        "update",
        "updates",
        "report",
        "reports",
        "story",
        "stories",
        "headline",
        "headlines",
        "return",
        "returns",
        "returning",
        "back",
        "season",
        "seasons",
        "game",
        "games",
        "week",
        "weeks",
        "about",
        "this",
        "that",
        "the",
        "any",
        "out",
        "for",
        "from",
        "what",
        "whats",
        "will",
        "nfl",
        "football",
        "team",
        "teams",
        "roster",
        "2024",
        "2025",
        "2026",
        "2027",
        "2028",
    }
)


def _subject_tokens(subject: str) -> list[str]:
    """Meaningful lowercased tokens from a classifier ``subject`` (>=3 chars, non-stop).

    Pure. Splits on non-alphanumerics, lowercases, drops short and generic-news words.
    An all-generic subject (e.g. "recent news") yields ``[]`` -> the caller applies NO
    narrowing (returns the team/league feed unchanged).
    """
    import re as _re

    raw = _re.split(r"[^a-z0-9]+", subject.lower()) if isinstance(subject, str) else []
    return [t for t in raw if len(t) >= 3 and t not in _SUBJECT_STOPWORDS]


def filter_news_by_subject(articles: list[dict], subject: str | None) -> list[dict] | None:
    """Narrow parsed news ``articles`` to those matching a specific ``subject``.

    Pure and never-raising. Returns:

    * ``None`` when there is nothing to narrow by (no subject, or an all-generic subject
      like "recent news") — the caller keeps the full team/league feed.
    * otherwise the subset of articles whose text (headline + description + the captured
      team/athlete descriptors, e.g. "Patrick Mahomes" from an ``athlete`` category)
      contains EVERY meaningful subject token. Possibly ``[]`` (no article is about that
      subject) — the caller then FALLS BACK to the full feed with a note, never empty.
    """
    if not subject:
        return None
    tokens = _subject_tokens(subject)
    if not tokens:
        return None
    out: list[dict] = []
    for article in articles:
        parts = [
            article.get("headline") or "",
            article.get("description") or "",
            " ".join(article.get("teams") or []),
        ]
        haystack = " ".join(parts).lower()
        if all(token in haystack for token in tokens):
            out.append(article)
    return out


# ESPN labels its roster groups in camelCase. A camelCase token handed to the model
# comes back out of the model's mouth verbatim (memory: qa-phrasing-inversion), so each
# one is mapped to the words a person would say; an unknown token falls back to the raw.
_ROSTER_GROUP_LABELS = {
    "offense": "offense",
    "defense": "defense",
    "specialTeam": "special teams",
    "injuredReserveOrOut": "injured reserve or out",
    "suspended": "suspended",
    "practiceSquad": "practice squad",
}

# The sentence the model is most likely to voice, so it is concrete and complete rather
# than a terse fragment (memory: qa-phrasing-inversion). This is the SECOND barrier
# against an invented starter; the tool description in ``qa_open`` is the first.
ROSTER_CAVEAT = (
    "This roster listing does not say who starts at any position, because ESPN does not "
    "publish an NFL depth chart, so do not call any of these players a starter."
)


def _parse_one_athlete(athlete: Any, group_label: str | None) -> dict[str, Any] | None:
    """Normalize one ``athletes[].items[]`` entry into a compact player fact dict.

    Defensive on every field (``isinstance`` guards + ``.get``); degrades a missing field
    to ``None`` and never raises. Returns ``None`` for an unusable (non-dict) entry, one
    with no display name (NEVER fabricates a player, mirroring :func:`_parse_one_article`)
    and one with no position abbreviation (it could be neither counted nor asked for).
    """
    if not isinstance(athlete, dict):
        return None

    name = _first_str(athlete.get("displayName"))
    if name is None:
        return None

    position = athlete.get("position")
    position = position if isinstance(position, dict) else {}
    abbreviation = _first_str(position.get("abbreviation"))
    if abbreviation is None:
        return None

    status = athlete.get("status")
    status = status if isinstance(status, dict) else {}
    experience = athlete.get("experience")
    experience = experience if isinstance(experience, dict) else {}
    years = experience.get("years")

    return {
        "display_name": name,
        "position": abbreviation,
        "position_name": _first_str(position.get("displayName"), position.get("name")),
        "jersey": _first_str(athlete.get("jersey")),
        "experience_years": years
        if isinstance(years, int) and not isinstance(years, bool)
        else None,
        "status": _first_str(status.get("name")),
        "group": group_label,
    }


def _position_matches(player: dict[str, Any], needle: str) -> bool:
    """Whether ``player`` plays ``needle`` (an upper-cased abbreviation OR full name)."""
    for value in (player.get("position"), player.get("position_name")):
        if isinstance(value, str) and value.strip().upper() == needle:
            return True
    return False


def parse_team_roster(payload: Any, *, position: str | None = None) -> dict[str, Any] | None:
    """Extract compact roster facts from a team ``roster`` payload.

    Pure and never-raising (mirrors :func:`parse_news`). Returns ``None`` ONLY when the
    top-level shape is unusable — a non-dict payload, or ``athletes`` is not a list.
    Otherwise returns the team display name, the season year and type, a
    ``position_counts`` map covering EVERY group (injured reserve included), the matching
    ``players``, a ``truncated`` flag and :data:`ROSTER_CAVEAT`.

    ``position`` matches case-insensitively against either the abbreviation or the full
    position name. With no ``position`` the ``players`` list is empty BY DESIGN: a full
    roster measured ~3000 tokens, which does not fit the tool loop's budget, so the counts
    alone are the answer and the tool description tells the model to ask again with a
    position. An unplayed position yields ``[]``, a valid empty answer.
    """
    if not isinstance(payload, dict):
        return None
    groups = payload.get("athletes")
    if not isinstance(groups, list):
        return None

    team = payload.get("team")
    team = team if isinstance(team, dict) else {}
    season = payload.get("season")
    season = season if isinstance(season, dict) else {}
    year = season.get("year")

    asked = position.strip().upper() if isinstance(position, str) and position.strip() else None
    counts: dict[str, int] = {}
    players: list[dict[str, Any]] = []
    truncated = False

    for group in groups:
        if not isinstance(group, dict):
            continue
        token = _first_str(group.get("position"))
        label = _ROSTER_GROUP_LABELS.get(token, token) if token is not None else None
        items = group.get("items")
        if not isinstance(items, list):
            continue
        for athlete in items:
            player = _parse_one_athlete(athlete, label)
            if player is None:
                continue
            abbreviation = player["position"]
            counts[abbreviation] = counts.get(abbreviation, 0) + 1
            if asked is None or not _position_matches(player, asked):
                continue
            if len(players) >= ROSTER_MAX_PLAYERS:
                truncated = True  # the count above still reports the true group size
                continue
            players.append(player)

    return {
        "team": _first_str(team.get("displayName")),
        "season_year": year if isinstance(year, int) and not isinstance(year, bool) else None,
        "season_type": _first_str(season.get("name")),
        "position_counts": counts,
        "players": players,
        "truncated": truncated,
        "caveat": ROSTER_CAVEAT,
    }


def league_season_year(payload: Any) -> int | None:
    """The season year the league root reports, or ``None``. Pure, never raises.

    Reads the EXPLICIT top-level ``season.year`` integer (measured 2026-08-21: ``2026``).
    The sibling ``$ref`` encodes the same year inside a URL and is deliberately not
    parsed — a year picked out of a URL string is a guess about ESPN's path shape, and
    this is the one value on the open path that must never be guessed. A payload without
    a usable integer yields ``None`` so the caller degrades rather than naming a year.
    """
    if not isinstance(payload, dict):
        return None
    season = payload.get("season")
    season = season if isinstance(season, dict) else {}
    year = season.get("year")
    return year if isinstance(year, int) and not isinstance(year, bool) else None


def _name_tokens(value: Any) -> list[str]:
    """Normalized, suffix-stripped name tokens from ``value``. Pure, never raises.

    Lowercases, drops apostrophes / periods / hyphens (so "Al'zillion" and "Hill-Green"
    match their unpunctuated spellings), collapses whitespace, then drops trailing
    generational suffixes. A value that is not a usable string yields ``[]``.
    """
    if not isinstance(value, str):
        return []
    lowered = value.lower()
    for character in ("'", "’", ".", "-"):
        lowered = lowered.replace(character, "")
    tokens = lowered.split()
    while len(tokens) > 1 and tokens[-1] in _NAME_SUFFIXES:
        tokens.pop()
    return tokens


def _first_names_match(query_first: str, roster_first: str) -> bool:
    """Whether a queried first name matches a roster first name.

    Prefix in EITHER direction, which is what makes "Matt" hit "Matthew" and "Matthew"
    hit a roster "Matt". A nickname unrelated to its formal name — Bill for William — is
    accepted as a miss; closing it would need a lookup table this does not carry.
    """
    return (
        query_first == roster_first
        or roster_first.startswith(query_first)
        or query_first.startswith(roster_first)
    )


def find_roster_athletes(payload: Any, name: Any) -> list[dict[str, Any]] | None:
    """Find every athlete on a raw ``roster`` payload whose name matches ``name``.

    Pure and never-raising. Reads the RAW payload rather than
    :func:`parse_team_roster`'s output, which deliberately drops the athlete ``id``
    (D-7). Returns ``None`` ONLY when the top-level shape is unusable — a non-dict
    payload, or ``athletes`` is not a list — mirroring the list / ``[]`` / ``None``
    trichotomy of :func:`parse_injuries`. ``[]`` means the roster was read fine and
    nobody on it matches, which is a fact the caller reports rather than a failure.

    A one-token query matches on surname alone; a longer one also requires the first
    names to match per :func:`_first_names_match`. An athlete whose ``id`` is not
    digit-shaped is skipped, so only a value that could legally reach the stats URL is
    ever returned (T-s5y-01).
    """
    if not isinstance(payload, dict):
        return None
    groups = payload.get("athletes")
    if not isinstance(groups, list):
        return None

    query = _name_tokens(name)
    if not query:
        return []
    query_surname = query[-1]
    query_first = query[0] if len(query) > 1 else None

    matches: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        items = group.get("items")
        if not isinstance(items, list):
            continue
        for athlete in items:
            if not isinstance(athlete, dict):
                continue
            display_name = _first_str(athlete.get("displayName"), athlete.get("fullName"))
            if display_name is None:
                continue  # never fabricate a player (mirrors _parse_one_athlete)
            athlete_id = athlete.get("id")
            athlete_id = athlete_id if isinstance(athlete_id, str) else ""
            if _ATHLETE_ID_RE.fullmatch(athlete_id.strip()) is None:
                continue
            full = _name_tokens(display_name)
            surname_tokens = _name_tokens(athlete.get("lastName")) or full[-1:]
            first_tokens = _name_tokens(athlete.get("firstName")) or full[:1]
            if not surname_tokens or surname_tokens[-1] != query_surname:
                continue
            if query_first is not None and not (
                first_tokens and _first_names_match(query_first, first_tokens[0])
            ):
                continue
            position = athlete.get("position")
            position = position if isinstance(position, dict) else {}
            matches.append(
                {
                    "athlete_id": athlete_id.strip(),
                    "display_name": display_name,
                    "position": _first_str(position.get("abbreviation")),
                }
            )
    return matches


def parse_athlete_search(payload: Any) -> list[dict[str, Any]] | None:
    """Extract the NFL players from a raw ``search/v2`` payload. Pure, never raises.

    Returns ``None`` ONLY when the top-level shape is unusable (a non-dict payload);
    ``[]`` when the search ran and matched no NFL player, which is a fact the caller
    reports rather than a failure (the same trichotomy as :func:`find_roster_athletes`).

    Two filters, both measured against issue #183's Route B traps. The id comes from
    ``uid`` and never from the GUID ``id``. Only an entry whose ``subtitle`` is one of the
    32 club display names survives, which is what drops the soccer and college players
    the cross-sport search returns for a football name.
    """
    if not isinstance(payload, dict):
        return None
    results = payload.get("results")
    if not isinstance(results, list):
        return []

    found: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        contents = result.get("contents")
        if not isinstance(contents, list):
            continue
        for entry in contents:
            if not isinstance(entry, dict) or entry.get("type") != "player":
                continue
            display_name = _first_str(entry.get("displayName"))
            team_name = _first_str(entry.get("subtitle"))
            if display_name is None or team_name is None:
                continue  # never fabricate a player, and never guess his team
            if team_name.upper() not in NFL_TEAM_DISPLAY_NAMES_UPPER:
                continue
            uid = entry.get("uid")
            match = _SEARCH_UID_ATHLETE_RE.search(uid) if isinstance(uid, str) else None
            if match is None:
                continue
            found.append(
                {
                    "athlete_id": match.group(1),
                    "display_name": display_name,
                    "team_name": team_name,
                }
            )
    return found


def _season_year(row: Any) -> int | None:
    """The integer ``season.year`` of one ``statistics[]`` row, or ``None``."""
    if not isinstance(row, dict):
        return None
    season = row.get("season")
    season = season if isinstance(season, dict) else {}
    year = season.get("year")
    return year if isinstance(year, int) and not isinstance(year, bool) else None


def _season_team_name(row: Any, teams: Any) -> str | None:
    """The club display name one ``statistics[]`` row belongs to, or ``None``.

    Resolves the row's own ``teamSlug`` against the payload's top-level ``teams`` block,
    so the team reported is the one the player played for THAT season rather than the one
    he is on today — the grounding defect measured live on Isiah Pacheco, whose 2025 row
    is ``kansas-city-chiefs`` while ESPN rosters him in Detroit now. An unresolvable slug
    yields ``None`` and is never guessed at from the slug text.
    """
    if not isinstance(row, dict) or not isinstance(teams, dict):
        return None
    slug = row.get("teamSlug")
    entry = teams.get(slug) if isinstance(slug, str) else None
    if not isinstance(entry, dict):
        return None
    return _first_str(entry.get("displayName"), entry.get("shortDisplayName"), entry.get("name"))


def _is_combined_row(row: Any, teams: Any) -> bool:
    """Whether ``row`` is ESPN's whole-season row for a season split between clubs.

    Measured 2026-08-20 (Davante Adams 2024, Diontae Johnson 2024): a per-club row carries
    ``teamId`` and a slug the ``teams`` block resolves, while the combined row carries
    neither and slugs itself "2024 Totals". BOTH signals are required so a merely
    malformed row is never mistaken for the season's total.
    """
    if not isinstance(row, dict):
        return False
    return "teamId" not in row and _season_team_name(row, teams) is None


def _select_season_row(rows: list[Any], teams: Any) -> Any | None:
    """The one row out of ``rows`` whose figures cover the WHOLE season, or ``None``.

    With several club rows and no combined row the category is dropped: relaying one
    club's partial figures as the season's own is the misattribution this guards against.
    """
    if len(rows) == 1:
        return rows[0]
    return next((row for row in rows if _is_combined_row(row, teams)), None)


def _stat_value(value: Any) -> str | None:
    """One statistic relayed VERBATIM as a string (D-2), or ``None`` if unusable.

    Thousands separators stay intact — "4,707" reaches the model as "4,707". Coercing
    would mean parsing "65.0", "4,707" and "-" through one conversion that can fail,
    inside a function contracted never to raise, for no gain.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _category_facts(category: dict, stats_row: Any) -> dict[str, str]:
    """One category's statistics as an ordered key -> verbatim value mapping.

    D-3: iterate by index and prefer ``displayNames[i]``, falling back to ``names[i]``
    then ``labels[i]``. ``labels`` is never the first choice because it is NOT unique —
    the measured ``defensive`` category repeats ``YDS``, and keying on it silently drops
    a statistic. An index whose key was already emitted is skipped so a collision can
    never overwrite a value.
    """
    if not isinstance(stats_row, list):
        return {}
    key_sources: list[list[Any]] = []
    for field in ("displayNames", "names", "labels"):
        source = category.get(field)
        key_sources.append(source if isinstance(source, list) else [])
    facts: dict[str, str] = {}
    for index, raw_value in enumerate(stats_row):
        value = _stat_value(raw_value)
        if value is None:
            continue
        candidates = [source[index] for source in key_sources if index < len(source)]
        key = _first_str(*candidates)
        if key is None or key in facts:
            continue
        facts[key] = value
    return facts


def _is_zeroish_category(facts: dict[str, str]) -> bool:
    """Whether every value but Games Played is zero-noise (D-6)."""
    values = [value for key, value in facts.items() if key not in _GAMES_PLAYED_KEYS]
    return bool(values) and all(value.strip() in _ZEROISH_STAT_VALUES for value in values)


def parse_athlete_stats(payload: Any, *, season: int | None = None) -> dict[str, Any] | None:
    """Extract ONE season's facts from a raw athlete career-stats payload.

    Pure and never-raising (mirrors :func:`parse_team_roster`). Returns ``None`` ONLY
    when the top-level shape is unusable — a non-dict payload, or ``categories`` is not
    a list.

    D-1 (LOCKED): with no ``season`` the target is ``max(available_seasons)``, the newest
    season in ESPN's OWN table. It is never a database read — the open path makes no
    ``db_bridge`` call at all, and the fetched payload already carries every season the
    athlete has. A ``season`` the table does not carry returns ``season`` ``None`` and
    ``stats`` ``{}`` with ``available_seasons`` intact; the caller attaches the note,
    because this function never phrases.

    The payload carries NO athlete name (measured), so the player's identity comes from
    the roster match, never from here. It DOES carry the team of each season, and
    ``season_teams`` reports the club (or, for a split season, the clubs) the target
    season's rows belong to — the fix for the live defect where the player's current team
    was the only team name in the payload and the model attached it to a past season.
    """
    if not isinstance(payload, dict):
        return None
    categories = payload.get("categories")
    if not isinstance(categories, list):
        return None
    teams = payload.get("teams")

    years: set[int] = set()
    for category in categories:
        if not isinstance(category, dict):
            continue
        rows = category.get("statistics")
        if not isinstance(rows, list):
            continue
        for row in rows:
            year = _season_year(row)
            if year is not None:
                years.add(year)
    available_seasons = sorted(years)

    asked = season if isinstance(season, int) and not isinstance(season, bool) else None
    target = asked if asked is not None else (max(years) if years else None)

    stats: dict[str, dict[str, str]] = {}
    season_teams: list[str] = []
    if target in years:
        # Teams are gathered in their own pass over EVERY category, ahead of the category
        # cap and the all-zero drop, so a season split between clubs still names them all
        # even when the category that recorded the move is not one of the ones relayed.
        for category in categories:
            rows = category.get("statistics") if isinstance(category, dict) else None
            for row in rows if isinstance(rows, list) else []:
                if _season_year(row) != target:
                    continue
                club = _season_team_name(row, teams)
                if club is not None and club not in season_teams:
                    season_teams.append(club)

        for category in categories:
            if len(stats) >= STATS_MAX_CATEGORIES:
                break
            if not isinstance(category, dict):
                continue
            label = _first_str(category.get("displayName"), category.get("name"))
            if label is None or label in stats:
                continue
            rows = category.get("statistics")
            rows = rows if isinstance(rows, list) else []
            row = _select_season_row([r for r in rows if _season_year(r) == target], teams)
            if row is None:
                continue
            facts = _category_facts(category, row.get("stats"))
            if not facts or _is_zeroish_category(facts):
                continue
            stats[label] = facts
    else:
        target = None

    return {
        "season": target,
        "season_teams": season_teams,
        "available_seasons": available_seasons,
        "stats": stats,
        "caveat": STATS_CAVEAT,
    }


def games_played(stats: Any) -> str | None:
    """The Games Played value out of :func:`parse_athlete_stats`'s ``stats``, or ``None``.

    Pure, never raises. Exists so the caller can say HOW MUCH of an unfinished season a
    total covers without re-walking the raw payload; the value stays verbatim (D-2).
    """
    if not isinstance(stats, dict):
        return None
    for facts in stats.values():
        if not isinstance(facts, dict):
            continue
        for key in _GAMES_PLAYED_KEYS:
            value = _first_str(facts.get(key))
            if value is not None:
                return value
    return None


def parse_team_schedule(payload: Any, *, week: int | None = None) -> dict | None:
    """Select ONE regular-season game out of a team ``schedule`` payload.

    Pure and never-raising (mirrors :func:`parse_team_roster`). Returns ``None`` ONLY when
    the top-level shape is unusable — a non-dict payload, ``events`` that is not a list, or
    a ``requestedSeason.type`` that is not the regular season, since a payload that is not
    the regular season is not the thing this parser contracts to read.

    The season comes from ``requestedSeason.year`` and NEVER from the sibling top-level
    ``season`` block, which measured ``{year: 2026, type: 1, Preseason}`` even on a request
    that correctly returned 2025's seventeen games. ``requestedSeason`` is the authoritative
    echo; the other one lies.

    With a ``week`` the selected ``game`` is the event at that week number, which is how a
    bye week and an unplayed week come back as no game rather than as a different one. With
    no ``week`` it is the COMPLETED event with the highest week number — chosen by week
    number rather than by list position, so a reordered payload cannot pick the wrong game.
    ``any_completed`` reports whether the season has begun at all, which is D-7's fallback
    predicate.
    """
    if not isinstance(payload, dict):
        return None
    events = payload.get("events")
    if not isinstance(events, list):
        return None
    requested = payload.get("requestedSeason")
    requested = requested if isinstance(requested, dict) else {}
    if requested.get("type") != REGULAR_SEASON_TYPE:
        return None

    year = requested.get("year")
    team = payload.get("team")
    team = team if isinstance(team, dict) else {}
    bye = payload.get("byeWeek")
    asked = week if isinstance(week, int) and not isinstance(week, bool) else None

    usable: list[dict[str, Any]] = []
    any_completed = False
    for event in events:
        parsed = _parse_one_scheduled_event(event)
        if parsed is None:
            continue
        any_completed = any_completed or parsed["completed"]
        usable.append(parsed)

    game: dict[str, Any] | None = None
    if asked is not None:
        for candidate in usable:
            if candidate["week"] == asked:
                game = candidate
                break
    else:
        for candidate in usable:
            if candidate["completed"] and (game is None or candidate["week"] > game["week"]):
                game = candidate

    return {
        "season": year if isinstance(year, int) and not isinstance(year, bool) else None,
        "team": _first_str(team.get("displayName")),
        "bye_week": bye if isinstance(bye, int) and not isinstance(bye, bool) else None,
        "any_completed": any_completed,
        "game": game,
    }


def _parse_one_scheduled_event(event: Any) -> dict[str, Any] | None:
    """Normalize one ``events[]`` entry, or ``None`` when it could not be selected anyway.

    Defensive on every field; degrades a missing name or date to ``None`` and never raises.
    An event with no digit id could not be fetched and one with no integer week number could
    not be asked for, so both are dropped rather than emitted half-empty.
    """
    if not isinstance(event, dict):
        return None
    event_id = _first_str(event.get("id"))
    if event_id is None or _EVENT_ID_RE.fullmatch(event_id) is None:
        return None
    week_block = event.get("week")
    week_block = week_block if isinstance(week_block, dict) else {}
    number = week_block.get("number")
    if not isinstance(number, int) or isinstance(number, bool):
        return None

    competitions = event.get("competitions")
    competition = competitions[0] if isinstance(competitions, list) and competitions else None
    competition = competition if isinstance(competition, dict) else {}
    status = competition.get("status")
    status = status if isinstance(status, dict) else {}
    status_type = status.get("type")
    status_type = status_type if isinstance(status_type, dict) else {}

    return {
        "event_id": event_id,
        "week": number,
        "name": _first_str(event.get("name")),
        "date": _first_str(event.get("date")),
        # Identity, not truthiness: only ESPN's own boolean means the game is finished.
        "completed": status_type.get("completed") is True,
    }


# The fields one scheduled game carries into a MODEL-facing payload. ``event_id`` is
# absent on purpose: an id the model can see is an id it may learn to send back (D-4 of
# 260820-s5y), and it is the one field a fixture list has no use for.
_SEASON_GAME_FIELDS = ("week", "name", "date", "completed")


def parse_team_season(payload: Any) -> dict | None:
    """Extract a team's WHOLE regular-season fixture list from a ``schedule`` payload.

    Pure and never-raising. Rejects the SAME three shapes :func:`parse_team_schedule`
    rejects and reads the season from ``requestedSeason`` for the same reason. Games are
    projected to :data:`_SEASON_GAME_FIELDS` and sorted by week, so a reordered payload
    cannot present a scrambled season.
    """
    if not isinstance(payload, dict):
        return None
    events = payload.get("events")
    if not isinstance(events, list):
        return None
    requested = payload.get("requestedSeason")
    requested = requested if isinstance(requested, dict) else {}
    if requested.get("type") != REGULAR_SEASON_TYPE:
        return None

    year = requested.get("year")
    team = payload.get("team")
    team = team if isinstance(team, dict) else {}
    bye = payload.get("byeWeek")

    parsed = [game for game in map(_parse_one_scheduled_event, events) if game is not None]
    parsed.sort(key=lambda game: game["week"])
    games = [
        {field: game[field] for field in _SEASON_GAME_FIELDS}
        for game in parsed[:SCHEDULE_MAX_EVENTS]
    ]

    return {
        "season": year if isinstance(year, int) and not isinstance(year, bool) else None,
        "team": _first_str(team.get("displayName")),
        "bye_week": bye if isinstance(bye, int) and not isinstance(bye, bool) else None,
        "games": games,
        "any_completed": any(game["completed"] for game in games),
    }


def _parse_one_game_leader(category: Any) -> dict[str, str | None] | None:
    """Normalize one leaders category into ``{category, player, position, stat_line}``.

    ``stat_line`` is ESPN's OWN formatted line ("10/22, 102 YDS"), relayed verbatim so no
    figure is ever recomputed here. A category with no usable athlete or no display value is
    dropped rather than emitted half-empty (mirrors :func:`_parse_one_athlete`).
    """
    if not isinstance(category, dict):
        return None
    label = _first_str(category.get("displayName"), category.get("name"))
    if label is None:
        return None
    entries = category.get("leaders")
    if not isinstance(entries, list) or not entries:
        return None
    entry = entries[0]
    if not isinstance(entry, dict):
        return None
    stat_line = _first_str(entry.get("displayValue"))
    if stat_line is None:
        return None
    athlete = entry.get("athlete")
    athlete = athlete if isinstance(athlete, dict) else {}
    player = _first_str(athlete.get("displayName"))
    if player is None:
        return None
    position = athlete.get("position")
    position = position if isinstance(position, dict) else {}

    return {
        "category": label,
        "player": player,
        "position": _first_str(position.get("abbreviation")),
        "stat_line": stat_line,
    }


# How much longer than a keyword its matching word may be — enough for a plural or a
# gerund ("catches", "interceptions") and no more, so a run-together injection attempt
# such as "rushing.rushingYards:desc&limit=9999" resolves to nothing at all.
_KEYWORD_SUFFIX_MAX = 3


def _keyword_matches(words: list[str], keyword: str) -> bool:
    """Whether ``keyword``'s words run consecutively through ``words``. Pure.

    The last keyword word matches as a BOUNDED prefix rather than anywhere in the string:
    "catches" lands on "catch", while "etcpasswd" does not land on "pass" and a
    run-together injection attempt does not land on "rushing".
    """
    parts = keyword.split()
    for start in range(len(words) - len(parts) + 1):
        window = words[start : start + len(parts)]
        last = window[-1]
        if (
            window[:-1] == parts[:-1]
            and last.startswith(parts[-1])
            and len(last) - len(parts[-1]) <= _KEYWORD_SUFFIX_MAX
        ):
            return True
    return False


def league_leader_category(category: Any) -> str | None:
    """The one :data:`LEADER_SORTS` key a model-written category name selects, or ``None``.

    The ONLY producer of the category :func:`fetch_league_leaders` accepts, so the sort
    value that reaches a URL is always a code-owned literal (T-jbh-01). Pure, never raises.
    """
    if not isinstance(category, str):
        return None
    letters = "".join(one for one in category.lower() if one.isalpha() or one.isspace())
    words = letters.split()
    if not words:
        return None
    if " ".join(words) in LEADER_SORTS:
        return " ".join(words)
    for keyword, key in _LEADER_KEYWORDS:
        if _keyword_matches(words, keyword):
            return key
    return None


def _leader_block(blocks: Any, name: str) -> dict | None:
    """The ``categories[]`` entry called ``name``, matched case-insensitively. Pure.

    Case-insensitively because the SORT spelling and the payload's own block name
    disagree: ``defensiveInterceptions`` against ``defensiveinterceptions`` (measured).
    """
    for block in blocks if isinstance(blocks, list) else []:
        if isinstance(block, dict) and str(block.get("name")).lower() == name.lower():
            return block
    return None


def parse_league_leaders(payload: Any, category: Any) -> dict | None:
    """Extract ONE statistic's league leaders from a raw ``byathlete`` payload.

    Pure and never-raising. Returns ``None`` on an unusable shape, on a category outside the
    allowlist, and on a payload whose OWN ``requestedSeason`` echo is not the regular season — the
    second half of the M-4 guard. Figures are keyed against the TOP-LEVEL block's names and read out
    of ``totals``, never ``values``, which is raw float. No athlete id is carried (D-4).
    """
    if not isinstance(payload, dict):
        return None
    key = league_leader_category(category)
    if key is None:
        return None
    requested = payload.get("requestedSeason")
    requested = requested if isinstance(requested, dict) else {}
    season_type = requested.get("type")
    season_type = season_type if isinstance(season_type, dict) else {}
    if season_type.get("type") != REGULAR_SEASON_TYPE:
        return None

    name, _stat = LEADER_SORTS[key]
    block = _leader_block(payload.get("categories"), name)
    if block is None:
        return None

    leaders: list[dict[str, Any]] = []
    for entry in payload.get("athletes") or []:
        if len(leaders) >= LEADERS_LIMIT:
            break
        if not isinstance(entry, dict):
            continue
        athlete = entry.get("athlete")
        athlete = athlete if isinstance(athlete, dict) else {}
        player = _first_str(athlete.get("displayName"))
        row = _leader_block(entry.get("categories"), name)
        if player is None or row is None:
            continue
        facts = _category_facts(block, row.get("totals"))
        if not facts:
            continue
        position = athlete.get("position")
        position = position if isinstance(position, dict) else {}
        leaders.append(
            {
                "player": player,
                "position": _first_str(position.get("abbreviation")),
                "team": _first_str(athlete.get("teamName")),
                "stats": dict(list(facts.items())[:LEADERS_MAX_FACTS]),
            }
        )

    year = requested.get("year")
    return {
        "season": year if isinstance(year, int) and not isinstance(year, bool) else None,
        "category": key,
        "leaders": leaders,
    }


def _gamelog_season(payload: dict) -> int | None:
    """The season a game log is for, or ``None``. Pure, never raises.

    M-1: this endpoint's ``requestedSeason`` is null, so the ``filters`` season value is
    read first and the four-digit prefix of a season-type name second. With neither, the
    caller names no year at all rather than guessing one.
    """
    filters = payload.get("filters")
    for entry in filters if isinstance(filters, list) else []:
        if not isinstance(entry, dict) or entry.get("name") != "season":
            continue
        value = _first_str(entry.get("value"))
        if value is not None and value.isdigit():
            return int(value)
    blocks = payload.get("seasonTypes")
    for block in blocks if isinstance(blocks, list) else []:
        name = _first_str(block.get("displayName")) if isinstance(block, dict) else None
        found = _GAMELOG_SEASON_RE.match(name) if name is not None else None
        if found is not None:
            return int(found.group(1))
    return None


def _parse_one_logged_game(
    event: Any, stats_row: Any, payload: dict, season_type: str | None
) -> dict[str, Any] | None:
    """Normalize ONE game log entry into a compact per-game fact dict.

    Defensive on every field; returns ``None`` for an entry with no integer week number,
    which could be neither asked for nor reported. ``links``, ``score``, ``homeTeamScore``
    and ``awayTeamScore`` are never read on ANY path: ``OPEN_OWNERSHIP_CLAUSE`` forbids
    the model stating a game score, and a field the model can see is one it may voice.
    """
    if not isinstance(event, dict):
        return None
    week = event.get("week")
    if not isinstance(week, int) or isinstance(week, bool):
        return None
    opponent = event.get("opponent")
    opponent = opponent if isinstance(opponent, dict) else {}
    facts = _category_facts(payload, stats_row)

    return {
        "week": week,
        "season_type": season_type,
        "date": _first_str(event.get("gameDate")),
        "venue": _GAMELOG_VENUES.get(_first_str(event.get("atVs")) or ""),
        "opponent": _first_str(opponent.get("displayName")),
        "result": _first_str(event.get("gameResult")),
        "stats": dict(list(facts.items())[:GAME_LOG_MAX_FACTS]),
    }


def parse_athlete_gamelog(payload: Any, *, week: int | None = None) -> dict | None:
    """Extract ONE game, or the most recent games, from a raw athlete game-log payload.

    Pure and never-raising. Returns ``None`` only on an unusable top-level shape; ``events``
    is the dict this endpoint keys by event id, not a list (M-1). The stats row is keyed
    against the TOP-LEVEL ``displayNames``/``names``, which are unique: ``labels`` repeats
    YDS, AVG, TD and LNG, so a label-keyed mapping overwrites passing yards with rushing.
    """
    if not isinstance(payload, dict):
        return None
    blocks = payload.get("seasonTypes")
    events = payload.get("events")
    if not isinstance(blocks, list) or not isinstance(events, dict):
        return None

    asked = week if isinstance(week, int) and not isinstance(week, bool) else None
    games: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        season_type = _first_str(block.get("displayName"))
        categories = block.get("categories")
        for category in categories if isinstance(categories, list) else []:
            if not isinstance(category, dict):
                continue
            regular = str(category.get("splitType")) == _GAMELOG_REGULAR_SPLIT
            if asked is not None and not regular:
                continue  # a member asking about a week means the regular season's week
            for entry in category.get("events") or []:
                if not isinstance(entry, dict):
                    continue
                event = events.get(_first_str(entry.get("eventId")) or "")
                parsed = _parse_one_logged_game(event, entry.get("stats"), payload, season_type)
                if parsed is None or (asked is not None and parsed["week"] != asked):
                    continue
                games.append(parsed)

    games.sort(key=lambda game: game["date"] or "", reverse=True)
    return {
        "season": _gamelog_season(payload),
        "week": asked,
        "games": games[:GAME_LOG_MAX_EVENTS],
    }


def _record_stat(record: Any, name: str) -> str | None:
    """One named ``records[].stats[]`` value, relayed verbatim as a string. Pure."""
    stats = record.get("stats") if isinstance(record, dict) else None
    for stat in stats if isinstance(stats, list) else []:
        if isinstance(stat, dict) and stat.get("name") == name:
            return _stat_value(stat.get("displayValue"))
    return None


def parse_team_record(payload: Any, team_abbr: Any) -> dict | None:
    """Extract ONE club's win-loss record summaries from a whole-league standings payload.

    Pure and never-raising. Returns ``None`` only on an unusable top-level shape. A club
    the payload does not carry comes back with no records for the caller to phrase, and
    another club's record is never substituted for it. No seed, rank, table place or
    points total is read on ANY path — ``OPEN_OWNERSHIP_CLAUSE`` bans both of those.
    """
    if not isinstance(payload, dict):
        return None
    entries = payload.get("standings")
    if not isinstance(entries, list):
        return None

    needle = team_abbr.strip().upper() if isinstance(team_abbr, str) else ""
    reference = payload.get("$ref")
    season = _STANDINGS_SEASON_RE.search(reference) if isinstance(reference, str) else None

    club: str | None = None
    records: dict[str, str] = {}
    games_played: str | None = None
    for entry in entries if needle else []:
        if not isinstance(entry, dict):
            continue
        team = entry.get("team")
        team = team if isinstance(team, dict) else {}
        ref = team.get("$ref")
        found = _TEAM_REF_ID_RE.search(ref) if isinstance(ref, str) else None
        if found is None:
            continue
        mapped = NFL_TEAM_BY_ID.get(found.group(1))
        if mapped is None or mapped[0] != needle:
            continue

        club = mapped[1]
        rows = entry.get("records")
        for index, record in enumerate(rows if isinstance(rows, list) else []):
            if not isinstance(record, dict):
                continue
            label = _TEAM_RECORD_LABELS.get(_first_str(record.get("type")) or "")
            summary = _stat_value(record.get("summary"))
            if label is not None and summary is not None and label not in records:
                records[label] = summary
            if index == 0:
                games_played = _record_stat(record, "gamesPlayed")
        break

    return {
        "season": int(season.group(1)) if season is not None else None,
        "team": club,
        "records": records,
        "games_played": games_played,
    }


def _winning_team(payload: dict) -> str | None:
    """The display name of the competitor ESPN flags as the winner, or ``None``.

    ``is True`` and not merely truthy: a ``1`` or a ``"true"`` in that field is a shape this
    parser does not recognize, and guessing a winner is worse than reporting none.
    """
    header = payload.get("header")
    header = header if isinstance(header, dict) else {}
    competitions = header.get("competitions")
    competition = competitions[0] if isinstance(competitions, list) and competitions else None
    competition = competition if isinstance(competition, dict) else {}
    competitors = competition.get("competitors")
    if not isinstance(competitors, list):
        return None
    for competitor in competitors:
        if not isinstance(competitor, dict) or competitor.get("winner") is not True:
            continue
        team = competitor.get("team")
        team = team if isinstance(team, dict) else {}
        return _first_str(team.get("displayName"))
    return None


def parse_game_leaders(payload: Any) -> dict | None:
    """Extract both clubs' game leaders and the winner from a game ``summary`` payload.

    Pure and never-raising. Returns ``None`` when the payload is not a dict or ``leaders``
    is not a list — which is the MEASURED shape of a game that has not been played, whose
    ``leaders`` key is absent entirely, so this is the second barrier behind the caller's
    completed-gate.

    Each club's leaders are keyed on its FULL display name rather than its abbreviation, so
    every player sits under a spelled-out club and cannot be read off against the other one
    (D-3). ``score`` is never read on ANY path: ``OPEN_OWNERSHIP_CLAUSE`` forbids the model
    stating a game score, and a field the model can see is a field it may voice (D-2).
    """
    if not isinstance(payload, dict):
        return None
    blocks = payload.get("leaders")
    if not isinstance(blocks, list):
        return None

    leaders: dict[str, list[dict[str, str | None]]] = {}
    for block in blocks:
        if not isinstance(block, dict):
            continue
        team = block.get("team")
        team = team if isinstance(team, dict) else {}
        club = _first_str(team.get("displayName"))
        if club is None:
            continue
        categories = block.get("leaders")
        if not isinstance(categories, list):
            continue
        rows = [row for row in map(_parse_one_game_leader, categories) if row is not None]
        leaders[club] = rows

    return {"leaders": leaders, "winner": _winning_team(payload)}


def league_season_phase(payload: Any) -> str | None:
    """The phase the league root says the season is in, or ``None``. Pure, never raises.

    Reads ``season.type.name`` (measured 2026-08-21: ``Preseason``), lower-cased because a
    capitalised token handed to the model comes back out of its mouth capitalised (memory:
    qa-phrasing-inversion). A payload without a usable name yields ``None`` so the caller
    says nothing about the phase rather than working one out from the calendar.
    """
    if not isinstance(payload, dict):
        return None
    season = payload.get("season")
    season = season if isinstance(season, dict) else {}
    type_block = season.get("type")
    type_block = type_block if isinstance(type_block, dict) else {}
    name = _first_str(type_block.get("name"))
    return name.lower() if name is not None else None


def _round_letters(round_name: Any) -> str:
    """``round_name`` reduced to its lowercase letters, so spacing and punctuation vary freely."""
    if not isinstance(round_name, str):
        return ""
    return "".join(character for character in round_name.lower() if character.isalpha())


def postseason_round_week(round_name: Any) -> int | None:
    """The postseason week one round name selects, or ``None``. Pure, never raises.

    The ONLY producer of the ``week`` :func:`fetch_postseason_scoreboard` accepts, and it
    can return nothing but 1, 2, 3 or 5: model-written text selects a LITERAL out of
    :data:`_ROUND_KEYWORD_WEEKS` and never reaches the URL itself. The Pro Bowl's week 4 is
    unreachable by construction, because no keyword maps to it.
    """
    letters = _round_letters(round_name)
    if not letters:
        return None
    for keyword, week in _ROUND_KEYWORD_WEEKS:
        if keyword in letters:
            return week
    return None


def asked_for_the_pro_bowl(round_name: Any) -> bool:
    """Whether ``round_name`` names the Pro Bowl — an exhibition game, never a playoff round.

    Its own predicate rather than a fifth keyword row, so the caller can tell "that is not
    a playoff round" apart from "I do not recognise that round" and say the right thing.
    """
    return "probowl" in _round_letters(round_name)


def _postseason_headline(*sources: Any) -> str | None:
    """ESPN's OWN name for a postseason game ("Super Bowl LX"), relayed verbatim, or ``None``.

    Measured on both shapes the payload carries the note in, hence the several sources.
    """
    for source in sources:
        notes = source.get("notes") if isinstance(source, dict) else None
        for note in notes if isinstance(notes, list) else []:
            headline = _first_str(note.get("headline")) if isinstance(note, dict) else None
            if headline is not None:
                return headline
    return None


def _parse_one_postseason_game(event: Any) -> dict[str, Any] | None:
    """Normalize one postseason ``events[]`` entry into a compact result fact dict.

    Defensive on every field; returns ``None`` for an entry naming no club at all rather
    than emitting a half-empty matchup (mirrors :func:`_parse_one_athlete`). ``score`` is
    never read on ANY path: ``OPEN_OWNERSHIP_CLAUSE`` forbids the model stating a game
    score, and a field the model can see is a field it may voice (D-2 of 260821-f0s).
    """
    if not isinstance(event, dict):
        return None
    competitions = event.get("competitions")
    competition = competitions[0] if isinstance(competitions, list) and competitions else None
    competition = competition if isinstance(competition, dict) else {}
    status = competition.get("status")
    status = status if isinstance(status, dict) else {}
    status_type = status.get("type")
    status_type = status_type if isinstance(status_type, dict) else {}

    teams: list[str] = []
    abbreviations: list[str] = []
    winner: str | None = None
    competitors = competition.get("competitors")
    for competitor in competitors if isinstance(competitors, list) else []:
        if not isinstance(competitor, dict):
            continue
        team = competitor.get("team")
        team = team if isinstance(team, dict) else {}
        club = _first_str(team.get("displayName"), team.get("abbreviation"))
        if club is None:
            continue
        teams.append(club)
        abbreviation = _first_str(team.get("abbreviation"))
        if abbreviation is not None:
            abbreviations.append(abbreviation.upper())
        # Identity, not truthiness: only ESPN's own boolean names a winner.
        if competitor.get("winner") is True:
            winner = club
    if not teams:
        return None

    event_id = _first_str(event.get("id"))
    if event_id is not None and _EVENT_ID_RE.fullmatch(event_id) is None:
        event_id = None  # could not be fetched anyway, and must never reach a URL

    return {
        "game": _first_str(
            _postseason_headline(competition, event), event.get("shortName"), event.get("name")
        ),
        "teams": teams,
        "winner": winner,
        "date": _first_str(event.get("date")),
        "completed": status_type.get("completed") is True,
        "event_id": event_id,
        "abbreviations": abbreviations,
    }


# The fields a postseason game carries into a MODEL-facing payload. ``event_id`` and
# ``abbreviations`` are absent on purpose: an id the model can see is an id it may learn to
# send back (D-4 of 260820-s5y), and the abbreviations exist only to match a game.
_POSTSEASON_RESULT_FIELDS = ("game", "teams", "winner", "date", "completed")


def find_postseason_games(payload: Any, *, week: int | None = None) -> list[dict[str, Any]] | None:
    """Every game in a raw postseason scoreboard, WITH the event id needed to fetch one.

    Pure and never-raising. Reads the RAW payload rather than
    :func:`parse_postseason_round`'s output, which drops the event id on purpose — the
    same split :func:`find_roster_athletes` makes against :func:`parse_team_roster`.
    Returns ``None`` ONLY when the top-level shape is unusable — a non-dict payload,
    ``events`` that is not a list, or a ``season.type`` that is not the postseason — and
    ``[]`` when the round carries no usable game, which is a fact the caller reports.

    ``week``, when given, additionally requires the payload's own ``week.number`` echo to
    match it, so a scoreboard answering a different round than the one asked for can never
    be read as that round's games.
    """
    if not isinstance(payload, dict):
        return None
    events = payload.get("events")
    if not isinstance(events, list):
        return None
    season = payload.get("season")
    season = season if isinstance(season, dict) else {}
    if season.get("type") != POSTSEASON_SEASON_TYPE:
        return None
    if week is not None:
        echoed = payload.get("week")
        echoed = echoed if isinstance(echoed, dict) else {}
        if echoed.get("number") != week:
            return None
    return [game for game in map(_parse_one_postseason_game, events) if game is not None]


def postseason_games_for_team(games: Any, team: Any) -> list[dict[str, Any]]:
    """The games out of ``games`` that ``team`` played in. Pure, never raises.

    Matching is EXACT against the club's abbreviation, its full display name, or the
    nickname its display name ends in — never a substring, because "NE" is a substring of
    "TENNESSEE TITANS" and a loose match here would return a different game than the one
    asked about, which is the whole defect this matching exists to prevent (260821-f0s).
    """
    needle = team.strip().upper() if isinstance(team, str) else ""
    if not needle:
        return []
    matched: list[dict[str, Any]] = []
    for game in games if isinstance(games, list) else []:
        if not isinstance(game, dict):
            continue
        candidates = {str(value).upper() for value in game.get("abbreviations") or []}
        for club in game.get("teams") or []:
            words = str(club).upper().split()
            if words:
                candidates.add(" ".join(words))
                candidates.add(words[-1])
        if needle in candidates:
            matched.append(game)
    return matched


def parse_postseason_round(payload: Any) -> dict | None:
    """Extract ONE postseason round's results from a pinned scoreboard payload.

    Pure and never-raising (mirrors :func:`parse_team_schedule`). Returns ``None`` ONLY
    when the top-level shape is unusable — a non-dict payload, ``events`` that is not a
    list, or a ``season.type`` that is not the postseason, since a payload that is not the
    postseason is not the thing this parser contracts to read.

    The season and the week come from the payload's own ``season`` / ``week`` echo of the
    request, so the year in the answer is never one the caller assumed. ``any_completed``
    is what tells a played round apart from a season whose postseason is still ahead of it
    — the measured unplayed shape is a full bracket of ``TBD`` competitors, which is why
    the caller must never relay the games on that branch.

    Each game is PROJECTED down to :data:`_POSTSEASON_RESULT_FIELDS`, which is where the
    event id stops before it can reach the model.
    """
    found = find_postseason_games(payload)
    if found is None or not isinstance(payload, dict):
        return None
    season = payload.get("season")
    season = season if isinstance(season, dict) else {}
    week = payload.get("week")
    week = week if isinstance(week, dict) else {}
    year = season.get("year")
    number = week.get("number")
    games = [{field: game[field] for field in _POSTSEASON_RESULT_FIELDS} for game in found]

    return {
        "season": year if isinstance(year, int) and not isinstance(year, bool) else None,
        "week": number if isinstance(number, int) and not isinstance(number, bool) else None,
        "games": games,
        "any_completed": any(game["completed"] for game in games),
    }


# ---------------------------------------------------------------------------
# Impure shell (best-effort HTTP + short Redis cache — never raises)
# ---------------------------------------------------------------------------


def _redis_client():
    """Build an async Redis client from ``settings.redis_url`` (single seam).

    Isolated as a tiny seam so tests monkeypatch it without touching a real socket
    and so the URL is never hardcoded (reuse the broker setting, mirroring
    :func:`app.services.notifications._redis_client`). A fresh client per call keeps
    the client bound to the calling event loop (these reads happen a few times a
    week, so no pool churn concern).
    """
    import redis.asyncio as aioredis

    return aioredis.Redis.from_url(settings.redis_url)


async def _fetch_cached(url: str, *, cache_key: str, ttl_seconds: int, label: str) -> dict | None:
    """Cache-first GET of ``url``, or ``None`` — every ESPN endpoint's one entry point.
    The contract lives in :func:`app.services.http_cache.fetch_cached`. NO headers are
    sent: ESPN's edge 403s a branded User-Agent (PR #171) — do NOT reintroduce one here.
    """
    # ``_redis_client`` is read from the module HERE, at call time, so the tests' patch
    # of this module's seam still takes effect (a default argument would defeat it).
    return await http_cache.fetch_cached(
        url,
        cache_key=cache_key,
        ttl_seconds=ttl_seconds,
        label=label,
        redis_client=_redis_client,
    )


async def fetch_game_summary(event_id: Any) -> dict | None:
    """Fetch the raw ESPN game ``summary`` payload for ``event_id`` — best-effort.

    The DIGIT GUARD runs FIRST, before the URL is formatted and before Redis or HTTP is
    touched: the id is read out of a schedule payload rather than supplied by the model,
    but a guard applied after the format string is not a guard (T-f0s-03, the same
    discipline as :func:`fetch_athlete_stats`). A ``bool`` is rejected explicitly, because
    ``str(True)`` failing the digit test is an accident rather than an intention.

    A pass delegates to :func:`_fetch_cached` (cache-first, one GET, never raises,
    fail-open Redis). ONE payload carries the injuries, the game leaders and the header,
    which is why :func:`fetch_injuries` delegates HERE rather than fetching its own copy.
    """
    candidate = "" if isinstance(event_id, bool) else str(event_id).strip()
    if _EVENT_ID_RE.fullmatch(candidate) is None:
        # Truncated before it reaches the log — the value is unbounded by contract.
        logger.warning("summary_event_id_rejected", event_id=str(event_id)[:12])
        return None

    return await _fetch_cached(
        SUMMARY_URL.format(event_id=candidate),
        cache_key=_cache_key(candidate),
        ttl_seconds=INJURIES_CACHE_TTL_SECONDS,
        label="injuries",
    )


async def fetch_injuries(espn_event_id: int) -> dict | None:
    """Fetch the raw ESPN ``summary`` payload for ``espn_event_id`` — best-effort.

    Returns the parsed ``summary`` dict, or ``None`` on any failure — the caller shows a
    fixed degrade line, never an invented injury.

    D-8: this name and :func:`fetch_game_summary` are ONE fetch writing ONE Redis entry,
    on purpose. The injuries path and the game-leaders path read the same 635 KB payload,
    so whichever asks first warms the other, and the ``qa:injuries:`` key namespace and
    log label stay as they were rather than forking into a second copy.
    """
    return await fetch_game_summary(espn_event_id)


async def fetch_news(limit: int = NEWS_FETCH_LIMIT) -> dict | None:
    """Fetch the raw ESPN league ``news`` payload — best-effort.

    A thin delegation to :func:`_fetch_cached` (cache-first, one GET, never raises,
    fail-open Redis) carrying the news URL, key, TTL and log label. Returns the parsed
    ``news`` dict, or ``None`` on any failure — the caller shows a fixed degrade line,
    never an invented headline. The URL carries ONLY the fixed integer ``limit`` —
    never user text (T-ikf-03).
    """
    return await _fetch_cached(
        NEWS_URL.format(limit=limit),
        cache_key=_news_cache_key(limit),
        ttl_seconds=NEWS_CACHE_TTL_SECONDS,
        label="news",
    )


async def fetch_team_roster(team_abbr: str) -> dict | None:
    """Fetch the raw ESPN ``roster`` payload for ``team_abbr`` — best-effort.

    The ALLOWLIST runs FIRST, before any URL is formatted and before Redis or HTTP is
    touched: ``team_abbr`` is model-supplied text, and an allowlist applied after the
    format string is not an allowlist (T-oym-01). Anything outside the canonical 32
    abbreviations returns ``None`` having attempted nothing, so only one of 32 static
    literals can ever enter the request path — the same effectively-constant target the
    news endpoint holds by carrying only a fixed integer.

    A hit delegates to :func:`_fetch_cached` (cache-first, one GET, never raises,
    fail-open Redis) with the roster key, TTL and log label.
    """
    canonical = team_abbr.strip().upper() if isinstance(team_abbr, str) else ""
    if canonical not in NFL_TEAM_ABBRS:
        # Model-supplied and unbounded — truncated before it reaches the log.
        logger.warning("roster_team_rejected", team=str(team_abbr)[:8])
        return None

    return await _fetch_cached(
        ROSTER_URL.format(team=canonical),
        cache_key=_roster_cache_key(canonical),
        ttl_seconds=ROSTER_CACHE_TTL_SECONDS,
        label="roster",
    )


async def fetch_athlete_stats(athlete_id: Any) -> dict | None:
    """Fetch one athlete's raw ESPN career-stats payload — best-effort.

    The DIGIT GUARD runs FIRST, before the URL is formatted and before Redis or HTTP is
    touched: the id was resolved from a roster payload rather than supplied by the
    model, but a guard applied after the format string is not a guard (T-s5y-01, the
    same discipline as :func:`fetch_team_roster`'s allowlist). A reject returns ``None``
    having attempted nothing.

    A pass delegates to :func:`_fetch_cached` (cache-first, one GET, never raises,
    fail-open Redis). One payload carries the athlete's WHOLE career, so a second
    question about a different season of his is served from this same cache entry —
    ESPN ignores a ``?season=`` parameter, and the season is selected in
    :func:`parse_athlete_stats` instead.
    """
    candidate = athlete_id.strip() if isinstance(athlete_id, str) else ""
    if _ATHLETE_ID_RE.fullmatch(candidate) is None:
        # Truncated before it reaches the log — the value is unbounded by contract.
        logger.warning("athlete_stats_id_rejected", athlete_id=str(athlete_id)[:12])
        return None

    return await _fetch_cached(
        ATHLETE_STATS_URL.format(athlete_id=candidate),
        cache_key=_athlete_stats_cache_key(candidate),
        ttl_seconds=ATHLETE_STATS_CACHE_TTL_SECONDS,
        label="athlete_stats",
    )


async def fetch_league_leaders(category: Any, season: int | None = None) -> dict | None:
    """Fetch ONE statistic's regular-season league leaders — best-effort.

    The ALLOWLIST runs FIRST and the sort comes OUT of :data:`LEADER_SORTS`, never out of
    the argument: an allowlist applied after the format string is not an allowlist
    (T-jbh-01). A miss returns ``None`` having attempted nothing. The resolved pair is
    encoded with ``quote(safe="")`` so the colon cannot alter the target.
    """
    key = league_leader_category(category)
    if key is None:
        # Model-written and unbounded — truncated before it reaches the log.
        logger.warning("leaders_category_rejected", category=str(category)[:24])
        return None

    if season is not None and (
        not isinstance(season, int)
        or isinstance(season, bool)
        or not _SCHEDULE_SEASON_MIN <= season <= _SCHEDULE_SEASON_MAX
    ):
        logger.warning("leaders_season_rejected", season=str(season)[:8])
        return None

    name, stat = LEADER_SORTS[key]
    url = LEADERS_URL.format(limit=LEADERS_LIMIT, sort=quote(f"{name}.{stat}:desc", safe=""))
    return await _fetch_cached(
        url if season is None else f"{url}&season={season}",
        cache_key=_leaders_cache_key(key, season),
        ttl_seconds=LEADERS_CACHE_TTL_SECONDS,
        label="leaders",
    )


async def fetch_athlete_gamelog(athlete_id: Any, season: int | None = None) -> dict | None:
    """Fetch one athlete's raw ESPN GAME LOG for one season — best-effort.

    BOTH guards run FIRST, before the URL is formatted, reusing the digit pattern
    :func:`fetch_athlete_stats` holds and the season bounds the schedule holds (T-s5y-01,
    T-jbh-02). ``?season=`` is appended only when a season was given: the parameterless
    form answers the CURRENT season (M-1), so no year is ever guessed here.
    """
    candidate = athlete_id.strip() if isinstance(athlete_id, str) else ""
    if _ATHLETE_ID_RE.fullmatch(candidate) is None:
        # Truncated before it reaches the log — the value is unbounded by contract.
        logger.warning("gamelog_athlete_id_rejected", athlete_id=str(athlete_id)[:12])
        return None

    if season is not None and (
        not isinstance(season, int)
        or isinstance(season, bool)
        or not _SCHEDULE_SEASON_MIN <= season <= _SCHEDULE_SEASON_MAX
    ):
        logger.warning("gamelog_season_rejected", season=str(season)[:8])
        return None

    url = GAMELOG_URL.format(athlete_id=candidate)
    return await _fetch_cached(
        url if season is None else f"{url}?season={season}",
        cache_key=_gamelog_cache_key(candidate, season),
        ttl_seconds=GAMELOG_CACHE_TTL_SECONDS,
        label="gamelog",
    )


async def fetch_athlete_search(name: Any) -> dict | None:
    """Search ESPN for players called ``name`` — best-effort, used only after a roster miss.

    The LENGTH CAP and the encoding both run BEFORE the URL is formatted and before Redis
    or HTTP is touched: ``name`` is model-influenced text, and ``quote(safe="")`` is what
    guarantees it stays one query-parameter VALUE and can never alter the request target
    (T-s5y-08 — the constant ``limit`` and ``type`` parameters that follow it cannot be
    displaced, because the ``&`` that would displace them is encoded as ``%26``).

    Lower-cased before both the URL and the cache key, because ESPN's search is
    case-insensitive (measured) and one cache entry should serve either spelling. A pass
    delegates to :func:`_fetch_cached` (cache-first, one GET, never raises, fail-open
    Redis); :func:`parse_athlete_search` filters the result down to NFL players.
    """
    query = " ".join(name.split()).lower() if isinstance(name, str) else ""
    if not query or len(query) > SEARCH_QUERY_MAX_CHARS:
        # Model-influenced and unbounded — truncated before it reaches the log.
        logger.warning("athlete_search_query_rejected", query=str(name)[:SEARCH_QUERY_MAX_CHARS])
        return None

    encoded = quote(query, safe="")
    return await _fetch_cached(
        ATHLETE_SEARCH_URL.format(query=encoded, limit=ATHLETE_SEARCH_LIMIT),
        cache_key=_athlete_search_cache_key(encoded),
        ttl_seconds=ATHLETE_SEARCH_CACHE_TTL_SECONDS,
        label="athlete_search",
    )


async def fetch_league() -> dict | None:
    """Fetch the raw ESPN NFL league-root payload — best-effort.

    A thin delegation to :func:`_fetch_cached` (cache-first, one GET, never raises,
    fail-open Redis). Nothing runs before the request the way the roster's allowlist and
    the search's encoding do, because there is no format string to run it before:
    :data:`LEAGUE_URL` is whole and constant. :func:`league_season_year` reads the season
    year out of the result; ``None`` here means the caller says nothing about the season
    being played rather than naming one.
    """
    return await _fetch_cached(
        LEAGUE_URL,
        cache_key=_LEAGUE_CACHE_KEY,
        ttl_seconds=LEAGUE_CACHE_TTL_SECONDS,
        label="league",
    )


async def fetch_team_schedule(team_abbr: str, *, season: int | None = None) -> dict | None:
    """Fetch one team's regular-season ``schedule`` payload — best-effort.

    BOTH guards run FIRST, before any URL is formatted and before Redis or HTTP is touched
    (T-f0s-01/02). ``team_abbr`` goes through the SAME canonical 32-abbreviation allowlist
    :func:`fetch_team_roster` uses — measured 2026-08-21, this path accepts the standard
    abbreviation, so no numeric team-id table and no second allowlist exists to drift.
    ``season``, when given, must be a real ``int`` (never a ``bool``) inside a plausible
    four-digit range. Either reject returns ``None`` having attempted nothing.

    A pass delegates to :func:`_fetch_cached` (cache-first, one GET, never raises,
    fail-open Redis). With no ``season`` the season-less URL is used, whose reply carries
    the CURRENT season's games AND echoes its year in ``requestedSeason`` (D-5).
    """
    canonical = team_abbr.strip().upper() if isinstance(team_abbr, str) else ""
    if canonical not in NFL_TEAM_ABBRS:
        # Model-supplied and unbounded — truncated before it reaches the log.
        logger.warning("schedule_team_rejected", team=str(team_abbr)[:8])
        return None

    if season is not None and (
        not isinstance(season, int)
        or isinstance(season, bool)
        or not _SCHEDULE_SEASON_MIN <= season <= _SCHEDULE_SEASON_MAX
    ):
        logger.warning("schedule_season_rejected", season=str(season)[:8])
        return None

    url = (
        CURRENT_TEAM_SCHEDULE_URL.format(team=canonical)
        if season is None
        else TEAM_SCHEDULE_URL.format(team=canonical, season=season)
    )
    return await _fetch_cached(
        url,
        cache_key=_schedule_cache_key(canonical, season),
        ttl_seconds=SCHEDULE_CACHE_TTL_SECONDS,
        label="schedule",
    )


async def fetch_standings(season: int) -> dict | None:
    """Fetch ONE season's whole-league regular-season standings — best-effort.

    The season range guard runs FIRST, before the URL is formatted, reusing the SAME
    bounds the schedule holds so one pair covers every endpoint (T-jbh-02). A pass
    delegates to :func:`_fetch_cached` (D-7): ONE fetch carries all 32 clubs, so a second
    club asked about is served from this same cache entry.
    """
    if (
        not isinstance(season, int)
        or isinstance(season, bool)
        or not _SCHEDULE_SEASON_MIN <= season <= _SCHEDULE_SEASON_MAX
    ):
        logger.warning("standings_season_rejected", season=str(season)[:8])
        return None

    return await _fetch_cached(
        STANDINGS_URL.format(season=season),
        cache_key=_standings_cache_key(season),
        ttl_seconds=STANDINGS_CACHE_TTL_SECONDS,
        label="standings",
    )


async def fetch_postseason_scoreboard(season: int, week: int) -> dict | None:
    """Fetch ONE postseason round's scoreboard payload — best-effort.

    BOTH guards run FIRST, before the URL is formatted and before Redis or HTTP is touched,
    the same discipline :func:`fetch_team_schedule` holds. ``season`` is the one
    model-influenced value here: it must be a real ``int`` (never a ``bool``) inside the
    SAME plausible range the schedule already bounds, so one pair of bounds covers both
    endpoints and no second copy can drift. ``week`` is never model-written at all — it is
    a literal out of :data:`POSTSEASON_ROUND_LABELS`, produced by
    :func:`postseason_round_week`, and this guard rejects anything else, so only 1, 2, 3 or
    5 can reach the URL and the Pro Bowl's week 4 is unreachable.

    A pass delegates to :func:`_fetch_cached` (cache-first, one GET, never raises,
    fail-open Redis).
    """
    if (
        not isinstance(season, int)
        or isinstance(season, bool)
        or not _SCHEDULE_SEASON_MIN <= season <= _SCHEDULE_SEASON_MAX
    ):
        logger.warning("postseason_season_rejected", season=str(season)[:8])
        return None
    if not isinstance(week, int) or isinstance(week, bool) or week not in POSTSEASON_WEEKS:
        logger.warning("postseason_week_rejected", week=str(week)[:8])
        return None

    return await _fetch_cached(
        POSTSEASON_SCOREBOARD_URL.format(season=season, week=week),
        cache_key=_postseason_cache_key(season, week),
        ttl_seconds=POSTSEASON_CACHE_TTL_SECONDS,
        label="postseason",
    )
