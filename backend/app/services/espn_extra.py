"""On-demand ESPN "extras" adapter — one shared fetch-and-cache shell + pure parsers.

The Path-B seam (outside-intelligence): when a league member asks the bot something the
app's own database does not own — an injury report, a headline, a team roster — the bot fetches
the public ESPN payload RIGHT THEN, caches it briefly in Redis, and parses it into a
deterministic fact structure. NO new DB tables, NO Celery-beat poller — freshness matters
most here, so on-demand is always fresh (design:
``.planning/notes/discord-query-bot-path-b-design.md``).

Design — impure shell / pure never-raising core (mirrors :mod:`app.scoreboard.espn`):

* IMPURE: ONE shell, :func:`_fetch_cached`, serves EVERY endpoint — cache-first, then a
  single ``httpx`` GET, then a best-effort cache write. It NEVER raises: any
  HTTP/timeout/non-200/parse error degrades to ``None`` (the caller shows a fixed degrade
  line, never an invented fact), and a Redis outage FAILS OPEN on both the read and the
  write. :func:`fetch_injuries`, :func:`fetch_news`, :func:`fetch_team_roster` and
  :func:`fetch_athlete_stats` are thin delegations supplying their own URL, cache key,
  TTL and log label.
* PURE: one parser per endpoint (:func:`parse_injuries`, :func:`parse_news`,
  :func:`parse_team_roster`, :func:`parse_athlete_stats`), plus :func:`find_roster_athletes`
  resolving a name against a raw roster payload, turning an already-parsed payload into
  facts. Defensive on EVERY field (isinstance guards,
  ``.get``, degrade to ``None``); never raises — this is what the offline tests exercise.

This module imports NO ``discord`` and lives on the Discord-free side: the qa.py brain
imports THIS seam for the HTTP+cache, staying itself HTTP-free.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
import structlog

from app.config import settings
from app.seeds.teams import NFL_TEAMS

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants (mirror the scoreboard adapter's endpoint/UA/timeout conventions)
# ---------------------------------------------------------------------------

# The public, no-auth ESPN game ``summary`` endpoint (SAME host family as the
# scoreboard we already poll). One call carries BOTH teams' injuries (+ game news).
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={event_id}"

DEFAULT_TIMEOUT = 10.0

# Short Redis cache: freshness matters for injuries, but ~10 min cushions repeat asks
# so a flurry of questions on the same game is ONE upstream call.
INJURIES_CACHE_TTL_SECONDS = 600


def _cache_key(event_id: int) -> str:
    """The Redis key for one event's cached ``summary`` payload."""
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

# The sentence the model is most likely to voice, so it is concrete and complete rather
# than a terse fragment (memory: qa-phrasing-inversion). The second clause exists because
# D-6 deliberately keeps a category ESPN publishes a rate for on zero attempts.
STATS_CAVEAT = (
    "These are the official ESPN season totals for the one NFL season this answer names, "
    "so state that season's year when you report any of them and never call them this "
    "year's or last year's figures. When a category shows the player had no attempts in "
    "it, do not report that category's rate statistics such as a passer rating or a "
    "yards per attempt average, because a rate worked out on no attempts is meaningless."
)


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


def _season_year(row: Any) -> int | None:
    """The integer ``season.year`` of one ``statistics[]`` row, or ``None``."""
    if not isinstance(row, dict):
        return None
    season = row.get("season")
    season = season if isinstance(season, dict) else {}
    year = season.get("year")
    return year if isinstance(year, int) and not isinstance(year, bool) else None


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
    key_sources = [
        category.get(field) if isinstance(category.get(field), list) else []
        for field in ("displayNames", "names", "labels")
    ]
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
    the roster match, never from here.
    """
    if not isinstance(payload, dict):
        return None
    categories = payload.get("categories")
    if not isinstance(categories, list):
        return None

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
    if target in years:
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
            row = next((r for r in rows if _season_year(r) == target), None)
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
        "available_seasons": available_seasons,
        "stats": stats,
        "caveat": STATS_CAVEAT,
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


async def _cache_read(key: str, *, label: str) -> dict | None:
    """Return the cached payload at ``key``, or ``None`` (FAIL-OPEN).

    Best-effort: any Redis/JSON error logs a warning and returns ``None`` so the
    caller degrades to a live fetch. A missing key also returns ``None``.
    """
    try:
        client = _redis_client()
        try:
            raw = await client.get(key)
        finally:
            await client.aclose()
        if raw is None:
            return None
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        logger.warning(f"{label}_cache_get_failed", key=key, exc_info=True)
        return None


async def _cache_write(key: str, payload: dict, *, ttl_seconds: int, label: str) -> None:
    """Best-effort write of ``payload`` under ``key`` + ``ttl_seconds``.

    FAIL-OPEN: any Redis/JSON error logs a warning and returns normally — a cache
    outage must NOT block the fetch that already succeeded.
    """
    try:
        client = _redis_client()
        try:
            await client.set(key, json.dumps(payload), ex=ttl_seconds)
        finally:
            await client.aclose()
    except Exception:
        logger.warning(f"{label}_cache_set_failed", key=key, exc_info=True)


async def _fetch_cached(url: str, *, cache_key: str, ttl_seconds: int, label: str) -> dict | None:
    """Cache-first GET of ``url`` returning the parsed JSON dict, or ``None``.

    The ONE fetch-and-cache shell behind every endpoint in this module. On a cache HIT
    the cached payload is returned WITHOUT any HTTP call; on a MISS it performs EXACTLY
    one ``httpx`` GET and best-effort writes the raw payload back under ``cache_key``.

    NEVER raises: any HTTP/timeout/non-200/parse error degrades to ``None`` (the caller
    shows a fixed degrade line, never an invented fact), and a Redis outage on either
    the read or the write fails open. A payload that is not a dict returns ``None``
    and is NOT cached.

    NO custom User-Agent — mirrors ``app.scoreboard.espn`` (see the long note there).
    ESPN's edge 403s branded UAs and allows recognized client defaults, so we let httpx
    send its own ``python-httpx/x.y.z``. Do NOT reintroduce a custom UA on ESPN hosts.
    ``DEFAULT_TIMEOUT`` is explicit so a hung ESPN response cannot block the bot loop.
    No credentials are ever sent — these are public, outbound-only GETs.

    ``label`` prefixes every emitted structlog event, so each endpoint keeps its own
    ``<label>_cache_get_failed`` / ``_cache_set_failed`` / ``_fetch_non_200`` /
    ``_fetch_failed`` names. The failure events carry ``cache_key``, which already
    embeds the event id or the limit, so no debugging detail is lost.
    """
    cached = await _cache_read(cache_key, label=label)
    if cached is not None:
        return cached

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.get(url)
        if response.status_code != 200:
            logger.warning(f"{label}_fetch_non_200", status_code=response.status_code)
            return None
        payload = response.json()
    except Exception:
        logger.warning(f"{label}_fetch_failed", key=cache_key, exc_info=True)
        return None

    if not isinstance(payload, dict):
        return None

    await _cache_write(cache_key, payload, ttl_seconds=ttl_seconds, label=label)
    return payload


async def fetch_injuries(espn_event_id: int) -> dict | None:
    """Fetch the raw ESPN ``summary`` payload for ``espn_event_id`` — best-effort.

    A thin delegation to :func:`_fetch_cached` (cache-first, one GET, never raises,
    fail-open Redis) carrying the injuries URL, key, TTL and log label. Returns the
    parsed ``summary`` dict, or ``None`` on any failure — the caller shows a fixed
    degrade line, never an invented injury.
    """
    return await _fetch_cached(
        SUMMARY_URL.format(event_id=espn_event_id),
        cache_key=_cache_key(espn_event_id),
        ttl_seconds=INJURIES_CACHE_TTL_SECONDS,
        label="injuries",
    )


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
