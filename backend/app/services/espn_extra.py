"""On-demand ESPN "extras" adapter — one shared fetch-and-cache shell + pure parsers.

The Path-B seam (outside-intelligence): when a league member asks the bot something the
app's own database does not own — an injury report, a league headline — the bot fetches
the public ESPN payload RIGHT THEN, caches it briefly in Redis, and parses it into a
deterministic fact structure. NO new DB tables, NO Celery-beat poller — freshness matters
most here, so on-demand is always fresh (design:
``.planning/notes/discord-query-bot-path-b-design.md``).

Design — impure shell / pure never-raising core (mirrors :mod:`app.scoreboard.espn`):

* IMPURE: ONE shell, :func:`_fetch_cached`, serves EVERY endpoint — cache-first, then a
  single ``httpx`` GET, then a best-effort cache write. It NEVER raises: any
  HTTP/timeout/non-200/parse error degrades to ``None`` (the caller shows a fixed degrade
  line, never an invented fact), and a Redis outage FAILS OPEN on both the read and the
  write. :func:`fetch_injuries` and :func:`fetch_news` are thin delegations supplying
  their own URL, cache key, TTL and log label.
* PURE: one parser per endpoint (:func:`parse_injuries`, :func:`parse_news`) turning an
  already-parsed payload into facts. Defensive on EVERY field (isinstance guards,
  ``.get``, degrade to ``None``); never raises — this is what the offline tests exercise.

This module imports NO ``discord`` and lives on the Discord-free side: the qa.py brain
imports THIS seam for the HTTP+cache, staying itself HTTP-free.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import structlog

from app.config import settings

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
