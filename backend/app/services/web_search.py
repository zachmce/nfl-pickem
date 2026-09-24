"""Web search through the operator's own SearXNG instance (issue #234).

The open path's ``search_web`` tool reads this seam. ONE cached GET per query, never
raises, ``None`` on any failure. The instance URL comes only from ``SEARXNG_URL``.
Every title and snippet is text from an arbitrary web page, so the caller fences it
before it reaches the model.
"""

from __future__ import annotations

import hashlib
from typing import Any
from urllib.parse import urlencode

from app.config import settings
from app.services import http_cache

QUERY_MAX_CHARS = 120
MAX_RESULTS = 5
SNIPPET_MAX_CHARS = 300
CACHE_TTL_SECONDS = 1800


def _base_url() -> str | None:
    """The configured instance with a scheme, or ``None`` when search is off."""
    raw = settings.searxng_url
    if not raw:
        return None
    base = raw.strip().rstrip("/")
    return base if base.startswith(("http://", "https://")) else f"https://{base}"


def enabled() -> bool:
    return _base_url() is not None


def _redis_client():
    import redis.asyncio as aioredis

    return aioredis.Redis.from_url(settings.redis_url)


async def fetch_search(query: Any) -> dict | None:
    """SearXNG's JSON results for ``query`` — best-effort, ``None`` on any failure."""
    base = _base_url()
    text = " ".join(query.split()) if isinstance(query, str) else ""
    if base is None or not text or len(text) > QUERY_MAX_CHARS:
        return None
    digest = hashlib.sha256(text.lower().encode()).hexdigest()[:32]
    return await http_cache.fetch_cached(
        f"{base}/search?{urlencode({'q': text, 'format': 'json'})}",
        cache_key=f"qa:websearch:{digest}",
        ttl_seconds=CACHE_TTL_SECONDS,
        label="web_search",
        redis_client=_redis_client,
    )


def parse_search(payload: Any) -> list[dict[str, str | None]] | None:
    """The top results as ``{title, url, snippet, published}``. Pure, never raises.

    ``None`` when the shape is unusable; ``[]`` when the search matched nothing.
    Results without a web URL are dropped, and a URL seen twice is kept once.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        return None
    results: list[dict[str, str | None]] = []
    seen: set[str] = set()
    for raw in payload["results"]:
        if not isinstance(raw, dict):
            continue
        url = raw.get("url")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")) or url in seen:
            continue
        seen.add(url)
        title = raw.get("title") if isinstance(raw.get("title"), str) else None
        snippet = raw.get("content") if isinstance(raw.get("content"), str) else None
        published = raw.get("publishedDate") if isinstance(raw.get("publishedDate"), str) else None
        results.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet[:SNIPPET_MAX_CHARS] if snippet else None,
                "published": published,
            }
        )
        if len(results) == MAX_RESULTS:
            break
    return results
