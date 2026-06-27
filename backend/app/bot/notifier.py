"""Bot-side subscriber for the Discord notification pipe (QT-1).

This is the BOT side of the pipe established in :mod:`app.services.notifications`:
it subscribes to the shared Redis ``pickem:events`` channel, renders each event,
and posts it into a Discord channel scoped to ``DISCORD_GUILD_ID``.

Why it lives HERE (not in ``db_bridge.py``): this module renders to Discord, so it
MAY import ``discord``. ``db_bridge.py`` is the Discord-free async/sync boundary and
must stay that way — the subscriber is started from ``client.py``'s ``setup_hook``.

Resilience contract
-------------------
* ``resolve_channel`` is guild-scoped (T-kd8-03): it searches ONLY the passed
  guild's channels, matching by numeric id OR by name, and returns ``None`` (with a
  warning) on a miss/blank/None — it never raises and never matches a channel in
  some other guild.
* ``run_notifier`` wraps per-message handling in try/except + continue (T-kd8-04):
  one malformed/bad event is logged and skipped — it can never kill the loop.
* ``run_notifier`` wraps the subscribe+listen in an OUTER reconnect loop with
  capped backoff: a dropped Redis connection (restart, network blip) raises out of
  ``pubsub.listen()`` — OUTSIDE the per-message guard — and is retried instead of
  silently tearing down the subscriber until a bot restart. Shutdown
  (``CancelledError``) still propagates promptly and is never retried.
"""

from __future__ import annotations

import asyncio
import json

import structlog

from app.config import get_settings
from app.services.notifications import EVENTS_CHANNEL

logger = structlog.get_logger(__name__)

# Reconnect backoff bounds (seconds). Starting small keeps a brief blip nearly
# seamless; the cap stops a long Redis outage from becoming a busy-loop.
_RECONNECT_BACKOFF_START = 1.0
_RECONNECT_BACKOFF_MAX = 30.0


async def _close_quietly(*resources) -> None:
    """Best-effort ``aclose`` of redis pubsub/client resources; never raises."""
    for resource in resources:
        try:
            await resource.aclose()
        except Exception:
            pass


def resolve_channel(guild, channel_setting: str | None):
    """Find a channel within ``guild`` by numeric id OR by name.

    Searches ONLY ``guild.channels`` (the guild-scoping requirement — never "first
    channel named X anywhere"). Returns the first match, or ``None`` (with a
    structlog warning) when ``guild`` is None, the setting is blank/None, or nothing
    matches. Accepts an id-as-string OR a name because names can change/duplicate.
    Never raises.
    """
    if guild is None:
        logger.warning("notifier_guild_unavailable", channel_setting=channel_setting)
        return None
    if channel_setting is None or not channel_setting.strip():
        logger.warning("notifier_channel_setting_blank")
        return None

    setting = channel_setting.strip()

    # Numeric setting => match by id ONLY (do not fall back to a name match).
    if setting.isdigit():
        target_id = int(setting)
        for channel in guild.channels:
            if channel.id == target_id:
                return channel
        logger.warning("notifier_channel_not_found", by="id", value=setting)
        return None

    # Otherwise match by exact channel name.
    for channel in guild.channels:
        if channel.name == setting:
            return channel
    logger.warning("notifier_channel_not_found", by="name", value=setting)
    return None


def _render(event: dict) -> str | None:
    """Render an event to its Discord line, or None for unknown types (ignored).

    The bot does NO resolution — every field it joins (the resolved side/team
    ``detail``, the ``target`` display_name, the ingest summary) is already shaped
    by the publisher's pure builders in :mod:`app.services.notifications`. QT-1
    handled only ``user.login``; QT-2 adds the seven granular pickem-logger types.
    Unknown types still return ``None`` (ignored upstream).
    """
    etype = event.get("type")

    if etype == "user.login":
        return f"{event.get('actor')} logged in"
    if etype in ("pick.created", "pick.changed"):
        return f"{event.get('actor')} pick · Week {event.get('week')} · {event.get('detail')}"
    if etype == "pick.cleared":
        return f"{event.get('actor')} cleared · Week {event.get('week')} · {event.get('detail')}"
    if etype == "admin.pick_set":
        return f"admin set {event.get('target')} · Week {event.get('week')} · {event.get('detail')}"
    if etype == "admin.pick_cleared":
        return f"admin cleared {event.get('target')} · Week {event.get('week')} · {event.get('slot')}"
    if etype == "player.registered":
        return f"new player: {event.get('actor')}"
    if etype == "ingest.season":
        return (
            f"ingested {event.get('season')} · {event.get('weeks')} wk / "
            f"{event.get('games')} games ({event.get('failed')} failed)"
        )
    if etype == "freeze.week":
        return f"Week {event.get('week')} lines frozen"
    return None


async def run_notifier(client) -> None:
    """Subscribe to ``pickem:events`` and post rendered lines into the guild.

    Builds a ``redis.asyncio`` client from ``settings.redis_url``, SUBSCRIBEs to
    :data:`EVENTS_CHANNEL`, and loops over messages. For ``user.login`` it renders
    ``"<actor> logged in"`` and sends it to the ``discord_chat_log_channel`` within
    ``DISCORD_GUILD_ID``.

    Two layers of resilience:

    * **Per message** — handling is wrapped in try/except + continue: one bad event
      is logged and skipped, never killing the loop.
    * **Per connection** — the subscribe+listen is wrapped in an OUTER reconnect
      loop with capped exponential backoff. A connection drop (Redis restart,
      network blip) raises out of ``pubsub.listen()`` — OUTSIDE the per-message
      guard — so it is caught here, logged, and the subscriber re-establishes once
      Redis returns. Cancellation (bot shutdown) propagates and is not retried.
    """
    import redis.asyncio as aioredis

    settings = get_settings()
    backoff = _RECONNECT_BACKOFF_START

    while True:
        redis_client = aioredis.from_url(settings.redis_url)
        pubsub = redis_client.pubsub()
        try:
            await pubsub.subscribe(EVENTS_CHANNEL)
            logger.info("notifier_subscribed", channel=EVENTS_CHANNEL)
            backoff = _RECONNECT_BACKOFF_START  # reset after a clean (re)subscribe

            async for message in pubsub.listen():
                try:
                    if message.get("type") != "message":
                        continue  # subscribe/confirmation frames, not payloads
                    event = json.loads(message["data"])
                    line = _render(event)
                    if line is None:
                        continue  # unknown event type — QT-2/QT-3 territory

                    guild = client.get_guild(settings.discord_guild_id)
                    channel = resolve_channel(guild, settings.discord_chat_log_channel)
                    if channel is not None:
                        await channel.send(line)
                except Exception:
                    logger.warning("notifier_message_failed", exc_info=True)
                    continue
        except asyncio.CancelledError:
            raise  # bot shutdown — stop the subscriber, do not reconnect
        except Exception:
            # Connection dropped / failed to subscribe. Log and reconnect after a
            # backoff — never let a transient Redis outage permanently kill the pipe.
            logger.warning("notifier_connection_lost", exc_info=True, retry_in_s=backoff)
        finally:
            await _close_quietly(pubsub, redis_client)

        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, _RECONNECT_BACKOFF_MAX)
