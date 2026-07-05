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

import discord
import structlog

from app.bot.team_emoji import decorate_team_logos
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
        return (
            f"admin cleared {event.get('target')} · Week {event.get('week')} · {event.get('slot')}"
        )
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


def render_chat(event: dict) -> str | None:
    """Render a player-facing pickem-CHAT event to its Discord line (QT-3).

    This is the SWAPPABLE seam where a FUTURE local-LLM personality layer will
    plug in — it takes the structured event and returns a chattier line than the
    terse logger feed. NO LLM / nudge / QT-4 logic lives here yet: it is a plain
    string map over the five ``targets:["chat"]`` event types. Unknown types
    return ``None`` (ignored upstream), mirroring :func:`_render`.

    The bot does NO resolution — every field (display names, integer scores, team
    abbreviations) is already shaped by the pure builders in
    :mod:`app.services.notifications`.
    """
    etype = event.get("type")

    if etype == "roster.complete":
        # NOT "locked in" — players make/set their picks; the SYSTEM locks at
        # first kickoff. No lock emoji here (260628-itg).
        return f"{event.get('actor')} got their Week {event.get('week')} picks in. 👍"
    if etype == "window.opened":
        return f"Week {event.get('week')} picks are open — get 'em in!"
    if etype == "window.closed":
        return f"Week {event.get('week')} is locked. Good luck, everyone."
    if etype == "game.final":
        return (
            f"Final: {event.get('home')} {event.get('home_score')}, "
            f"{event.get('away')} {event.get('away_score')}."
        )
    if etype == "week.recap":
        return (
            f"Week {event.get('week')}'s in the books — "
            f"{event.get('winner')} takes the week with {event.get('winner_score')}; "
            f"{event.get('leader')} leads the season."
        )
    if etype == "misc.picked":
        # LEAK-SAFE (260628-itg): actor + week ONLY — NEVER any prediction/text
        # content (predictions are hidden until the window closes).
        return f"{event.get('actor')} got their Week {event.get('week')} misc call in. 👀"
    if etype == "misc.graded":
        # Signed points: a positive shows a leading plus, a negative its sign.
        return (
            f"Week {event.get('week')} MISC graded — {event.get('actor')}'s call "
            f'"{event.get("prediction")}" was {event.get("verdict")} '
            f"({event.get('points'):+d})."
        )
    if etype == "freeze.week":
        # Deterministic body of the LIGHT lines-locked chat card (260705-jo9); also
        # the text fallback when build_freeze_week_embed fails.
        return f"Point spreads are locked for Week {event.get('week')}."
    return None


async def run_notifier(client) -> None:
    """Subscribe to ``pickem:events`` and post rendered lines into the guild.

    Builds a ``redis.asyncio`` client from ``settings.redis_url``, SUBSCRIBEs to
    :data:`EVENTS_CHANNEL`, and loops over messages. Each event is dispatched to TWO
    INDEPENDENT surfaces by its ``targets`` (260705-jo9): the LOGGER surface (when
    ``"logger" in targets``) renders via :func:`_render` and posts to
    ``discord_chat_log_channel``; the CHAT surface (when ``"chat" in targets``)
    renders via :func:`render_chat` / the embellish + embed pipeline and posts to
    ``discord_chat_channel`` — both resolved within ``DISCORD_GUILD_ID``. The two
    checks are independent ``if``\\ s (NOT an XOR else-branch), so an event targeting
    BOTH surfaces (e.g. ``freeze.week`` -> ``["logger", "chat"]``) posts to BOTH.

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

                    # Dual-surface dispatch (260705-jo9): an event is routed to the
                    # LOGGER surface AND the CHAT surface INDEPENDENTLY — two
                    # separate `if target` checks, NOT an XOR else-branch. An event
                    # targeting only one surface posts only there (unchanged); an
                    # event targeting BOTH (freeze.week -> ["logger", "chat"]) posts
                    # to BOTH. The LOGGER surface is dispatched FIRST so the chat
                    # embed blocks' `continue` statements (which advance the async
                    # for) fire only AFTER the logger post has already been sent —
                    # keeping the chat pipeline byte-for-byte. Both channels resolve
                    # within DISCORD_GUILD_ID.
                    targets = event.get("targets") or []
                    guild = client.get_guild(settings.discord_guild_id)

                    # LOGGER surface — terse ops-log line via _render (unchanged
                    # behavior: the old else-branch flowed through the generic send
                    # with mass-mention suppression, so reproduce that here).
                    if "logger" in targets:
                        logger_line = _render(event)
                        if logger_line is not None:
                            log_channel = resolve_channel(guild, settings.discord_chat_log_channel)
                            if log_channel is not None:
                                await log_channel.send(
                                    logger_line,
                                    allowed_mentions=discord.AllowedMentions.none(),
                                )

                    # CHAT surface — the full player-facing pipeline, unchanged.
                    if "chat" in targets:
                        # week.recap (260627-tfb) routes through the Tier-2 recap
                        # orchestrator: an LLM-narrated column over the full week's
                        # scores + season standings, with the deterministic
                        # render_chat one-liner baked in as the fallback
                        # (build_week_recap returns that string itself on any db OR
                        # LLM failure — so exactly one chat line lands and it NEVER
                        # raises). The Tier-1 reactive events (260627-t5u) get an
                        # LLM-phrased personality line with the same deterministic
                        # fallback via embellish_chat. All other chat types keep the
                        # plain render_chat path. window.opened and window.closed
                        # both stay on render_chat (260705-j8o): they render as LIGHT
                        # embeds below whose deterministic render_chat text is the
                        # embed's fallback line — no LLM round-trip.
                        etype = event.get("type")
                        if etype == "week.recap":
                            from app.bot.recap import build_week_recap

                            line = await build_week_recap(event)
                        elif etype in (
                            "game.final",
                            "roster.complete",
                            "misc.graded",
                            "misc.picked",
                        ):
                            from app.bot.chat_personality import embellish_chat

                            line = await embellish_chat(event)
                        else:
                            line = render_chat(event)

                        # None => unknown chat type — nothing to post on this surface.
                        if line is not None:
                            # Team-logo decoration (260627-wt5): tag team references
                            # in player-facing CHAT lines with their Discord
                            # application-emoji logo. Applied ONLY on the chat surface
                            # — the terse logger feed is never decorated.
                            # decorate_team_logos is best-effort and NEVER raises
                            # (no-op when the startup emoji fetch failed / the cache is
                            # empty), so it cannot regress posting.
                            line = decorate_team_logos(line)

                            channel = resolve_channel(guild, settings.discord_chat_channel)
                            if channel is not None:
                                # game.final (260703-piv): render a RICH embed card on
                                # the chat channel — plain title, deterministic
                                # away→home logo'd score, the voiced quip (the
                                # decorated `line`), a winner-color bar, and
                                # Busted/Cashed fields from event["impacts"].
                                # BEST-EFFORT (T-piv-02): the embed build+send is
                                # wrapped so ANY failure falls back to the existing
                                # text send — the message still posts and the loop
                                # never dies.
                                if event.get("type") == "game.final":
                                    try:
                                        from app.bot.game_final_embed import (
                                            build_game_final_embed,
                                        )

                                        embed = build_game_final_embed(event, line)
                                        await channel.send(
                                            embed=embed,
                                            allowed_mentions=discord.AllowedMentions.none(),
                                        )
                                    except Exception:
                                        logger.warning("game_final_embed_failed", exc_info=True)
                                        # Best-effort fallback: post the text line
                                        # instead of dropping the message entirely.
                                        await channel.send(
                                            line,
                                            allowed_mentions=discord.AllowedMentions.none(),
                                        )
                                    continue

                                # week.recap (260705-kuv): render the marquee amethyst
                                # "closing ceremony" embed card — plain title, the LLM
                                # narration (the decorated `line`) as the description,
                                # and omit-empty Week Winner / Best Call / Biggest Bust
                                # / Mortal Locks / Standings fields. BEST-EFFORT
                                # (T-kuv-03): the embed build+send is wrapped so ANY
                                # failure falls back to the existing text `line` send —
                                # the message still posts and the loop never dies.
                                # Mirrors the game.final block. build_week_recap (the
                                # narration + its RECAP_GUARD) is untouched.
                                if event.get("type") == "week.recap":
                                    try:
                                        from app.bot.week_recap_embed import (
                                            build_week_recap_embed,
                                        )

                                        embed = build_week_recap_embed(event, line)
                                        await channel.send(
                                            embed=embed,
                                            allowed_mentions=discord.AllowedMentions.none(),
                                        )
                                    except Exception:
                                        logger.warning("week_recap_embed_failed", exc_info=True)
                                        # Best-effort fallback: post the text line
                                        # instead of dropping the message entirely.
                                        await channel.send(
                                            line,
                                            allowed_mentions=discord.AllowedMentions.none(),
                                        )
                                    continue

                                # misc.graded (260705-if1): render a LIGHT embed card
                                # on the chat channel — plain title, a binary hit/miss
                                # marker line, the voiced quip (the decorated `line`),
                                # a green/red bar by points sign, and compact
                                # Player/Verdict/(omit-empty) Prediction fields.
                                # BEST-EFFORT (T-if1-02): the embed build+send is
                                # wrapped so ANY failure falls back to the existing
                                # text send — the message still posts and the loop
                                # never dies. Mirrors the game.final block.
                                if event.get("type") == "misc.graded":
                                    try:
                                        from app.bot.misc_graded_embed import (
                                            build_misc_graded_embed,
                                        )

                                        embed = build_misc_graded_embed(event, line)
                                        await channel.send(
                                            embed=embed,
                                            allowed_mentions=discord.AllowedMentions.none(),
                                        )
                                    except Exception:
                                        logger.warning("misc_graded_embed_failed", exc_info=True)
                                        # Best-effort fallback: post the text line
                                        # instead of dropping the message entirely.
                                        await channel.send(
                                            line,
                                            allowed_mentions=discord.AllowedMentions.none(),
                                        )
                                    continue

                                # window.opened / window.closed (260705-j8o): render a
                                # LIGHT embed card on the chat channel — plain title
                                # carrying the week, a green/red bar (open/locked), and
                                # one deterministic body line. NO LLM. BEST-EFFORT
                                # (T-j8o-02): the embed build+send is wrapped so ANY
                                # failure falls back to the existing text `line` send —
                                # the message still posts and the loop never dies.
                                # Mirrors the game.final / misc.graded blocks.
                                if event.get("type") in ("window.opened", "window.closed"):
                                    try:
                                        from app.bot.window_embed import (
                                            build_window_embed,
                                        )

                                        embed = build_window_embed(event)
                                        await channel.send(
                                            embed=embed,
                                            allowed_mentions=discord.AllowedMentions.none(),
                                        )
                                    except Exception:
                                        logger.warning("window_embed_failed", exc_info=True)
                                        # Best-effort fallback: post the text line
                                        # instead of dropping the message entirely.
                                        await channel.send(
                                            line,
                                            allowed_mentions=discord.AllowedMentions.none(),
                                        )
                                    # ADDITIVE pickem-chat personality layer
                                    # (260627-nef): AFTER the window.closed embed (or
                                    # its text fallback), post one personality line
                                    # per flagged player to the SAME chat channel.
                                    # Runs whether the embed or the fallback posted, so
                                    # the lock-commentary follow-on is preserved
                                    # byte-for-byte. Fired here — inside the
                                    # per-message try/except and only once the channel
                                    # resolved — so any LLM/db hiccup is caught by the
                                    # notifier_message_failed guard and the loop
                                    # survives (T-nef-03). build_lock_commentary is
                                    # itself best-effort and Discord-free; firing on
                                    # window.closed (all picks final) avoids leaking
                                    # any open-window pick (T-nef-02).
                                    if event.get("type") == "window.closed":
                                        from app.bot.commentary import (
                                            build_lock_commentary,
                                        )

                                        for extra in await build_lock_commentary(event.get("week")):
                                            # Chat-channel send — decorate team refs.
                                            await channel.send(
                                                decorate_team_logos(extra),
                                                allowed_mentions=discord.AllowedMentions.none(),
                                            )
                                    continue

                                # freeze.week (260705-jo9): render a LIGHT gold "lines
                                # locked" embed card on the chat channel — plain title
                                # carrying the week, a gold bar, and one deterministic
                                # body line. NO LLM. This is the CHAT half of the dual
                                # dispatch; the terse ops-log line already fired on the
                                # LOGGER surface above. BEST-EFFORT (T-jo9-02): the
                                # embed build+send is wrapped so ANY failure falls back
                                # to the existing text `line` send — the message still
                                # posts and the loop never dies. Mirrors the
                                # misc.graded / window blocks.
                                if event.get("type") == "freeze.week":
                                    try:
                                        from app.bot.freeze_week_embed import (
                                            build_freeze_week_embed,
                                        )

                                        embed = build_freeze_week_embed(event)
                                        await channel.send(
                                            embed=embed,
                                            allowed_mentions=discord.AllowedMentions.none(),
                                        )
                                    except Exception:
                                        logger.warning("freeze_week_embed_failed", exc_info=True)
                                        # Best-effort fallback: post the text line
                                        # instead of dropping the message entirely.
                                        await channel.send(
                                            line,
                                            allowed_mentions=discord.AllowedMentions.none(),
                                        )
                                    continue

                                # Mention hygiene (T-t5u-04): suppress
                                # @everyone/@here/role pings so LLM-authored chat text
                                # can never ping the server. Serves the remaining
                                # plain-text chat types (roster.complete, misc.picked).
                                await channel.send(
                                    line, allowed_mentions=discord.AllowedMentions.none()
                                )
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
