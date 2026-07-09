"""PickemBot — Discord gateway client for the NFL pick'em bot.

Responsibilities:
- `PickemBot(commands.Bot)` with `setup_hook` that loads the three cog extensions
  and guild-scopes the slash-command tree (never a bare global tree.sync).
- A gateway-aware heartbeat loop (discord.ext.tasks) that touches /tmp/bot_heartbeat
  only while the gateway is connected — makes the compose healthcheck fail when the
  gateway drops, not just when the process dies.
- `async def main()` fail-fast entrypoint: asserts discord_bot_token and
  discord_guild_id non-None, then `async with bot: await bot.start(token)`.
  Graceful SIGTERM/SIGINT handled by explicit loop.add_signal_handler() calls in
  main() that trigger bot.close(); discord.py does NOT install these on the
  await bot.start() path, and asyncio.run() only handles SIGINT on non-PID-1 processes.

Heartbeat file path: /tmp/bot_heartbeat  (the compose healthcheck references this
exact path — do not change without updating the healthcheck).
"""

from __future__ import annotations

import asyncio
import signal
from pathlib import Path

import discord
import structlog
from discord.ext import commands, tasks

from app.bot import db_bridge
from app.config import get_settings
from app.logging_config import configure_logging

logger = structlog.get_logger(__name__)

# Heartbeat file touched by the loop every 15s while the gateway is live.
HEARTBEAT_FILE = Path("/tmp/bot_heartbeat")

# Cog extensions loaded by setup_hook
COG_MODULES = (
    "app.bot.commands.register",
    "app.bot.commands.reset_password",
    "app.bot.commands.admin",
    "app.bot.commands.mention_qa",
)


class PickemBot(commands.Bot):
    """Discord bot client for the NFL pick'em platform."""

    async def setup_hook(self) -> None:
        """Load cog extensions and guild-scope the slash-command tree.

        Called exactly once after login, before event dispatch. Guild-scoped sync
        propagates instantly; bare global sync can take up to 1 hour.
        """
        guild = discord.Object(id=get_settings().discord_guild_id)
        for module in COG_MODULES:
            await self.load_extension(module)
            logger.info("cog_loaded", module=module)

        self.tree.copy_global_to(guild=guild)
        await self.tree.sync(guild=guild)
        logger.info("command_tree_synced", guild_id=get_settings().discord_guild_id)

        # Team-logo emoji cache (260627-wt5): fetch the application's custom emojis
        # ONCE at startup and populate the team_emoji cache so chat lines can be
        # decorated with team logos — with NO hardcoded emoji ids. Best-effort: a
        # fetch failure logs a warning and continues (the resolver then returns None
        # and chat lines simply post undecorated). The emojis are static, so a single
        # fetch is sufficient — no re-fetch loop.
        try:
            from app.bot.team_emoji import populate_emoji_cache

            emojis = await self.fetch_application_emojis()
            count = populate_emoji_cache(emojis)
            logger.info("team_emojis_cached", count=count)
        except Exception:
            logger.warning("team_emoji_fetch_failed", exc_info=True)

        # Start the gateway-aware heartbeat loop.
        self._heartbeat_loop.start()

        # Start the guild avatar sweep loop. Its before_loop waits until ready so
        # the first tick (the "on startup" sweep) iterates a populated member cache.
        self._avatar_sweep_loop.start()

        # Start the Redis event subscriber as a fire-and-forget background task.
        # It must NOT block setup_hook; one bad event never kills the loop (the
        # subscriber is resilient per-message). Keep a reference so close() can
        # cancel it cleanly, mirroring the heartbeat loop.
        from app.bot.notifier import run_notifier

        self._notifier_task = asyncio.create_task(run_notifier(self))

    async def on_ready(self) -> None:
        """Log a ready line — informational only; init lives in setup_hook."""
        logger.info(
            "bot_ready",
            user=str(self.user),
            guild_count=len(self.guilds),
        )

    @tasks.loop(seconds=15)
    async def _heartbeat_loop(self) -> None:
        """Touch the heartbeat file only while the gateway is live.

        The compose healthcheck fails when the mtime goes stale — this detects a
        dropped gateway, not just a crashed process.
        """
        if self.is_ready() and not self.is_closed():
            HEARTBEAT_FILE.touch()

    @_heartbeat_loop.before_loop
    async def _heartbeat_before(self) -> None:
        await self.wait_until_ready()

    @tasks.loop(minutes=get_settings().discord_avatar_sweep_minutes)
    async def _avatar_sweep_loop(self) -> None:
        """Upsert every guild member's current Discord avatar hash, keyed by id.

        Runs once at startup (first tick after ready) and every
        ``discord_avatar_sweep_minutes`` thereafter. Resolves the configured guild
        from the local cache via ``get_guild``; on a cache miss it skips this tick
        (the next tick after the cache warms covers it) rather than fetching per
        tick. Iterates ``guild.members`` (the ``members`` privileged intent is
        already enabled in main()), computing ``member.avatar.key`` (None for a
        default avatar) and upserting through db_bridge.

        Best-effort: the whole sweep is wrapped so one bad tick logs a warning and
        never kills the loop (mirrors the team-emoji posture).
        """
        try:
            guild = self.get_guild(get_settings().discord_guild_id)
            if guild is None:
                # Cache not warm yet — skip; a later tick will cover it.
                logger.warning("avatar_sweep_guild_cache_miss")
                return
            swept = 0
            for member in guild.members:
                avatar_hash = member.avatar.key if member.avatar else None
                await db_bridge.upsert_avatar_hash_async(member.id, avatar_hash)
                swept += 1
            logger.info("avatar_sweep_complete", swept=swept)
        except Exception:
            logger.warning("avatar_sweep_failed", exc_info=True)

    @_avatar_sweep_loop.before_loop
    async def _avatar_sweep_before(self) -> None:
        # Wait until ready so the member cache is populated AND the startup tick
        # iterates real members (this is the "on startup" sweep).
        await self.wait_until_ready()

    async def close(self) -> None:
        """Cancel background tasks on close (heartbeat loop + avatar sweep + event subscriber)."""
        self._heartbeat_loop.cancel()
        self._avatar_sweep_loop.cancel()
        task = getattr(self, "_notifier_task", None)
        if task is not None:
            task.cancel()
        await super().close()


async def main() -> None:
    """Fail-fast async entrypoint for the bot container.

    Raises RuntimeError if discord_bot_token or discord_guild_id are None —
    non-bot containers (API, worker, beat) leave them unset and must not start
    the bot. Explicit SIGTERM/SIGINT handlers are installed on the running event
    loop via loop.add_signal_handler(); each fires asyncio.create_task(bot.close())
    so the gateway disconnects cleanly and bot.start() returns — discord.py does
    NOT install signal handlers on the await bot.start() path.
    """
    settings = get_settings()
    configure_logging(settings.log_level)

    if settings.discord_bot_token is None:
        raise RuntimeError("DISCORD_BOT_TOKEN required for the bot container")
    if settings.discord_guild_id is None:
        raise RuntimeError("DISCORD_GUILD_ID required for the bot container")

    # Touch the heartbeat file once at process start so the compose healthcheck
    # has a baseline before the gateway becomes ready.
    HEARTBEAT_FILE.touch()

    # Minimal intents only: guilds + dm_messages + members + message_content
    intents = discord.Intents.none()
    intents.guilds = True
    intents.dm_messages = True
    intents.members = True  # privileged — must be toggled in Developer Portal
    # PRIVILEGED: required by the inbound @mention Q&A listener (on_message) to read
    # message text. Like `members`, it MUST also be toggled ON in the Developer
    # Portal (Bot -> Privileged Gateway Intents -> Message Content Intent) — enabling
    # it here alone is not enough (see user_setup in the plan / SUMMARY).
    intents.message_content = True

    bot = PickemBot(command_prefix="!", intents=intents)  # prefix unused; slash-only
    async with bot:
        # Install explicit SIGTERM/SIGINT handlers before starting the gateway.
        # asyncio.run() only handles SIGINT, and discord.py's signal handling only
        # activates via Client.run() — not via await bot.start(). As PID 1 in the
        # container the kernel ignores unhandled signals, so without these handlers
        # `docker compose stop` would SIGKILL after the grace period (exit 137).
        loop = asyncio.get_running_loop()
        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(
                    sig,
                    lambda: asyncio.create_task(bot.close()),
                )
        except NotImplementedError:
            # add_signal_handler is only available on Unix with the default event
            # loop; log and continue so the bot still starts in unusual environments.
            logger.warning("signal_handler_registration_skipped", reason="NotImplementedError")

        await bot.start(settings.discord_bot_token)


if __name__ == "__main__":
    asyncio.run(main())
