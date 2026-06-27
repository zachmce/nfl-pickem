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

        # Start the gateway-aware heartbeat loop.
        self._heartbeat_loop.start()

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

    async def close(self) -> None:
        """Cancel background tasks on close (heartbeat loop + event subscriber)."""
        self._heartbeat_loop.cancel()
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

    # Minimal intents only: guilds + dm_messages + members
    intents = discord.Intents.none()
    intents.guilds = True
    intents.dm_messages = True
    intents.members = True  # privileged — must be toggled in Developer Portal

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
