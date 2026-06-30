"""RegistrationCog — Discord slash-command cog for /register.

This cog is a thin translation layer between discord.py interactions and the
db_bridge async wrappers. It contains no business logic — all validation,
commit, and user-derivation live in services/auth.py (via db_bridge).

Locked decisions implemented here:
  - commit-then-compensate: provision_user_async commits first; on discord.Forbidden
    from DM delivery, delete_user_async hard-deletes the row (no orphans).
  - read-first: get_account_async called before provision; already-registered
    users get an ephemeral pointer to their existing account.
  - per-user cooldown via @app_commands.checks.cooldown(1, 300.0, key=user.id).
  - membership gate via guild.fetch_member() authoritative API call.
  - DM copy frames password as "temporary" and links app_base_url.
  - every in-channel reply uses ephemeral=True.
  - interaction.response.defer(ephemeral=True) as first action; all replies
    use interaction.followup.send so slow ops don't blow the 3s ACK window.
  - get_guild() None guard — falls back to fetch_guild() on cache miss.
"""

from __future__ import annotations

import discord
import structlog
from discord import app_commands
from discord.ext import commands

from app.bot import db_bridge
from app.config import get_settings

logger = structlog.get_logger(__name__)


class RegistrationCog(commands.Cog):
    """Handles /register for new pick'em account creation."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @app_commands.command(name="register", description="Create your pick'em account")
    @app_commands.checks.cooldown(1, 300.0, key=lambda i: i.user.id)
    async def register(self, interaction: discord.Interaction) -> None:
        """/register — provision a new pick'em account for the invoking Discord member.

        Flow:
        1. Defer: ack the interaction immediately so slow ops don't expire it.
        2. Read-first: if an account already exists, followup pointer + stop.
        3. Membership gate: fetch_member authoritative check; NotFound → followup + stop.
           If get_guild() returns None (cache miss), fall back to fetch_guild().
        4. Provision: provision_user_async commits the row internally.
        5. DM: send temp password with app_base_url framing.
        6. Forbidden rollback: on DM failure, hard-delete the just-created row.
        7. Ephemeral in-channel confirm via followup.
        """
        # Defer immediately — argon2 hash + DB commit + DM round-trip can exceed 3s
        await interaction.response.defer(ephemeral=True)

        settings = get_settings()

        # Read-first — check for existing account before provisioning
        existing_name = await db_bridge.get_account_async(interaction.user.id)
        if existing_name is not None:
            await interaction.followup.send(
                f"You already have an account: **{existing_name}** — log in at {settings.app_base_url}",
                ephemeral=True,
            )
            return

        # Membership gate — authoritative API call (cache may be stale).
        # get_guild() reads the local guild cache and returns None on a cache miss.
        # Fall back to an API fetch so the membership gate never silently fails open.
        guild = interaction.client.get_guild(settings.discord_guild_id)
        if guild is None:
            try:
                guild = await interaction.client.fetch_guild(settings.discord_guild_id)
            except discord.HTTPException:
                await interaction.followup.send(
                    "Couldn't verify server membership right now — try again shortly.",
                    ephemeral=True,
                )
                return
        try:
            await guild.fetch_member(interaction.user.id)
        except discord.NotFound:
            await interaction.followup.send(
                "You must be a member of the server to register.",
                ephemeral=True,
            )
            return

        # Provision the account (commits internally). Capture the invoking
        # member's avatar hash inline — interaction.user.avatar is None when the
        # user has only a default avatar; .key is the bare hash we persist.
        avatar_hash = interaction.user.avatar.key if interaction.user.avatar else None
        user_id, display_name, plain = await db_bridge.provision_user_async(
            interaction.user.id, str(interaction.user), avatar_hash
        )

        # DM the temp password with app_base_url framing
        try:
            await interaction.user.send(
                f"Your pick'em username is **{display_name}**.\n"
                f"Temporary password: `{plain}`\n"
                f"This is a **temporary password** — please change it at "
                f"{settings.app_base_url} after logging in."
            )
        except discord.Forbidden:
            # DM closed — hard-delete the committed row (no orphaned accounts)
            await db_bridge.delete_user_async(user_id)
            await interaction.followup.send(
                "I couldn't send you a DM. Enable DMs from server members and run /register again.",
                ephemeral=True,
            )
            return

        # Ephemeral in-channel confirm via followup
        await interaction.followup.send(
            "Account created — check your DMs.",
            ephemeral=True,
        )

    async def cog_app_command_error(
        self,
        interaction: discord.Interaction,
        error: app_commands.AppCommandError,
    ) -> None:
        """Cog-level error sink — maps errors to ephemeral user-facing replies.

        Never surfaces a raw ValueError or traceback to the user. Branches on
        interaction.response.is_done() to avoid double-response.
        """
        if isinstance(error, app_commands.CommandOnCooldown):
            msg = f"Slow down — try again in {error.retry_after:.0f}s."
        elif isinstance(error, app_commands.CommandInvokeError) and isinstance(
            error.original, ValueError
        ):
            msg = str(error.original)
        elif isinstance(error, ValueError):
            msg = str(error)
        else:
            logger.warning("unhandled_cog_error", error=str(error))
            msg = "Something went wrong. Try again later."

        if interaction.response.is_done():
            await interaction.followup.send(msg, ephemeral=True)
        else:
            await interaction.response.send_message(msg, ephemeral=True)


async def setup(bot: commands.Bot) -> None:
    """Required by load_extension — registers RegistrationCog with the bot."""
    await bot.add_cog(RegistrationCog(bot))
