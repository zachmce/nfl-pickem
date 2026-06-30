"""AdminCog — Discord slash-command cog for /admin group + 4 subcommands.

The /admin group exposes:
  /admin deactivate  @member  — deactivates a member's account
  /admin reactivate  @member  — reactivates a member's account
  /admin grant-admin @member  — grants admin rights
  /admin revoke-admin @member — revokes admin rights (with self-demote guard)

Locked decisions:
  - DB is_admin_async is the SOLE authorization check on every subcommand.
    Discord roles/permissions are NEVER trusted for admin authorization.
  - /admin app_commands.Group with 4 subcommands taking discord.Member.
  - revoke-admin passes (caller_discord_id, target_discord_id) to
    revoke_admin_async — both Discord snowflakes, caller FIRST.
  - every send_message call uses ephemeral=True.
"""

from __future__ import annotations

import discord
import structlog
from discord import app_commands
from discord.ext import commands

from app.bot import db_bridge

logger = structlog.get_logger(__name__)


class AdminCog(commands.Cog):
    """Handles the /admin command group for pick'em user management."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    # app_commands.Group declared at class scope (attaches to the cog automatically)
    admin = app_commands.Group(name="admin", description="Admin user management")

    @admin.command(name="deactivate", description="Deactivate a player's account")
    async def deactivate(self, interaction: discord.Interaction, member: discord.Member) -> None:
        """DB is_admin gate; deactivate_user_async with member.id."""
        if not await db_bridge.is_admin_async(interaction.user.id):
            await interaction.response.send_message(
                "Not authorized.",
                ephemeral=True,
            )
            return
        await db_bridge.deactivate_user_async(member.id)
        await interaction.response.send_message(
            f"Deactivated {member.mention}.",
            ephemeral=True,
        )

    @admin.command(name="reactivate", description="Reactivate a player's account")
    async def reactivate(self, interaction: discord.Interaction, member: discord.Member) -> None:
        """DB is_admin gate; reactivate_user_async with member.id."""
        if not await db_bridge.is_admin_async(interaction.user.id):
            await interaction.response.send_message(
                "Not authorized.",
                ephemeral=True,
            )
            return
        await db_bridge.reactivate_user_async(member.id)
        await interaction.response.send_message(
            f"Reactivated {member.mention}.",
            ephemeral=True,
        )

    @admin.command(name="grant-admin", description="Grant admin rights to a player")
    async def grant_admin(self, interaction: discord.Interaction, member: discord.Member) -> None:
        """DB is_admin gate; grant_admin_async with member.id."""
        if not await db_bridge.is_admin_async(interaction.user.id):
            await interaction.response.send_message(
                "Not authorized.",
                ephemeral=True,
            )
            return
        await db_bridge.grant_admin_async(member.id)
        await interaction.response.send_message(
            f"Granted admin to {member.mention}.",
            ephemeral=True,
        )

    @admin.command(name="revoke-admin", description="Revoke admin rights from a player")
    async def revoke_admin(self, interaction: discord.Interaction, member: discord.Member) -> None:
        """DB is_admin gate; revoke_admin_async(caller_id, target_id).

        Caller's discord_id is passed FIRST (Discord snowflakes, not pickem user
        IDs). The self-demote guard raises ValueError when caller == target; the
        cog_app_command_error sink surfaces it ephemerally.
        """
        if not await db_bridge.is_admin_async(interaction.user.id):
            await interaction.response.send_message(
                "Not authorized.",
                ephemeral=True,
            )
            return
        # caller discord_id FIRST, target discord_id SECOND
        await db_bridge.revoke_admin_async(interaction.user.id, member.id)
        await interaction.response.send_message(
            f"Revoked admin from {member.mention}.",
            ephemeral=True,
        )

    async def cog_app_command_error(
        self,
        interaction: discord.Interaction,
        error: app_commands.AppCommandError,
    ) -> None:
        """Cog-level error sink — maps errors to ephemeral user-facing replies.

        CommandInvokeError wrapping a ValueError (e.g. self-demote guard, target
        not found) is surfaced as the service message. Never leaks a traceback.
        """
        if isinstance(error, app_commands.CommandInvokeError) and isinstance(
            error.original, ValueError
        ):
            msg = str(error.original)
        elif isinstance(error, ValueError):
            msg = str(error)
        else:
            logger.warning("unhandled_admin_cog_error", error=str(error))
            msg = "Something went wrong. Try again later."

        if interaction.response.is_done():
            await interaction.followup.send(msg, ephemeral=True)
        else:
            await interaction.response.send_message(msg, ephemeral=True)


async def setup(bot: commands.Bot) -> None:
    """Required by load_extension — registers AdminCog with the bot."""
    await bot.add_cog(AdminCog(bot))
