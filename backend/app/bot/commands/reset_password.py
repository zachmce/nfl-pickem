"""ResetPasswordCog — Discord slash-command cog for /reset-password.

Thin translation layer: gates on guild membership, calls reset_password_async
(raises ValueError for unknown/deactivated accounts), DMs the new temp password
with app_base_url framing, and sends an ephemeral in-channel confirmation.

  - interaction.response.defer(ephemeral=True) as first action; all replies use
    interaction.followup.send so slow ops don't blow the 3s window.
  - get_guild() None guard — falls back to fetch_guild() on cache miss.
  - explicit discord.Forbidden handler after DM — password already rotated so we
    can't roll back; tell the user to enable DMs and retry.
"""

from __future__ import annotations

import discord
import structlog
from discord import app_commands
from discord.ext import commands

from app.bot import db_bridge
from app.config import get_settings

logger = structlog.get_logger(__name__)


class ResetPasswordCog(commands.Cog):
    """Handles /reset-password for re-generating a member's pick'em password."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @app_commands.command(
        name="reset-password",
        description="Reset your pick'em account password",
    )
    async def reset_password(self, interaction: discord.Interaction) -> None:
        """/reset-password — re-generate and DM a new temporary password.

        Flow:
        1. Defer: ack the interaction immediately so slow ops don't expire it.
        2. Membership gate: fetch_member authoritative check; NotFound → followup + stop.
           If get_guild() returns None (cache miss), fall back to fetch_guild().
        3. reset_password_async: raises ValueError for unknown/deactivated → error sink.
        4. DM the new password with "temporary" framing + app_base_url.
           On discord.Forbidden (DMs closed), surface a clear followup telling the
           user to enable DMs and run /reset-password again for a fresh password.
           NOTE: the password is already committed at this point — a second reset will
           issue a new plaintext. The old one is gone; there is no rollback path.
        5. Ephemeral in-channel confirmation via followup.
        """
        # Defer immediately — argon2 hash + DB commit + DM round-trip can exceed 3s
        await interaction.response.defer(ephemeral=True)

        settings = get_settings()

        # Membership gate — authoritative API call (cache may be stale).
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
                "You must be a member of the server to reset your password.",
                ephemeral=True,
            )
            return

        # Reset password (raises ValueError on unknown/deactivated)
        new_pw = await db_bridge.reset_password_async(interaction.user.id)

        # DM the new password with temporary framing.
        # On discord.Forbidden — DMs closed means the rotated password was never
        # delivered. We cannot un-rotate (the new hash is already committed), so we
        # surface a clear actionable message.
        try:
            await interaction.user.send(
                f"Your pick'em password has been reset.\n"
                f"Temporary password: `{new_pw}`\n"
                f"This is a **temporary password** — please change it at "
                f"{settings.app_base_url} after logging in."
            )
        except discord.Forbidden:
            logger.warning(
                "reset_password_dm_forbidden",
                discord_id=interaction.user.id,
            )
            await interaction.followup.send(
                "Your password was reset but I couldn't send you a DM. "
                "Enable DMs from server members and run /reset-password again "
                "to receive a fresh password.",
                ephemeral=True,
            )
            return

        # Ephemeral in-channel confirm via followup
        await interaction.followup.send(
            "Password reset — check your DMs.",
            ephemeral=True,
        )

    async def cog_app_command_error(
        self,
        interaction: discord.Interaction,
        error: app_commands.AppCommandError,
    ) -> None:
        """Cog-level error sink — maps errors to ephemeral user-facing replies."""
        if isinstance(error, app_commands.CommandInvokeError) and isinstance(
            error.original, ValueError
        ):
            msg = str(error.original)
        elif isinstance(error, ValueError):
            msg = str(error)
        else:
            logger.warning("unhandled_reset_cog_error", error=str(error))
            msg = "Something went wrong. Try again later."

        if interaction.response.is_done():
            await interaction.followup.send(msg, ephemeral=True)
        else:
            await interaction.response.send_message(msg, ephemeral=True)


async def setup(bot: commands.Bot) -> None:
    """Required by load_extension — registers ResetPasswordCog with the bot."""
    await bot.add_cog(ResetPasswordCog(bot))
