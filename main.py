"""
main.py — Bot entry point.

Loads configuration, registers all cogs, and starts the py-cord bot.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import discord

from bot.config import settings

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("discord-voice-bot")


def create_bot() -> discord.Bot:
    """Instantiate the bot with the correct intents."""
    intents = discord.Intents.default()
    intents.message_content = True   # needed for text-channel hybrid mode
    intents.voice_states = True       # needed to track who is in voice channels

    debug_guilds = [settings.discord_guild_id] if settings.discord_guild_id else None
    bot = discord.Bot(intents=intents, debug_guilds=debug_guilds)
    return bot


async def load_cogs(bot: discord.Bot) -> None:
    """Dynamically load all cog modules."""
    cog_paths = [
        "bot.cogs.voice_commands",
        "bot.cogs.voice_listener",
        "bot.cogs.text_listener",
    ]
    for path in cog_paths:
        try:
            bot.load_extension(path)
            log.info("Loaded cog: %s", path)
        except Exception as exc:
            log.exception("Failed to load cog %s: %s", path, exc)
            raise



async def main() -> None:
    # Ensure assets directory exists
    Path("assets").mkdir(exist_ok=True)

    bot = create_bot()

    @bot.event  # type: ignore[misc]
    async def on_ready() -> None:
        log.info("[OK] Logged in as %s (ID: %s)", bot.user, bot.application_id)
        log.info("TTS backend: %s | STT backend: %s | LLM: %s (%s)",
                 settings.tts_backend, settings.stt_backend,
                 settings.llm_backend, settings.llm_model)
        if bot.guilds:
            for guild in bot.guilds:
                log.info("Connected to guild: %r (ID: %s)", guild.name, guild.id)
            if not settings.discord_guild_id:
                log.info(
                    "💡 Tip: Set DISCORD_GUILD_ID=%s in .env for instant slash command sync during development.",
                    bot.guilds[0].id,
                )

    await load_cogs(bot)
    await bot.start(settings.discord_token)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Shutting down.")
