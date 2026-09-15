"""
bot/cogs/voice_commands.py — Slash commands for voice channel lifecycle.

Commands
--------
/join       — Joins the invoking user's voice channel.
/leave      — Cleans up the audio sink and disconnects.
/setvoice   — Sets a new voice cloning reference audio (file attachment or path).
/mode       — Switches between voice-to-voice and text-to-voice hybrid mode.
/skip       — Skips the currently playing audio response.
/clearctx   — Wipes the conversation memory buffer.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import discord
from discord.ext import commands
from discord import option

from bot.config import settings
from bot.voice.playback import AudioPlaybackQueue
from bot.voice.tts_adapter import get_tts_adapter

log = logging.getLogger(__name__)


class VoiceCommands(commands.Cog):
    """Slash commands for managing the bot's voice channel participation."""

    def __init__(self, bot: discord.Bot) -> None:
        self.bot = bot
        # Shared state — accessed by other cogs via bot.voice_state
        self.bot.voice_state = {  # type: ignore[attr-defined]
            "playback_queue": None,   # AudioPlaybackQueue | None
            "mode": "voice",          # 'voice' | 'text'
            "tts": get_tts_adapter(
                backend=settings.tts_backend,
                elevenlabs_api_key=settings.elevenlabs_api_key,
                elevenlabs_voice_id=settings.elevenlabs_voice_id,
                cartesia_api_key=settings.cartesia_api_key,
                cartesia_voice_id=settings.cartesia_voice_id,
            ),
        }

    # ── /join ─────────────────────────────────────────────────────────────────

    @discord.slash_command(name="join", description="Join your current voice channel")
    async def join(self, ctx: discord.ApplicationContext) -> None:
        if ctx.author.voice is None:
            await ctx.respond("❌ You must be in a voice channel first.", ephemeral=True)
            return

        channel = ctx.author.voice.channel
        await ctx.defer()

        # Disconnect from any existing channel
        if ctx.guild.voice_client:
            await ctx.guild.voice_client.disconnect(force=True)

        vc = await channel.connect()
        log.info("Joined voice channel: %s", channel.name)

        # Create playback queue
        queue = AudioPlaybackQueue(vc, volume=settings.playback_volume)
        queue.start()
        self.bot.voice_state["playback_queue"] = queue  # type: ignore[attr-defined]

        await ctx.followup.send(
            f"✅ Joined **{channel.name}**. I'm listening! "
            f"(TTS: `{settings.tts_backend}` | LLM: `{settings.llm_model}`)"
        )

    # ── /leave ────────────────────────────────────────────────────────────────

    @discord.slash_command(name="leave", description="Leave the voice channel")
    async def leave(self, ctx: discord.ApplicationContext) -> None:
        vc = ctx.guild.voice_client
        if vc is None:
            await ctx.respond("❌ I'm not in a voice channel.", ephemeral=True)
            return

        queue: AudioPlaybackQueue | None = self.bot.voice_state.get("playback_queue")  # type: ignore[attr-defined]
        if queue:
            await queue.stop()
            self.bot.voice_state["playback_queue"] = None  # type: ignore[attr-defined]

        await vc.disconnect()
        log.info("Left voice channel in guild: %s", ctx.guild.name)
        await ctx.respond("👋 Disconnected and cleaned up.")

    # ── /setvoice ─────────────────────────────────────────────────────────────

    @discord.slash_command(
        name="setvoice",
        description="Set the voice cloning reference audio (attach a .wav/.mp3/.m4a file)",
    )
    async def setvoice(
        self,
        ctx: discord.ApplicationContext,
        attachment: discord.Attachment,
    ) -> None:
        await ctx.defer(ephemeral=True)

        # Validate file extension
        allowed = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
        suffix = Path(attachment.filename).suffix.lower()
        if suffix not in allowed:
            await ctx.followup.send(
                f"❌ Unsupported file type `{suffix}`. Allowed: {', '.join(allowed)}"
            )
            return

        # Download attachment
        assets_dir = Path("assets")
        assets_dir.mkdir(exist_ok=True)
        raw_path = assets_dir / f"reference_raw{suffix}"

        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(attachment.url) as resp:
                resp.raise_for_status()
                raw_path.write_bytes(await resp.read())

        log.info("Downloaded reference audio to: %s", raw_path)

        # Auto-process the audio using the prepare_reference script logic
        try:
            from scripts.prepare_reference import process_audio
            out_path = process_audio(str(raw_path), str(settings.reference_audio_path))
            await ctx.followup.send(
                f"✅ Voice reference updated from `{attachment.filename}`. "
                f"Saved as `{out_path}`."
            )
        except Exception as exc:
            log.exception("Audio processing failed: %s", exc)
            # Fall back: just copy the raw file
            shutil.copy2(str(raw_path), str(settings.reference_audio_path))
            await ctx.followup.send(
                f"⚠️ Auto-processing failed (`{exc}`). "
                f"Raw file saved as `{settings.reference_audio_path}`. "
                "Consider running `scripts/prepare_reference.py` manually."
            )

    # ── /mode ─────────────────────────────────────────────────────────────────

    @discord.slash_command(
        name="mode",
        description="Switch between voice-to-voice and text-to-voice hybrid mode",
    )
    @option(
        "mode",
        str,
        description="Listening mode",
        choices=["voice", "text"],
    )
    async def mode(self, ctx: discord.ApplicationContext, mode: str) -> None:
        self.bot.voice_state["mode"] = mode  # type: ignore[attr-defined]
        self.bot.voice_state["text_channel_id"] = ctx.channel.id  # type: ignore[attr-defined]
        emoji = "🎙️" if mode == "voice" else "💬"
        extra = f" (listening in <#{ctx.channel.id}>)" if mode == "text" else ""
        await ctx.respond(
            f"{emoji} Switched to **{mode}** mode{extra}.",
            ephemeral=True,
        )

    # ── /skip ─────────────────────────────────────────────────────────────────

    @discord.slash_command(name="skip", description="Skip the current audio response")
    async def skip(self, ctx: discord.ApplicationContext) -> None:
        queue: AudioPlaybackQueue | None = self.bot.voice_state.get("playback_queue")  # type: ignore[attr-defined]
        if queue:
            queue.skip()
            await ctx.respond("⏭️ Skipped.", ephemeral=True)
        else:
            await ctx.respond("Nothing is playing.", ephemeral=True)

    # ── /clearctx ─────────────────────────────────────────────────────────────

    @discord.slash_command(
        name="clearctx",
        description="Wipe the conversation history buffer",
    )
    async def clearctx(self, ctx: discord.ApplicationContext) -> None:
        memory = getattr(self.bot, "memory", None)
        if memory:
            memory.clear()
        await ctx.respond("🧹 Conversation history cleared.", ephemeral=True)


def setup(bot: discord.Bot) -> None:
    bot.add_cog(VoiceCommands(bot))
