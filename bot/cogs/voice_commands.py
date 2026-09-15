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

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}


def get_audio_file_choices(ctx: discord.AutocompleteContext) -> list[str]:
    """Provide slash command autocomplete options for local audio files."""
    choices: list[str] = []
    search_dirs = [Path("Sound Example"), Path("assets")]
    for d in search_dirs:
        if d.exists():
            for f in d.iterdir():
                if f.is_file() and f.suffix.lower() in ALLOWED_AUDIO_EXTENSIONS:
                    choices.append(f.as_posix())
    val = (ctx.value or "").lower()
    return [c for c in choices if val in c.lower()][:25]


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

    # ── /setvoice & Voice Reference Handlers ──────────────────────────────────

    def _finalize_audio(self, raw_path: Path, source_name: str) -> tuple[bool, str]:
        """Processes raw audio using prepare_reference.py and updates settings."""
        try:
            from scripts.prepare_reference import process_audio
            out_path = process_audio(str(raw_path), str(settings.reference_audio_path))
            log.info("Voice reference updated from %s -> %s", source_name, out_path)
            return True, (
                f"✅ **Voice reference updated from `{source_name}`!**\n"
                f"Cleaned, normalised, and saved as `{out_path}`."
            )
        except Exception as exc:
            log.exception("Audio processing failed: %s", exc)
            shutil.copy2(str(raw_path), str(settings.reference_audio_path))
            return True, (
                f"⚠️ Auto-processing encountered an issue (`{exc}`).\n"
                f"Raw file saved directly as `{settings.reference_audio_path}`."
            )

    async def _handle_setvoice_source(
        self,
        file: discord.Attachment | None = None,
        path: str | None = None,
    ) -> tuple[bool, str]:
        """
        Processes an audio file from an uploaded attachment or path/URL.
        Returns (success: bool, message: str).
        """
        assets_dir = Path("assets")
        assets_dir.mkdir(exist_ok=True)

        if not file and not path:
            local_files = [
                f.as_posix()
                for d in [Path("Sound Example"), Path("assets")]
                if d.exists()
                for f in d.iterdir()
                if f.is_file() and f.suffix.lower() in ALLOWED_AUDIO_EXTENSIONS
            ]
            files_str = "\n".join(f"• `{f}`" for f in local_files) if local_files else "• None found"
            return False, (
                "ℹ️ **No audio file or path provided!**\n\n"
                "You can update the reference voice using any of these methods:\n"
                "1. **Upload file**: `/setvoice file:` (select a `.wav`, `.mp3`, `.m4a`, `.ogg`, or `.flac` file)\n"
                "2. **Local path or URL**: `/setvoice path:` (e.g., `Sound Example/0809.mp3` or a web URL)\n"
                "3. **Right-click an audio in chat**: Apps → **Set as Voice Reference**\n"
                "4. **Chat message**: Upload an audio file with `!setvoice` in your message\n\n"
                f"**Available local sample files:**\n{files_str}"
            )

        # ── Case 1: Attachment provided
        if file is not None:
            suffix = Path(file.filename).suffix.lower()
            if suffix not in ALLOWED_AUDIO_EXTENSIONS:
                return False, f"❌ Unsupported file type `{suffix}`. Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}"

            raw_path = assets_dir / f"reference_raw{suffix}"
            try:
                await file.save(raw_path)
            except Exception:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(file.url) as resp:
                        resp.raise_for_status()
                        raw_path.write_bytes(await resp.read())

            log.info("Downloaded reference audio attachment to: %s", raw_path)
            return self._finalize_audio(raw_path, file.filename)

        # ── Case 2: Path or URL provided
        assert path is not None
        clean_path = path.strip().strip('"\'')
        if clean_path.startswith(("http://", "https://")):
            import aiohttp
            url_suffix = Path(clean_path.split("?")[0]).suffix.lower()
            if url_suffix not in ALLOWED_AUDIO_EXTENSIONS:
                url_suffix = ".mp3"
            raw_path = assets_dir / f"reference_raw{url_suffix}"
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(clean_path) as resp:
                        resp.raise_for_status()
                        raw_path.write_bytes(await resp.read())
            except Exception as exc:
                return False, f"❌ Failed to download audio from URL: {exc}"
            return self._finalize_audio(raw_path, clean_path)

        # Local file path resolution
        candidates = [
            Path(clean_path),
            Path("Sound Example") / clean_path,
            Path("assets") / clean_path,
        ]
        found_path: Path | None = None
        for cand in candidates:
            if cand.exists() and cand.is_file():
                found_path = cand
                break

        # Fallback: search by filename in Sound Example and assets
        if not found_path:
            for d in [Path("Sound Example"), Path("assets")]:
                if d.exists():
                    for f in d.iterdir():
                        if f.is_file() and f.name.lower() == Path(clean_path).name.lower():
                            found_path = f
                            break
                if found_path:
                    break

        if not found_path:
            local_files = [
                f.as_posix()
                for d in [Path("Sound Example"), Path("assets")]
                if d.exists()
                for f in d.iterdir()
                if f.is_file() and f.suffix.lower() in ALLOWED_AUDIO_EXTENSIONS
            ]
            files_str = "\n".join(f"• `{f}`" for f in local_files) if local_files else "• None found"
            return False, (
                f"❌ Local audio file not found: `{clean_path}`.\n\n"
                f"**Available local audio files:**\n{files_str}"
            )

        suffix = found_path.suffix.lower()
        if suffix not in ALLOWED_AUDIO_EXTENSIONS:
            return False, f"❌ Unsupported file type `{suffix}`. Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}"

        raw_path = assets_dir / f"reference_raw{suffix}"
        shutil.copy2(str(found_path), str(raw_path))
        return self._finalize_audio(raw_path, found_path.name)

    @discord.slash_command(
        name="setvoice",
        description="Set voice cloning reference audio from uploaded file, local path, or URL",
    )
    @option(
        "file",
        discord.Attachment,
        description="Audio file to upload (.wav, .mp3, .m4a, .ogg, .flac)",
        required=False,
    )
    @option(
        "path",
        str,
        description="Local file path, filename in Sound Example/assets, or audio URL",
        required=False,
        autocomplete=discord.utils.basic_autocomplete(get_audio_file_choices),
    )
    async def setvoice(
        self,
        ctx: discord.ApplicationContext,
        file: discord.Attachment | None = None,
        path: str | None = None,
    ) -> None:
        await ctx.defer(ephemeral=True)
        ok, msg = await self._handle_setvoice_source(file=file, path=path)
        await ctx.followup.send(msg)

    @discord.message_command(name="Set as Voice Reference")
    async def setvoice_message(
        self,
        ctx: discord.ApplicationContext,
        message: discord.Message,
    ) -> None:
        """Right-click any message with an audio file to set it as voice reference."""
        await ctx.defer(ephemeral=True)
        audio_attachment = None
        for att in message.attachments:
            if Path(att.filename).suffix.lower() in ALLOWED_AUDIO_EXTENSIONS:
                audio_attachment = att
                break
        if not audio_attachment:
            await ctx.followup.send(
                "❌ This message does not have a supported audio attachment "
                f"({', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))})."
            )
            return

        ok, msg = await self._handle_setvoice_source(file=audio_attachment)
        await ctx.followup.send(msg)

    @commands.Cog.listener("on_message")
    async def on_voice_command_message(self, message: discord.Message) -> None:
        """Fallback for users who send !setvoice or /setvoice as a regular message."""
        if message.author.bot:
            return
        content = message.content.strip()
        if not (content.startswith("!setvoice") or content.startswith("/setvoice")):
            return

        parts = content.split(maxsplit=1)
        path_arg = parts[1].strip() if len(parts) > 1 else None

        audio_att = None
        for att in message.attachments:
            if Path(att.filename).suffix.lower() in ALLOWED_AUDIO_EXTENSIONS:
                audio_att = att
                break

        ok, msg = await self._handle_setvoice_source(file=audio_att, path=path_arg)
        await message.reply(msg)

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
