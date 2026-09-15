"""
bot/cogs/voice_listener.py — Voice-to-Voice active listening pipeline.

This cog:
  1. Attaches the VoiceActivitySink when the bot joins a voice channel.
  2. Receives ``speech_finished`` callbacks from the sink.
  3. Transcribes the captured audio.
  4. Appends to conversation memory.
  5. Streams an LLM response sentence-by-sentence.
  6. Synthesises each sentence via TTS and enqueues for playback.

Echo prevention: the bot's own Discord user ID is passed to the sink so its
own audio stream is never transcribed.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import discord
from discord.ext import commands

from bot.brain.llm_client import get_llm_client
from bot.brain.memory import ConversationMemory
from bot.brain.prompts import build_system_prompt
from bot.config import settings
from bot.voice.sink import VoiceActivitySink
from bot.voice.transcriber import get_transcriber

log = logging.getLogger(__name__)


class VoiceListener(commands.Cog):
    """
    Manages voice capture, transcription, and LLM→TTS response pipeline.
    """

    def __init__(self, bot: discord.Bot) -> None:
        self.bot = bot
        self._sink: Optional[VoiceActivitySink] = None
        self._transcriber = get_transcriber(
            backend=settings.stt_backend,
            model_size=settings.whisper_model_size,
            device=settings.whisper_device,
            openai_api_key=settings.openai_api_key,
        )
        self._llm = get_llm_client(
            backend=settings.llm_backend,
            gemini_api_key=settings.gemini_api_key,
            openai_api_key=settings.openai_api_key,
            model=settings.llm_model,
        )
        # _responding guards against concurrent LLM calls
        self._responding = asyncio.Lock()

    # ── Voice state tracking ──────────────────────────────────────────────────

    @commands.Cog.listener("on_voice_state_update")
    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        """Attach the sink when the bot joins a voice channel."""
        if member.id != self.bot.user.id:
            return  # Only care about bot's own state changes

        # Bot just joined a channel
        if after.channel is not None and before.channel != after.channel:
            await self._attach_sink(after.channel.guild)

        # Bot left / was disconnected
        if after.channel is None and before.channel is not None:
            if self._sink:
                self._sink.cleanup()
                self._sink = None
            log.info("VoiceListener: sink cleaned up after disconnect.")

    async def _attach_sink(self, guild: discord.Guild) -> None:
        """Create and attach a fresh VoiceActivitySink to the guild's VC."""
        vc = guild.voice_client
        if vc is None:
            return

        # Wait briefly for voice handshake to complete if just connecting
        for _ in range(10):
            if vc.is_connected():
                break
            await asyncio.sleep(0.5)

        if not vc.is_connected():
            log.warning("VoiceClient not connected after waiting; cannot attach sink.")
            return

        # `.recording` is py-cord only; use `is_recording()` with a fallback
        # so this works on both discord.py and py-cord without crashing.
        if callable(getattr(vc, "is_recording", None)):
            if vc.is_recording():
                return
        elif getattr(vc, "recording", False):
            return

        bot_user_ids = {self.bot.user.id}
        self._sink = VoiceActivitySink(
            speech_callback=self._on_speech_finished,
            silence_threshold_ms=settings.vad_silence_threshold_ms,
            vad_aggressiveness=settings.vad_aggressiveness,
            bot_user_ids=bot_user_ids,
        )
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                vc.start_recording(self._sink, self._on_recording_finished)
            log.info("VoiceActivitySink attached in guild: %s", guild.name)
        except Exception as exc:
            log.warning("Voice recording could not be started: %s", exc)

    def _on_recording_finished(self, exception: Exception | None) -> None:
        """py-cord 2.7+ callback when stop_recording is called. Takes a single exception arg."""
        if exception:
            log.warning("Recording finished with error: %s", exception)
        else:
            log.debug("Recording stopped cleanly.")


    # ── Speech pipeline ───────────────────────────────────────────────────────

    async def _on_speech_finished(self, user_id: int, pcm_16k_mono: bytes) -> None:
        """
        Called by VoiceActivitySink when a user finishes speaking.

        Full pipeline:
          PCM bytes → transcribe → memory → LLM stream → TTS per sentence → enqueue
        """
        voice_state: dict | None = getattr(self.bot, "voice_state", None)
        if voice_state is None:
            return
        if voice_state.get("mode") != "voice":
            return  # hybrid text mode — skip voice processing

        queue = voice_state.get("playback_queue")
        if queue is None:
            return

        # Step 1: Transcribe
        text = await self._transcriber.transcribe(pcm_16k_mono)
        if not text:
            log.debug("Empty transcription for user %s; skipping.", user_id)
            return

        # Resolve username from guild member cache
        guild = next(
            (g for g in self.bot.guilds if g.voice_client is not None), None
        )
        member = guild.get_member(user_id) if guild else None
        username = member.display_name if member else f"User_{user_id}"

        log.info("Transcribed [%s]: %r", username, text)

        # Step 2: Update memory
        memory: ConversationMemory = getattr(self.bot, "memory", ConversationMemory(
            max_turns=settings.context_window_size
        ))
        if not hasattr(self.bot, "memory"):
            self.bot.memory = memory  # type: ignore[attr-defined]
        memory.add_user_message(user_id=user_id, username=username, text=text)

        # Step 3: Guard against overlapping LLM calls
        if self._responding.locked():
            log.debug("LLM already responding; queuing user message only.")
            return

        async with self._responding:
            guild_name = guild.name if guild else None
            channel_name = (
                guild.voice_client.channel.name
                if guild and guild.voice_client
                else None
            )
            system_prompt = build_system_prompt(
                bot_name=settings.bot_name,
                persona=settings.bot_persona,
                guild_name=guild_name,
                channel_name=channel_name,
            )

            tts = voice_state.get("tts")
            full_response_parts: list[str] = []

            # Step 4–5: Stream LLM → TTS → enqueue
            try:
                async for sentence in self._llm.stream_sentences(
                    system_prompt=system_prompt,
                    messages=memory.get_context_messages(),
                    new_user_text=text,
                ):
                    if not sentence:
                        continue
                    full_response_parts.append(sentence)
                    try:
                        audio_path = await tts.generate_audio(
                            text=sentence,
                            reference_path=str(settings.reference_audio_path),
                        )
                        queue.enqueue(audio_path)
                    except Exception as exc:
                        log.exception(
                            "TTS synthesis failed for sentence %r: %s", sentence, exc
                        )
            except Exception as exc:
                log.exception("LLM streaming failed: %s", exc)
                return

            # Step 6: Store assistant response in memory
            full_response = " ".join(full_response_parts)
            if full_response:
                memory.add_assistant_message(
                    text=full_response, bot_name=settings.bot_name
                )
                log.info("Response [%s]: %r", settings.bot_name, full_response)


def setup(bot: discord.Bot) -> None:
    bot.add_cog(VoiceListener(bot))
