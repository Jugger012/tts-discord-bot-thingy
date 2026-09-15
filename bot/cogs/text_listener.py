"""
bot/cogs/text_listener.py — Text-to-Voice hybrid mode listener.

When ``HYBRID_TEXT_CHANNEL_ID`` is configured and the bot is in 'text' mode,
every user message sent to that channel triggers an LLM response that is
synthesised in the cloned voice and played in the active voice channel.

Flow:
    User types in text channel
        → LLM generates response (sentence-streaming)
        → Each sentence → TTS synthesis → enqueued for playback
"""
from __future__ import annotations

import logging

import discord
from discord.ext import commands

from bot.brain.llm_client import get_llm_client
from bot.brain.memory import ConversationMemory
from bot.brain.prompts import build_system_prompt
from bot.config import settings

log = logging.getLogger(__name__)


class TextListener(commands.Cog):
    """Listens to a designated text channel and responds via voice."""

    def __init__(self, bot: discord.Bot) -> None:
        self.bot = bot
        self._memory: ConversationMemory = ConversationMemory(
            max_turns=settings.context_window_size
        )
        # Share memory with voice_listener via bot attribute
        self.bot.memory = self._memory  # type: ignore[attr-defined]

        self._llm = get_llm_client(
            backend=settings.llm_backend,
            gemini_api_key=settings.gemini_api_key,
            openai_api_key=settings.openai_api_key,
            model=settings.llm_model,
        )

    @commands.Cog.listener("on_message")
    async def on_message(self, message: discord.Message) -> None:
        # Guard clauses
        if message.author.bot:
            return

        voice_state: dict | None = getattr(self.bot, "voice_state", None)
        if voice_state is None or voice_state.get("mode") != "text":
            return

        target_channel_id = settings.hybrid_text_channel_id or voice_state.get("text_channel_id")
        if target_channel_id is not None and message.channel.id != target_channel_id:
            return

        queue = voice_state.get("playback_queue")
        if queue is None:
            log.debug("TextListener: no active playback queue, ignoring message.")
            return

        user_text = message.content.strip()
        if not user_text:
            return

        log.info(
            "TextListener: %s#%s said: %r",
            message.author.display_name,
            message.author.discriminator,
            user_text,
        )

        # Update memory
        self._memory.add_user_message(
            user_id=message.author.id,
            username=message.author.display_name,
            text=user_text,
        )

        guild_name = message.guild.name if message.guild else None
        channel_name = message.channel.name if hasattr(message.channel, "name") else None

        system_prompt = build_system_prompt(
            bot_name=settings.bot_name,
            persona=settings.bot_persona,
            guild_name=guild_name,
            channel_name=channel_name,
        )

        tts = voice_state.get("tts")
        full_response_parts: list[str] = []

        async with message.channel.typing():
            async for sentence in self._llm.stream_sentences(
                system_prompt=system_prompt,
                messages=self._memory.get_context_messages(),
                new_user_text=user_text,
            ):
                if not sentence:
                    continue
                full_response_parts.append(sentence)
                # Synthesise and enqueue each sentence immediately
                try:
                    audio_path = await tts.generate_audio(
                        text=sentence,
                        reference_path=str(settings.reference_audio_path),
                    )
                    queue.enqueue(audio_path)
                except Exception as exc:
                    log.exception("TTS failed for sentence %r: %s", sentence, exc)

        # Store the complete response in memory
        full_response = " ".join(full_response_parts)
        if full_response:
            self._memory.add_assistant_message(
                text=full_response, bot_name=settings.bot_name
            )


def setup(bot: discord.Bot) -> None:
    bot.add_cog(TextListener(bot))
