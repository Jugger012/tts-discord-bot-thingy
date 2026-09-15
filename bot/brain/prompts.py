"""
bot/brain/prompts.py — System prompt construction and voice-output guardrails.

The system prompt is dynamically assembled from:
  1. A persona block (name + custom personality text from .env)
  2. Hard voice-output rules injected on every turn
  3. Contextual channel/server info when available
"""
from __future__ import annotations


# ── Voice-Output Guardrails ───────────────────────────────────────────────────
# These rules are injected into EVERY system prompt to ensure the LLM generates
# text that is pleasant and natural when spoken aloud by the TTS engine.

VOICE_OUTPUT_RULES = """
## Voice Output Rules (CRITICAL — always follow these)
- Your response will be read aloud via a text-to-speech engine. Format accordingly.
- NEVER use markdown: no asterisks, pound signs, backticks, bullets, or numbered lists.
- NEVER include URLs, file paths, or code blocks.
- Keep every response SHORT and PUNCHY — 1 to 3 sentences unless the user explicitly requests detail.
- Use natural spoken language: contractions, casual phrasing, pauses implied by commas.
- If you don't know something, say so briefly and naturally.
- NEVER repeat the username or greet the user every turn — it sounds robotic.
- Avoid filler phrases like "Certainly!", "Of course!", "Great question!".
""".strip()


def build_system_prompt(
    bot_name: str,
    persona: str,
    guild_name: str | None = None,
    channel_name: str | None = None,
) -> str:
    """
    Assemble the full system prompt for the LLM.

    Parameters
    ----------
    bot_name : str
        The bot's display name (used to identify the assistant role).
    persona : str
        Free-form personality description from .env ``BOT_PERSONA``.
    guild_name : str | None
        Discord server name for contextual grounding.
    channel_name : str | None
        Voice/text channel name for contextual grounding.
    """
    context_block = ""
    if guild_name or channel_name:
        parts = []
        if guild_name:
            parts.append(f"Server: {guild_name}")
        if channel_name:
            parts.append(f"Channel: {channel_name}")
        context_block = "\n## Context\n" + " | ".join(parts)

    return f"""You are {bot_name}, an AI assistant participating in a live Discord voice conversation.

## Persona
{persona}
{context_block}

{VOICE_OUTPUT_RULES}
""".strip()


def build_user_turn(username: str, text: str) -> str:
    """Format a user utterance for injection into the message history."""
    return f"[{username}]: {text}"
