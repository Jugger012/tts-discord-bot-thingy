"""
bot/config.py — Centralised settings loaded from environment / .env file.

All secrets and tuneable parameters live here. Never hardcode values elsewhere.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Discord ────────────────────────────────────────────────────────────────
    discord_token: str = Field(..., description="Discord bot token")
    discord_guild_id: Optional[int] = Field(
        None, description="Restrict slash-command sync to a single guild (dev)"
    )

    @field_validator("discord_token", mode="before")
    @classmethod
    def _sanitize_discord_token(cls, v: str) -> str:
        if isinstance(v, str) and (v.startswith("$") or v.startswith("your_")):
            from dotenv import dotenv_values
            token = dotenv_values(".env").get("DISCORD_TOKEN")
            if token and not token.startswith("$") and not token.startswith("your_"):
                return token.strip()
        return v.strip() if isinstance(v, str) else v

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_backend: Literal["gemini", "openai"] = Field("gemini")
    llm_model: str = Field("gemini-3.8-flash")
    gemini_api_key: Optional[str] = Field(None)
    openai_api_key: Optional[str] = Field(None)

    # ── TTS ───────────────────────────────────────────────────────────────────
    tts_backend: Literal["xtts", "f5", "elevenlabs", "cartesia"] = Field("xtts")
    elevenlabs_api_key: Optional[str] = Field(None)
    elevenlabs_voice_id: Optional[str] = Field(None)
    cartesia_api_key: Optional[str] = Field(None)
    cartesia_voice_id: Optional[str] = Field(None)

    # ── STT ───────────────────────────────────────────────────────────────────
    stt_backend: Literal["faster-whisper", "openai"] = Field("faster-whisper")
    whisper_model_size: str = Field("base")
    whisper_device: Literal["cpu", "cuda", "auto"] = Field("auto")

    # ── Audio Reference ────────────────────────────────────────────────────────
    reference_audio_path: Path = Field(Path("assets/reference_voice.wav"))

    # ── Context / Memory ──────────────────────────────────────────────────────
    context_window_size: int = Field(20, ge=1, le=100)

    # ── VAD ───────────────────────────────────────────────────────────────────
    vad_silence_threshold_ms: int = Field(1200, ge=300, le=5000)
    vad_aggressiveness: int = Field(2, ge=0, le=3)

    # ── Persona ────────────────────────────────────────────────────────────────
    bot_name: str = Field("Aria")
    bot_persona: str = Field(
        "You are Aria, a witty, concise, and friendly AI assistant. "
        "You speak naturally and conversationally, as if talking to a friend. "
        "Keep responses short and punchy — no more than 2-3 sentences unless "
        "specifically asked for detail."
    )

    # ── Hybrid Mode ───────────────────────────────────────────────────────────
    hybrid_text_channel_id: Optional[int] = Field(None)

    # ── Playback ──────────────────────────────────────────────────────────────
    playback_volume: float = Field(0.9, ge=0.0, le=2.0)

    @field_validator("reference_audio_path", mode="before")
    @classmethod
    def resolve_path(cls, v: object) -> Path:
        return Path(str(v))


# Singleton — import and use `settings` everywhere
settings = Settings()  # type: ignore[call-arg]
