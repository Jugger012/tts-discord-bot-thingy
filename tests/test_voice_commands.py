"""
tests/test_voice_commands.py — Unit tests for /setvoice command logic.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import discord

from bot.cogs.voice_commands import VoiceCommands, get_audio_file_choices


@pytest.fixture
def dummy_cog():
    bot = MagicMock(spec=discord.Bot)
    with patch("bot.cogs.voice_commands.get_tts_adapter"):
        cog = VoiceCommands(bot)
    return cog


@pytest.mark.asyncio
async def test_setvoice_no_input_returns_guide(dummy_cog):
    ok, msg = await dummy_cog._handle_setvoice_source(file=None, path=None)
    assert not ok
    assert "No audio file or path provided" in msg


@pytest.mark.asyncio
async def test_setvoice_invalid_extension(dummy_cog):
    fake_attachment = MagicMock(spec=discord.Attachment)
    fake_attachment.filename = "song.txt"
    ok, msg = await dummy_cog._handle_setvoice_source(file=fake_attachment)
    assert not ok
    assert "Unsupported file type" in msg


@pytest.mark.asyncio
async def test_setvoice_local_path_not_found(dummy_cog):
    ok, msg = await dummy_cog._handle_setvoice_source(path="nonexistent_audio_file.wav")
    assert not ok
    assert "Local audio file not found" in msg


@pytest.mark.asyncio
async def test_setvoice_local_path_success(dummy_cog, tmp_path):
    sample_file = tmp_path / "sample.wav"
    sample_file.write_bytes(b"RIFF dummy wav data")

    with patch.object(dummy_cog, "_finalize_audio", return_value=(True, "Success!")):
        ok, msg = await dummy_cog._handle_setvoice_source(path=str(sample_file))
        assert ok
        assert msg == "Success!"


@pytest.mark.asyncio
async def test_setvoice_attachment_success(dummy_cog, tmp_path):
    fake_attachment = MagicMock(spec=discord.Attachment)
    fake_attachment.filename = "voice.mp3"
    fake_attachment.save = AsyncMock()

    with patch.object(dummy_cog, "_finalize_audio", return_value=(True, "Updated!")):
        ok, msg = await dummy_cog._handle_setvoice_source(file=fake_attachment)
        assert ok
        assert msg == "Updated!"
        fake_attachment.save.assert_awaited_once()


def test_get_audio_file_choices():
    ctx = MagicMock(spec=discord.AutocompleteContext)
    ctx.value = ""
    choices = get_audio_file_choices(ctx)
    assert isinstance(choices, list)


def test_get_premade_voice_choices():
    from bot.cogs.voice_commands import get_premade_voice_choices
    ctx = MagicMock(spec=discord.AutocompleteContext)
    ctx.value = "sar"
    choices = get_premade_voice_choices(ctx)
    assert any("Sarah" in c for c in choices)


@pytest.mark.asyncio
async def test_setvoice_by_name_success(dummy_cog):
    with patch("dotenv.set_key"):
        ok, msg = await dummy_cog._handle_setvoice_source(voice="George")
        assert ok
        assert "Voice switched to George" in msg
        assert "no restart needed" in msg
