"""
tests/test_tts_adapter.py — Unit tests for TTS adapters using a mock backend.

Tests:
  - MockTTSAdapter produces a valid WAV file.
  - Output is 24kHz mono 16-bit WAV.
  - File cleanup works correctly.
  - Factory raises ValueError for unknown backends.
"""
from __future__ import annotations

import asyncio
import struct
import wave
from pathlib import Path

import pytest
import pytest_asyncio

from bot.voice.tts_adapter import BaseTTSAdapter, get_tts_adapter


# ── Mock adapter ──────────────────────────────────────────────────────────────


class MockTTSAdapter(BaseTTSAdapter):
    """
    Silent TTS adapter for testing — generates a 0.5s silent 24kHz mono WAV.
    """

    SAMPLE_RATE = 24_000
    DURATION_S = 0.5

    async def generate_audio(self, text: str, reference_path: str) -> Path:
        import tempfile

        n_samples = int(self.SAMPLE_RATE * self.DURATION_S)
        silence = struct.pack(f"<{n_samples}h", *([0] * n_samples))

        tmp = Path(tempfile.mktemp(suffix=".wav"))
        with wave.open(str(tmp), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)   # 16-bit
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(silence)

        return tmp


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mock_tts_produces_wav() -> None:
    adapter = MockTTSAdapter()
    path = await adapter.generate_audio("Hello world", "fake/ref.wav")
    assert path.exists()
    assert path.suffix == ".wav"
    path.unlink()


@pytest.mark.asyncio
async def test_mock_tts_wav_format() -> None:
    """Output WAV must be 24kHz mono 16-bit."""
    adapter = MockTTSAdapter()
    path = await adapter.generate_audio("Test sentence.", "fake/ref.wav")

    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1, "Expected mono"
        assert wf.getsampwidth() == 2, "Expected 16-bit"
        assert wf.getframerate() == MockTTSAdapter.SAMPLE_RATE

    path.unlink()


@pytest.mark.asyncio
async def test_mock_tts_file_is_non_empty() -> None:
    adapter = MockTTSAdapter()
    path = await adapter.generate_audio("Non-empty audio.", "fake/ref.wav")
    assert path.stat().st_size > 44, "WAV must be larger than header-only (44 bytes)"
    path.unlink()


@pytest.mark.asyncio
async def test_mock_tts_cleanup() -> None:
    """After unlink the file should not exist."""
    adapter = MockTTSAdapter()
    path = await adapter.generate_audio("Cleanup test.", "fake/ref.wav")
    path.unlink()
    assert not path.exists()


def test_get_tts_adapter_unknown_backend() -> None:
    """Factory should raise ValueError for unsupported backends."""
    with pytest.raises(ValueError, match="Unknown TTS backend"):
        get_tts_adapter("nonexistent_backend")


def test_get_tts_adapter_elevenlabs_missing_key() -> None:
    with pytest.raises(ValueError, match="ELEVENLABS_API_KEY"):
        get_tts_adapter("elevenlabs", elevenlabs_api_key=None)


def test_get_tts_adapter_cartesia_missing_key() -> None:
    with pytest.raises(ValueError, match="CARTESIA_API_KEY"):
        get_tts_adapter("cartesia", cartesia_api_key=None, cartesia_voice_id=None)
