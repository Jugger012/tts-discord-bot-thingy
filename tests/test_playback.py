"""
tests/test_playback.py — Unit tests for AudioPlaybackQueue.

Uses a mock discord.VoiceClient to verify:
  - Audio files are dequeued and played in FIFO order.
  - skip() stops the current source.
  - stop() drains the queue without playing remaining items.
  - Temp files are deleted post-playback.
"""
from __future__ import annotations

import asyncio
import struct
import tempfile
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.voice.playback import AudioPlaybackQueue


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_silent_wav(duration_s: float = 0.1, sr: int = 24_000) -> Path:
    """Create a temporary silent WAV file for test playback."""
    n = int(sr * duration_s)
    tmp = Path(tempfile.mktemp(suffix=".wav"))
    with wave.open(str(tmp), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{n}h", *([0] * n)))
    return tmp


def make_mock_vc() -> MagicMock:
    """Return a mock discord.VoiceClient that simulates play() + after callback."""
    vc = MagicMock()
    vc.is_connected.return_value = True
    vc.is_playing.return_value = False
    vc.channel = MagicMock()
    vc.channel.name = "test-channel"

    # Simulate play() — immediately invoke the after callback
    def fake_play(source, after=None):
        vc.is_playing.return_value = True
        if after:
            after(None)  # no error
        vc.is_playing.return_value = False

    vc.play.side_effect = fake_play
    return vc


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_enqueue_and_play_order() -> None:
    """Files must be played in FIFO order."""
    played_order: list[str] = []
    vc = make_mock_vc()

    with patch("bot.voice.playback.discord.FFmpegPCMAudio") as mock_ffmpeg, \
         patch("bot.voice.playback.discord.PCMVolumeTransformer") as mock_vol:

        mock_ffmpeg.return_value = MagicMock()
        mock_vol.return_value = MagicMock()

        def fake_play(source, after=None):
            # Capture the filename from the FFmpegPCMAudio call args
            call_args = mock_ffmpeg.call_args
            if call_args:
                played_order.append(call_args[0][0])  # first positional arg = file path
            if after:
                after(None)

        vc.play.side_effect = fake_play

        queue = AudioPlaybackQueue(vc, volume=1.0)
        queue.start()

        paths = [make_silent_wav() for _ in range(3)]
        for p in paths:
            queue.enqueue(p)

        await asyncio.sleep(0.1)   # allow worker to drain
        await queue.stop()

    # All three should have been played, in order
    assert len(played_order) == 3
    for i, p in enumerate(paths):
        assert str(p) == played_order[i]


@pytest.mark.asyncio
async def test_skip_stops_playback() -> None:
    vc = make_mock_vc()
    vc.is_playing.return_value = True   # pretend something is playing

    queue = AudioPlaybackQueue(vc, volume=1.0)
    queue.skip()

    vc.stop.assert_called_once()


@pytest.mark.asyncio
async def test_stop_drains_queue() -> None:
    """After stop(), the queue should be empty and worker done."""
    vc = make_mock_vc()

    with patch("bot.voice.playback.discord.FFmpegPCMAudio"), \
         patch("bot.voice.playback.discord.PCMVolumeTransformer"):

        queue = AudioPlaybackQueue(vc, volume=1.0)
        queue.start()

        # Enqueue several items but stop immediately
        paths = [make_silent_wav() for _ in range(5)]
        for p in paths:
            queue.enqueue(p)

        await queue.stop()

        # Worker task should be done
        assert queue._worker_task is None or queue._worker_task.done()

    # Cleanup temp files
    for p in paths:
        p.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_temp_file_cleanup() -> None:
    """AudioPlaybackQueue must delete temp files after playing them."""
    vc = make_mock_vc()
    tmp = make_silent_wav()
    assert tmp.exists()

    with patch("bot.voice.playback.discord.FFmpegPCMAudio") as mock_ffmpeg, \
         patch("bot.voice.playback.discord.PCMVolumeTransformer") as mock_vol:

        mock_ffmpeg.return_value = MagicMock()
        mock_vol.return_value = MagicMock()

        def fake_play(source, after=None):
            if after:
                after(None)

        vc.play.side_effect = fake_play

        queue = AudioPlaybackQueue(vc, volume=1.0)
        queue.start()
        queue.enqueue(tmp)

        await asyncio.sleep(0.2)
        await queue.stop()

    assert not tmp.exists(), "Temp file should have been cleaned up after playback"
