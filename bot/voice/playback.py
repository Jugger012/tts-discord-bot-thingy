"""
bot/voice/playback.py — Thread-safe audio playback queue for Discord voice.

Architecture
------------
AudioPlaybackQueue wraps an asyncio.Queue of audio file paths.  A background
worker coroutine dequeues entries and plays them back sequentially through the
Discord voice client using FFmpegPCMAudio.

Features:
  - Volume normalisation via discord.PCMVolumeTransformer.
  - Auto-reconnect if the voice client disconnects mid-playback.
  - Skippable playback (``skip()``).
  - Graceful drain on shutdown (``stop()``).
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

import discord

log = logging.getLogger(__name__)

_FFMPEG_OPTIONS = {
    "before_options": "-nostdin",
    "options": "-vn",  # no video
}


class AudioPlaybackQueue:
    """
    Async audio playback queue for a Discord voice channel.

    Parameters
    ----------
    voice_client : discord.VoiceClient
        The active voice client to play audio through.
    volume : float
        Playback volume multiplier (default: 0.9).
    """

    def __init__(self, voice_client: discord.VoiceClient, volume: float = 0.9) -> None:
        self._vc = voice_client
        self._volume = volume
        self._queue: asyncio.Queue[Path | None] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._current_file: Optional[Path] = None
        self._running = False

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Spin up the background playback worker."""
        if self._worker_task and not self._worker_task.done():
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        log.debug("Playback worker started.")

    async def stop(self) -> None:
        """
        Gracefully stop the worker.  Drains queued items without playing them
        and cancels any current playback.
        """
        self._running = False
        await self._queue.put(None)  # sentinel
        if self._vc.is_playing():
            self._vc.stop()
        if self._worker_task:
            try:
                await asyncio.wait_for(self._worker_task, timeout=3.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._worker_task.cancel()

    def enqueue(self, audio_path: Path) -> None:
        """
        Add an audio file to the playback queue.

        Parameters
        ----------
        audio_path : Path
            Path to a WAV or MP3 file to play.
        """
        self._queue.put_nowait(audio_path)
        log.debug("Enqueued: %s (queue size: %d)", audio_path.name, self._queue.qsize())

    def skip(self) -> None:
        """Interrupt the current audio and move to the next queued item."""
        if self._vc.is_playing():
            self._vc.stop()
            log.debug("Playback skipped.")

    def update_voice_client(self, vc: discord.VoiceClient) -> None:
        """Replace the voice client (e.g. after a reconnect)."""
        self._vc = vc

    # ── Worker ────────────────────────────────────────────────────────────────

    async def _worker(self) -> None:
        """Dequeue and play audio files sequentially."""
        while self._running:
            try:
                audio_path = await self._queue.get()
            except asyncio.CancelledError:
                break

            if audio_path is None:
                # Sentinel — shutdown signal
                break

            self._current_file = audio_path
            await self._play(audio_path)
            self._current_file = None

            # Clean up temp file after playback
            try:
                audio_path.unlink(missing_ok=True)
            except Exception:
                pass

            self._queue.task_done()

        log.debug("Playback worker stopped.")

    async def _play(self, audio_path: Path) -> None:
        """
        Play a single audio file, waiting for completion.  Handles reconnect
        if the voice client drops mid-playback.
        """
        if not self._vc.is_connected():
            log.warning("Voice client disconnected; attempting reconnect…")
            if not await self._reconnect():
                log.error("Reconnect failed; skipping audio: %s", audio_path.name)
                return

        try:
            source = discord.FFmpegPCMAudio(str(audio_path), **_FFMPEG_OPTIONS)
            source = discord.PCMVolumeTransformer(source, volume=self._volume)
        except Exception as exc:
            log.exception("Failed to create audio source for %s: %s", audio_path, exc)
            return

        done_event = asyncio.Event()

        def _after(error: Optional[Exception]) -> None:
            if error:
                log.warning("Playback error: %s", error)
            done_event.set()

        self._vc.play(source, after=_after)
        log.debug("Playing: %s", audio_path.name)
        await done_event.wait()

    async def _reconnect(self) -> bool:
        """
        Attempt to reconnect the voice client to the same channel.

        Returns True on success.
        """
        channel = self._vc.channel
        if channel is None:
            return False
        try:
            await self._vc.disconnect(force=True)
            new_vc = await channel.connect(timeout=10.0)
            self._vc = new_vc
            log.info("Reconnected to channel: %s", channel.name)
            return True
        except Exception as exc:
            log.exception("Failed to reconnect: %s", exc)
            return False
