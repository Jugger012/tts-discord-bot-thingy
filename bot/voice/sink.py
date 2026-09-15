"""
bot/voice/sink.py — Custom py-cord 2.6+ voice sink with per-user VAD.

py-cord 2.6+ (the ``discord.voice`` AudioReader refactor) uses a *new* Sink
protocol that is incompatible with the old ``discord.sinks.Sink`` base class.
The new router calls:

  * ``sink.__sink_listeners__``  — list of ``(event_name, method_name)`` tuples
  * ``sink.walk_children()``     — iterable of child sinks (empty for a leaf)
  * ``sink.client``              — the live ``VoiceClient`` (set by AudioReader)
  * ``sink.is_opus()``           — whether the sink wants raw Opus bytes
  * ``sink.write(data, source)`` — called per decoded packet
      - ``data``   : ``VoiceData``  (.pcm: bytes, .source: Member|User|None)
      - ``source`` : same as ``data.source`` (passed separately for convenience)

PCM format delivered by py-cord: 16-bit signed, 48 kHz, stereo (2-ch).
WebRTC VAD requires:            16-bit signed, 16 kHz, mono.

Architecture
------------
                     ┌──────────────────────────────┐
Discord voice ──────▶│  VoiceActivitySink           │
 (per-user PCM)      │  ├── VAD per-user state      │
                     │  └── silence-timer asyncio   │──▶ speech_finished(user_id, pcm)
                     └──────────────────────────────┘
"""
from __future__ import annotations

import asyncio
import logging
import struct
import time
from typing import Awaitable, Callable, Dict, Iterator, List, Optional, Set, Tuple

log = logging.getLogger(__name__)

# ── PCM conversion constants ──────────────────────────────────────────────────

_IN_RATE = 48_000           # Discord sends 48 kHz stereo
_OUT_RATE = 16_000           # WebRTC VAD needs 16 kHz mono
_IN_CHANNELS = 2
_DOWNSAMPLE_FACTOR = _IN_RATE // _OUT_RATE  # 3

# VAD frame durations supported by webrtcvad: 10, 20, or 30 ms
_VAD_FRAME_MS = 20
_VAD_FRAME_SAMPLES_16K = _OUT_RATE * _VAD_FRAME_MS // 1000  # 320 samples
_VAD_FRAME_BYTES = _VAD_FRAME_SAMPLES_16K * 2               # 16-bit = 2 bytes/sample

SpeechCallback = Callable[[int, bytes], Awaitable[None]]


def _stereo48k_to_mono16k(raw_pcm: bytes) -> bytes:
    """
    Convert 48 kHz stereo 16-bit PCM to 16 kHz mono 16-bit PCM.

    Steps:
      1. Unpack all 16-bit signed samples.
      2. Average left + right channels → mono.
      3. Keep every _DOWNSAMPLE_FACTOR-th sample → 16 kHz.
    """
    n_frames = len(raw_pcm) // 4  # 4 bytes per stereo sample (2×2)
    samples: List[int] = []
    for i in range(n_frames):
        left  = struct.unpack_from("<h", raw_pcm, i * 4)[0]
        right = struct.unpack_from("<h", raw_pcm, i * 4 + 2)[0]
        samples.append((left + right) // 2)
    downsampled = samples[::_DOWNSAMPLE_FACTOR]
    return struct.pack(f"<{len(downsampled)}h", *downsampled)


# ── Per-user VAD state ────────────────────────────────────────────────────────

class _UserVADState:
    """Mutable state machine for a single user's voice activity detection."""

    def __init__(self, silence_threshold_ms: int, aggressiveness: int) -> None:
        import webrtcvad
        self.vad = webrtcvad.Vad(aggressiveness)
        self.silence_threshold_ms = silence_threshold_ms
        self.audio_buffer: bytearray = bytearray()
        self.pcm_16k_overflow: bytes = b""   # leftover bytes between receive calls
        self.is_speaking: bool = False
        self.last_speech_time: float = 0.0
        self.silence_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]


# ── New-style Sink ────────────────────────────────────────────────────────────

class VoiceActivitySink:
    """
    py-cord 2.6+ compatible Sink with per-user VAD.

    This class implements the *new* Sink protocol introduced in py-cord 2.6:
    ``__sink_listeners__``, ``walk_children()``, ``client``, ``is_opus()``,
    and ``write(data, source)``.

    Parameters
    ----------
    speech_callback:
        Async callable ``(user_id: int, pcm_16k_mono: bytes) -> None``
        invoked when a user finishes speaking.
    silence_threshold_ms:
        Milliseconds of silence before a speech segment is considered done.
    vad_aggressiveness:
        WebRTC VAD aggressiveness (0–3).  Higher → more aggressive filtering.
    bot_user_ids:
        User IDs belonging to the bot itself — their audio is never transcribed.
    """

    # The new SinkEventRouter iterates this to register event handlers.
    # We have no special sink events, so it is empty.
    __sink_listeners__: List[Tuple[str, str]] = []

    def __init__(
        self,
        speech_callback: SpeechCallback,
        silence_threshold_ms: int = 1200,
        vad_aggressiveness: int = 2,
        bot_user_ids: Optional[Set[int]] = None,
    ) -> None:
        self._callback = speech_callback
        self._silence_threshold_ms = silence_threshold_ms
        self._vad_aggressiveness = vad_aggressiveness
        self._bot_user_ids: Set[int] = bot_user_ids or set()
        self._user_state: Dict[int, _UserVADState] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Set by AudioReader after construction (new-style protocol)
        self.client = None  # VoiceClient | None

    # ── New-style Sink protocol ───────────────────────────────────────────────

    def walk_children(self) -> Iterator["VoiceActivitySink"]:
        """Return child sinks — this sink has none."""
        return iter(())

    def is_opus(self) -> bool:
        """Tell the decoder we want decoded PCM, not raw Opus bytes."""
        return False

    def write(self, data: object, source: object) -> None:
        """
        Called by PacketRouter for each decoded audio packet.

        ``data``   is a ``VoiceData`` object: ``.pcm`` (bytes) and
                   ``.source`` (Member | User | None).
        ``source`` is the same as ``data.source`` (convenience copy).
        """
        # Resolve user_id from the source Member/User object
        user_id: Optional[int] = getattr(source, "id", None)
        if user_id is None:
            # Fall back to data.source if source arg is None
            src = getattr(data, "source", None)
            user_id = getattr(src, "id", None)
        if user_id is None:
            return

        # Loopback guard
        if user_id in self._bot_user_ids:
            return

        # Grab the decoded PCM bytes from the VoiceData object
        raw_pcm: bytes = getattr(data, "pcm", b"")
        if not raw_pcm:
            return

        if self._loop is None:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                return

        # Initialise per-user state on first packet
        if user_id not in self._user_state:
            self._user_state[user_id] = _UserVADState(
                self._silence_threshold_ms, self._vad_aggressiveness
            )

        state = self._user_state[user_id]

        # Downsample 48 kHz stereo → 16 kHz mono
        pcm_16k = _stereo48k_to_mono16k(raw_pcm)
        pcm_16k = state.pcm_16k_overflow + pcm_16k
        state.pcm_16k_overflow = b""

        # Process VAD in fixed-size frames
        offset = 0
        while offset + _VAD_FRAME_BYTES <= len(pcm_16k):
            frame = pcm_16k[offset : offset + _VAD_FRAME_BYTES]
            offset += _VAD_FRAME_BYTES
            try:
                is_speech = state.vad.is_speech(frame, _OUT_RATE)
            except Exception:
                is_speech = False

            if is_speech:
                state.is_speaking = True
                state.last_speech_time = time.monotonic()
                state.audio_buffer.extend(frame)
                # Cancel any pending silence timer
                if state.silence_task and not state.silence_task.done():
                    state.silence_task.cancel()
                    state.silence_task = None
            else:
                if state.is_speaking:
                    state.audio_buffer.extend(frame)
                    if state.silence_task is None or state.silence_task.done():
                        state.silence_task = self._loop.create_task(
                            self._silence_countdown(user_id, state)
                        )

        # Save unprocessed remainder for next call
        state.pcm_16k_overflow = pcm_16k[offset:]

    def cleanup(self) -> None:
        """Cancel all pending silence timers and clear user state."""
        for state in self._user_state.values():
            if state.silence_task and not state.silence_task.done():
                state.silence_task.cancel()
        self._user_state.clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _silence_countdown(self, user_id: int, state: _UserVADState) -> None:
        """
        Wait for the silence threshold, then emit the buffered audio.
        If new speech arrives in the meantime the task is cancelled externally.
        """
        await asyncio.sleep(self._silence_threshold_ms / 1000.0)

        if not state.is_speaking:
            return

        pcm_bytes = bytes(state.audio_buffer)
        state.audio_buffer.clear()
        state.is_speaking = False
        state.silence_task = None

        if len(pcm_bytes) < _VAD_FRAME_BYTES * 3:
            # Too short — likely noise; discard
            return

        log.debug("Speech segment done for user %s (%d bytes)", user_id, len(pcm_bytes))
        try:
            await self._callback(user_id, pcm_bytes)
        except Exception:
            log.exception("speech_callback raised an exception for user %s", user_id)

