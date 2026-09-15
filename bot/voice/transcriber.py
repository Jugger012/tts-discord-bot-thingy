"""
bot/voice/transcriber.py — Speech-to-Text adapters.

Supported backends:
  - FasterWhisperTranscriber  — local GPU/CPU inference via faster-whisper
  - OpenAIWhisperTranscriber  — cloud via openai.audio.transcriptions

Both adapters accept raw 16kHz mono 16-bit PCM bytes and return a plain text
transcription string (or empty string on silence/error).
"""
from __future__ import annotations

import abc
import asyncio
import io
import logging
import struct
import tempfile
import wave
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_PCM_RATE = 16_000
_PCM_WIDTH = 2   # 16-bit
_PCM_CHANNELS = 1


def _pcm_to_wav_bytes(pcm: bytes) -> bytes:
    """Wrap raw PCM bytes in a WAV container (in-memory)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(_PCM_CHANNELS)
        wf.setsampwidth(_PCM_WIDTH)
        wf.setframerate(_PCM_RATE)
        wf.writeframes(pcm)
    return buf.getvalue()


# ── Abstract Base ─────────────────────────────────────────────────────────────


class BaseTranscriber(abc.ABC):
    @abc.abstractmethod
    async def transcribe(self, pcm_16k_mono: bytes) -> str:
        """
        Transcribe 16kHz mono 16-bit PCM audio bytes to text.

        Returns
        -------
        str
            Transcribed text, or empty string on silence / error.
        """
        ...


# ── Faster-Whisper ────────────────────────────────────────────────────────────


class FasterWhisperTranscriber(BaseTranscriber):
    """
    Local Whisper inference using CTranslate2 (faster-whisper).

    The model is loaded once at init and reused for all transcriptions.
    Inference is dispatched to a thread executor to avoid blocking the event
    loop.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
    ) -> None:
        from faster_whisper import WhisperModel

        if device == "auto":
            # Try CUDA; fall back to CPU silently
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        log.info(
            "Loading Faster-Whisper model '%s' on device '%s'…", model_size, device
        )
        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type="float16" if device == "cuda" else "int8",
        )
        log.info("Faster-Whisper ready.")

    async def transcribe(self, pcm_16k_mono: bytes) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, pcm_16k_mono)

    def _transcribe_sync(self, pcm_16k_mono: bytes) -> str:
        """Blocking transcription — run in a thread executor."""
        wav_bytes = _pcm_to_wav_bytes(pcm_16k_mono)
        buf = io.BytesIO(wav_bytes)
        segments, info = self._model.transcribe(
            buf,
            beam_size=5,
            language="en",
            vad_filter=True,   # built-in VAD to skip silence at boundaries
            vad_parameters={"min_silence_duration_ms": 200},
        )
        text = " ".join(seg.text.strip() for seg in segments)
        text = text.strip()
        if text:
            log.debug("Transcribed: %r", text)
        return text


# ── OpenAI Whisper API ────────────────────────────────────────────────────────


class OpenAIWhisperTranscriber(BaseTranscriber):
    """
    Cloud STT via the OpenAI Whisper API (``openai.audio.transcriptions``).

    Writes PCM to a temp WAV file, POSTs it, returns the text.
    """

    def __init__(self, api_key: str, model: str = "whisper-1") -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        log.info("OpenAI Whisper client initialised (model: %s).", model)

    async def transcribe(self, pcm_16k_mono: bytes) -> str:
        wav_bytes = _pcm_to_wav_bytes(pcm_16k_mono)
        # Write to a named temp file — OpenAI SDK requires a file-like with a name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = Path(tmp.name)

        try:
            with tmp_path.open("rb") as f:
                result = await self._client.audio.transcriptions.create(
                    model=self._model,
                    file=f,
                    response_format="text",
                )
            text = (result or "").strip()
            if text:
                log.debug("Transcribed (OpenAI): %r", text)
            return text
        except Exception as exc:
            log.warning("OpenAI Whisper transcription failed: %s", exc)
            return ""
        finally:
            tmp_path.unlink(missing_ok=True)


# ── Factory ───────────────────────────────────────────────────────────────────


def get_transcriber(
    backend: str,
    model_size: str = "base",
    device: str = "auto",
    openai_api_key: Optional[str] = None,
) -> BaseTranscriber:
    """
    Return the configured transcriber.

    Parameters
    ----------
    backend : str
        ``'faster-whisper'`` or ``'openai'``.
    model_size : str
        Whisper model size (for faster-whisper).
    device : str
        Inference device (for faster-whisper): ``'cpu'``, ``'cuda'``, or ``'auto'``.
    openai_api_key : str | None
        Required when backend is ``'openai'``.
    """
    if backend == "faster-whisper":
        return FasterWhisperTranscriber(model_size=model_size, device=device)
    elif backend == "openai":
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY required for STT_BACKEND=openai")
        return OpenAIWhisperTranscriber(api_key=openai_api_key)
    else:
        raise ValueError(f"Unknown STT backend: {backend!r}")
