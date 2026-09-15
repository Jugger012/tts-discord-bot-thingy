"""
bot/voice/tts_adapter.py — Pluggable Text-to-Speech / Voice Cloning adapters.

Interface
---------
All adapters implement ``BaseTTSAdapter``:

    async def generate_audio(text: str, reference_path: str) -> Path

Returns the path to a synthesised WAV file ready for FFmpeg playback.

Supported Backends
------------------
  - XTTSAdapter       — Coqui XTTS-v2 (local, zero-shot cloning)
  - F5TTSAdapter      — F5-TTS (local, zero-shot cloning)
  - ElevenLabsAdapter — ElevenLabs Instant Voice Cloning (cloud)
  - CartesiaAdapter   — Cartesia voice API (cloud)
"""
from __future__ import annotations

import abc
import asyncio
import hashlib
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Temporary directory for synthesised audio files
_TMP_DIR = Path(tempfile.gettempdir()) / "discord_bot_tts"
_TMP_DIR.mkdir(parents=True, exist_ok=True)


def _unique_path(prefix: str = "tts", suffix: str = ".wav") -> Path:
    """Generate a unique temp file path."""
    ts = str(time.time()).encode()
    h = hashlib.md5(ts).hexdigest()[:8]
    return _TMP_DIR / f"{prefix}_{h}{suffix}"


# ── Abstract Base ─────────────────────────────────────────────────────────────


class BaseTTSAdapter(abc.ABC):
    """
    Abstract TTS adapter.

    Subclasses must implement ``generate_audio`` which takes a text string
    and a reference audio path and returns a Path to the synthesised WAV file.
    """

    @abc.abstractmethod
    async def generate_audio(self, text: str, reference_path: str) -> Path:
        """
        Synthesise ``text`` in the voice described by ``reference_path``.

        Parameters
        ----------
        text : str
            The text to synthesise (should be plain — no markdown).
        reference_path : str
            Path to the reference audio WAV file used for voice cloning.

        Returns
        -------
        Path
            Path to the output WAV file.  Caller is responsible for cleanup.
        """
        ...


# ── XTTS-v2 Adapter ───────────────────────────────────────────────────────────


class XTTSAdapter(BaseTTSAdapter):
    """
    Coqui XTTS-v2 zero-shot voice cloning.

    Loads the model once at init.  Inference runs in a thread executor to
    avoid blocking the asyncio event loop.

    Requires:
        pip install TTS
    """

    def __init__(self, device: str = "cpu") -> None:
        os.environ["COQUI_TOS_AGREED"] = "1"
        from collections import defaultdict
        import torch
        from TTS.config.shared_configs import BaseDatasetConfig
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig
        from TTS.utils.radam import RAdam

        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([
                dict, defaultdict, RAdam, BaseDatasetConfig, XttsConfig, XttsAudioConfig, XttsArgs
            ])

        from TTS.api import TTS  # type: ignore[import-untyped]

        log.info("Loading XTTS-v2 model (device: %s)...", device)
        self._tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=True,
        )
        if device != "cpu":
            self._tts.to(device)  # type: ignore[attr-defined]
        self._device = device
        log.info("XTTS-v2 ready.")

    async def generate_audio(self, text: str, reference_path: str) -> Path:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._generate_sync, text, reference_path
        )

    def _generate_sync(self, text: str, reference_path: str) -> Path:
        out_path = _unique_path("xtts")
        self._tts.tts_to_file(
            text=text,
            speaker_wav=reference_path,
            language="en",
            file_path=str(out_path),
            split_sentences=False,
            speed=1.1,
        )
        log.debug("XTTS generated: %s", out_path)
        return out_path



# ── F5-TTS Adapter ────────────────────────────────────────────────────────────


class F5TTSAdapter(BaseTTSAdapter):
    """
    F5-TTS zero-shot voice cloning.

    Calls the ``f5-tts_infer-cli`` command-line tool.

    Requires:
        pip install f5-tts
    """

    async def generate_audio(self, text: str, reference_path: str) -> Path:
        out_path = _unique_path("f5tts")
        cmd = [
            "f5-tts_infer-cli",
            "--model", "F5TTS_v1_Base",
            "--ref_audio", reference_path,
            "--ref_text", "",        # auto-transcribe reference
            "--gen_text", text,
            "--output_file", str(out_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            log.error("F5-TTS error:\n%s", stderr.decode())
            raise RuntimeError("F5-TTS inference failed")
        log.debug("F5-TTS generated: %s", out_path)
        return out_path


# ── ElevenLabs Adapter ────────────────────────────────────────────────────────


class ElevenLabsAdapter(BaseTTSAdapter):
    """
    ElevenLabs Instant Voice Cloning via the official SDK.

    If ``voice_id`` is set, uses the pre-created voice.  Otherwise uploads
    the reference audio via the Add Voice API on first call and caches the ID.

    Requires:
        pip install elevenlabs
    """

    DEFAULT_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # Sarah (premade voice fallback)

    def __init__(self, api_key: str, voice_id: Optional[str] = None) -> None:
        from elevenlabs.client import ElevenLabs  # type: ignore[import-untyped]

        self._client = ElevenLabs(api_key=api_key)
        self._voice_id = voice_id
        self._uploaded = False
        log.info("ElevenLabsAdapter initialised (voice_id: %s).", voice_id or "auto")

    async def generate_audio(self, text: str, reference_path: str) -> Path:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._generate_sync, text, reference_path
        )

    def _generate_sync(self, text: str, reference_path: str) -> Path:
        # Lazily upload reference audio if no voice_id supplied
        if not self._voice_id:
            log.info("Uploading reference audio to ElevenLabs…")
            try:
                with open(reference_path, "rb") as f:
                    voice = self._client.voices.ivc.create(
                        name="DiscordBotVoice",
                        files=[f],
                        description="Auto-cloned voice for Discord bot",
                    )
                self._voice_id = voice.voice_id
                log.info("ElevenLabs voice cloned: %s", self._voice_id)
            except Exception as exc:
                log.warning(
                    "ElevenLabs voice cloning failed (%s). Falling back to premade voice (%s).",
                    exc,
                    self.DEFAULT_VOICE_ID,
                )
                self._voice_id = self.DEFAULT_VOICE_ID

        audio_stream = self._client.text_to_speech.convert(
            voice_id=self._voice_id,
            text=text,
            model_id="eleven_turbo_v2_5",
            output_format="mp3_44100_128",
        )
        audio_bytes = b"".join(audio_stream)

        # Save to temp WAV (convert MP3 → WAV via pydub for FFmpeg compatibility)
        out_path = _unique_path("elevenlabs", ".wav")
        self._mp3_to_wav(audio_bytes, out_path)
        return out_path

    @property
    def voice_id(self) -> Optional[str]:
        return self._voice_id

    def set_voice(self, voice_id: str) -> None:
        """Dynamically update the active voice ID without restarting."""
        self._voice_id = voice_id
        log.info("ElevenLabs voice dynamically updated to: %s", voice_id)

    def reload_reference(self, reference_path: str) -> tuple[bool, str]:
        """
        Attempt to clone a new voice from reference audio.
        Returns (success, voice_id_or_error_message).
        """
        log.info("Attempting to clone updated reference audio in ElevenLabs: %s", reference_path)
        try:
            with open(reference_path, "rb") as f:
                voice = self._client.voices.ivc.create(
                    name="DiscordBotVoice",
                    files=[f],
                    description="Auto-cloned voice for Discord bot",
                )
            self._voice_id = voice.voice_id
            log.info("ElevenLabs voice cloned successfully: %s", self._voice_id)
            return True, self._voice_id
        except Exception as exc:
            log.warning("ElevenLabs voice cloning failed: %s", exc)
            return False, str(exc)

    @staticmethod
    def _mp3_to_wav(mp3_bytes: bytes, out_path: Path) -> None:
        from pydub import AudioSegment  # type: ignore[import-untyped]
        import io
        seg = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
        seg.export(str(out_path), format="wav")


# ── Cartesia Adapter ──────────────────────────────────────────────────────────


class CartesiaAdapter(BaseTTSAdapter):
    """
    Cartesia TTS via the REST API.

    Requires a pre-existing ``voice_id`` from the Cartesia dashboard.

    Requires:
        pip install httpx
    """

    _BASE_URL = "https://api.cartesia.ai"

    def __init__(self, api_key: str, voice_id: str) -> None:
        import httpx

        self._api_key = api_key
        self._voice_id = voice_id
        self._session: Optional[httpx.Client] = None
        log.info("CartesiaAdapter initialised (voice_id: %s).", voice_id)

    @property
    def voice_id(self) -> str:
        return self._voice_id

    def set_voice(self, voice_id: str) -> None:
        """Dynamically update the active voice ID without restarting."""
        self._voice_id = voice_id
        log.info("Cartesia voice dynamically updated to: %s", voice_id)

    def _get_session(self):  # type: ignore[return]
        import httpx
        if self._session is None:
            self._session = httpx.Client(
                base_url=self._BASE_URL,
                headers={
                    "X-API-Key": self._api_key,
                    "Cartesia-Version": "2024-06-10",
                },
                timeout=30.0,
            )
        return self._session

    async def generate_audio(self, text: str, reference_path: str) -> Path:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._generate_sync, text
        )

    def _generate_sync(self, text: str) -> Path:
        session = self._get_session()
        resp = session.post(
            "/tts/bytes",
            json={
                "model_id": "sonic-2",
                "transcript": text,
                "voice": {"mode": "id", "id": self._voice_id},
                "output_format": {
                    "container": "wav",
                    "encoding": "pcm_f32le",
                    "sample_rate": 24000,
                },
            },
        )
        resp.raise_for_status()
        out_path = _unique_path("cartesia", ".wav")
        out_path.write_bytes(resp.content)
        log.debug("Cartesia generated: %s", out_path)
        return out_path


# ── Factory ───────────────────────────────────────────────────────────────────


def get_tts_adapter(
    backend: str,
    elevenlabs_api_key: Optional[str] = None,
    elevenlabs_voice_id: Optional[str] = None,
    cartesia_api_key: Optional[str] = None,
    cartesia_voice_id: Optional[str] = None,
    device: str = "cpu",
) -> BaseTTSAdapter:
    """Return the configured TTS adapter."""
    if backend == "xtts":
        return XTTSAdapter(device=device)
    elif backend == "f5":
        return F5TTSAdapter()
    elif backend == "elevenlabs":
        if not elevenlabs_api_key:
            raise ValueError("ELEVENLABS_API_KEY required for TTS_BACKEND=elevenlabs")
        return ElevenLabsAdapter(api_key=elevenlabs_api_key, voice_id=elevenlabs_voice_id)
    elif backend == "cartesia":
        if not cartesia_api_key or not cartesia_voice_id:
            raise ValueError(
                "CARTESIA_API_KEY and CARTESIA_VOICE_ID required for TTS_BACKEND=cartesia"
            )
        return CartesiaAdapter(api_key=cartesia_api_key, voice_id=cartesia_voice_id)
    else:
        raise ValueError(f"Unknown TTS backend: {backend!r}")
