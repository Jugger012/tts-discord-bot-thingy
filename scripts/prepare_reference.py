"""
scripts/prepare_reference.py — Audio reference preparation utility.

Usage:
    python scripts/prepare_reference.py <input_audio> [--output assets/reference_voice.wav]

What it does:
  1. Load audio file (supports .mp3, .wav, .m4a, .ogg, .flac via pydub).
  2. Convert to mono.
  3. Trim leading/trailing silence.
  4. Reduce background noise (spectral subtraction via noisereduce).
  5. Resample to 24kHz (optimal for XTTS-v2) or 16kHz (for F5-TTS/STT).
  6. Normalise amplitude to -3 dBFS.
  7. Export as 16-bit mono WAV.

Importable function: ``process_audio(input_path, output_path) -> str``
This is used by the /setvoice Discord command for in-bot processing.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Target sample rate for XTTS-v2 (use 16000 for F5-TTS / STT pipelines)
_TARGET_SR = 24_000
_TARGET_CHANNELS = 1
_TARGET_DB = -3.0        # normalise to -3 dBFS
_SILENCE_THRESHOLD = -45  # dBFS threshold for silence trimming


def strip_silence(
    audio: AudioSegment,
    silence_thresh: int = _SILENCE_THRESHOLD,
    silence_len: int = 500,
    padding: int = 100,
) -> AudioSegment:
    """Trim leading and trailing silence from an AudioSegment."""
    from pydub.silence import detect_nonsilent

    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=silence_len,
        silence_thresh=silence_thresh,
    )
    if not nonsilent_ranges:
        return audio

    start_ms = max(0, nonsilent_ranges[0][0] - padding)
    end_ms = min(len(audio), nonsilent_ranges[-1][1] + padding)
    return audio[start_ms:end_ms]


def process_audio(input_path: str, output_path: str = "assets/reference_voice.wav") -> str:
    """
    Process a raw audio file into a clean reference sample.

    Parameters
    ----------
    input_path : str
        Path to the input audio file.
    output_path : str
        Destination path for the processed WAV file.

    Returns
    -------
    str
        Absolute path to the output WAV file.
    """
    from pydub import AudioSegment
    import noisereduce as nr
    import soundfile as sf

    input_path = str(input_path)
    output_path = str(output_path)

    log.info("Loading: %s", input_path)
    suffix = Path(input_path).suffix.lower()

    # ── Step 1: Load audio ───────────────────────────────────────────────────
    fmt_map = {".mp3": "mp3", ".wav": "wav", ".m4a": "mp4",
               ".ogg": "ogg", ".flac": "flac"}
    fmt = fmt_map.get(suffix, "mp3")
    audio: AudioSegment = AudioSegment.from_file(input_path, format=fmt)
    log.info("Loaded: %.1fs, %dHz, %dch", len(audio) / 1000, audio.frame_rate, audio.channels)

    # ── Step 2: Mono conversion ───────────────────────────────────────────────
    if audio.channels > 1:
        audio = audio.set_channels(1)
        log.info("Converted to mono.")

    # ── Step 3: Trim leading/trailing silence ─────────────────────────────────
    audio = strip_silence(
        audio,
        silence_thresh=_SILENCE_THRESHOLD,
        silence_len=500,   # min silence length to strip (ms)
        padding=100,        # leave 100ms of silence at edges
    )
    log.info("After silence trim: %.1fs", len(audio) / 1000)

    # ── Step 4: Resample to target rate ───────────────────────────────────────
    if audio.frame_rate != _TARGET_SR:
        audio = audio.set_frame_rate(_TARGET_SR)
        log.info("Resampled to %dHz.", _TARGET_SR)

    # ── Step 5: Noise reduction (spectral subtraction) ────────────────────────
    # Convert pydub → numpy float32 for noisereduce
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples /= (2 ** 15)  # normalise 16-bit int to float [-1, 1]

    # Use the first 0.5s as the noise profile if available
    noise_profile_samples = min(int(_TARGET_SR * 0.5), len(samples))
    noise_profile = samples[:noise_profile_samples]

    reduced = nr.reduce_noise(
        y=samples,
        y_noise=noise_profile,
        sr=_TARGET_SR,
        prop_decrease=0.75,
        stationary=False,
    )
    log.info("Noise reduction applied.")

    # ── Step 6: Amplitude normalisation ──────────────────────────────────────
    peak = np.max(np.abs(reduced))
    if peak > 0:
        target_linear = 10 ** (_TARGET_DB / 20.0)
        reduced = reduced * (target_linear / peak)
        reduced = np.clip(reduced, -1.0, 1.0)
    log.info("Normalised to %.1f dBFS.", _TARGET_DB)

    # ── Step 7: Export as 16-bit WAV ─────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_int16 = (reduced * 32767).astype(np.int16)
    sf.write(output_path, out_int16, _TARGET_SR, subtype="PCM_16")
    abs_out = str(Path(output_path).resolve())
    log.info("Saved to: %s", abs_out)
    return abs_out


# ── CLI entry point ───────────────────────────────────────────────────────────

def _cli() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Prepare a reference audio file for voice cloning."
    )
    parser.add_argument(
        "input",
        help="Input audio file (.mp3, .wav, .m4a, .ogg, .flac)",
    )
    parser.add_argument(
        "--output",
        default="assets/reference_voice.wav",
        help="Output path for the processed WAV (default: assets/reference_voice.wav)",
    )
    args = parser.parse_args()

    out = process_audio(args.input, args.output)
    print(f"\n[OK] Reference audio ready: {out}")
    print("You can now start the bot with:  python main.py")


if __name__ == "__main__":
    _cli()
