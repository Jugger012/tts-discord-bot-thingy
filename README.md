# 🎙️ Aria — Real-Time Voice-to-Voice AI Discord Bot

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Discord Py-Cord](https://img.shields.io/badge/Discord-Py--Cord%202.6%2B-5865F2.svg?logo=discord&logoColor=white)](https://pycord.dev/)
[![Google GenAI](https://img.shields.io/badge/LLM-Gemini%20%26%20OpenAI-orange.svg?logo=google&logoColor=white)](https://ai.google.dev/)
[![STT faster-whisper](https://img.shields.io/badge/STT-faster--whisper-yellow.svg)](https://github.com/SYSTRAN/faster-whisper)
[![TTS XTTS & ElevenLabs](https://img.shields.io/badge/TTS-XTTS%20%7C%20ElevenLabs%20%7C%20Cartesia-brightgreen.svg)](https://github.com/coqui-ai/TTS)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A low-latency, modular voice assistant for Discord featuring zero-shot voice cloning, WebRTC voice activity detection (VAD), sentence-streaming LLM responses, and hybrid text-to-voice interaction.**

[Features](#-key-features) • [Architecture](#-architecture) • [Getting Started](#-getting-started) • [Configuration](#-configuration) • [Slash Commands](#-slash-commands) • [Audio Preparation](#-reference-audio-preparation) • [Troubleshooting](#-troubleshooting)

</div>

---

## 🌟 Overview

**Aria** is an autonomous AI companion that joins your Discord voice channels and talks with you in real time. Unlike traditional bots that wait for entire responses to finish before synthesising speech, Aria uses **sentence-level streaming**: as the LLM streams its response, complete sentences are immediately dispatched to the Text-to-Speech (TTS) engine and queued for playback. This dramatically cuts perceived latency and delivers a natural, conversation-like experience.

Equipped with **zero-shot voice cloning**, Aria can adopt any voice persona using just a short reference audio clip uploaded directly through Discord slash commands or stored locally.

---

## ✨ Key Features

- ⚡ **Streaming Voice-to-Voice Pipeline**: Overlaps LLM generation and TTS synthesis on a sentence-by-sentence boundary for near-instant audio responses.
- 🗣️ **Zero-Shot Voice Cloning**: Clone voices on the fly with reference audio using **Coqui XTTS-v2**, **F5-TTS**, **ElevenLabs**, or **Cartesia**.
- 🧠 **Multi-Backend LLM**: Native integration with **Google Gemini** (`gemini-3.8-flash` via modern `google-genai` SDK) and **OpenAI** (`gpt-4o`, `gpt-4o-mini`).
- 👂 **Accurate Speech-to-Text (STT)**: High-speed local transcription using **`faster-whisper`** (CTranslate2 with CUDA/CPU acceleration) or cloud-based **OpenAI Whisper**.
- 🎯 **WebRTC Voice Activity Detection (VAD)**: Real-time per-user voice segmenting with configurable silence thresholds and aggressiveness. No push-to-talk required.
- 🛡️ **Echo & Loopback Protection**: Intelligently ignores the bot's own voice streams to eliminate feedback loops.
- 💬 **Hybrid Text-to-Voice Mode**: Send messages in a designated text channel, and Aria speaks the generated response in the active voice channel.
- 🎛️ **Discord Slash Commands**: Complete control with `/join`, `/leave`, `/setvoice`, `/mode`, `/skip`, and `/clearctx`.
- 🧹 **Audio Processing Utility**: Includes `scripts/prepare_reference.py` for automated silence trimming, spectral noise reduction (`noisereduce`), resampling, and dBFS normalization.
- 🔐 **Discord DAVE E2EE Ready**: Compatible with Discord's End-to-End Voice Encryption via `davey` and Py-Cord 2.6+.

---

## 🏗️ Architecture

```
                                  VOICE PIPELINE
┌───────────────────────────┐
│   Discord Voice Channel   │
└─────────────┬─────────────┘
              │ 48kHz Stereo PCM (per user)
              ▼
┌───────────────────────────┐
│     VoiceActivitySink     │ ── Downsamples to 16kHz Mono
│   (WebRTC VAD Engine)     │ ── Tracks speech & silence threshold
└─────────────┬─────────────┘
              │ Finished Speech Audio (PCM)
              ▼
┌───────────────────────────┐
│   STT Transcriber Engine  │ ── faster-whisper (local CUDA/CPU)
│ (faster-whisper / OpenAI) │ ── or OpenAI Whisper (cloud)
└─────────────┬─────────────┘
              │ Transcribed User Text
              ▼
┌───────────────────────────┐
│    Conversation Memory    │ ── Rolling multi-turn context window
└─────────────┬─────────────┘
              │ Formatted History + Guardrails
              ▼
┌───────────────────────────┐
│     LLM Client Engine     │ ── Google Gemini / OpenAI
│  (Sentence-Stream Yield)  │ ── Splits stream by sentence delimiters (. ! ?)
└─────────────┬─────────────┘
              │ Sentence-by-sentence text chunks
              ▼
┌───────────────────────────┐
│        TTS Adapter        │ ── XTTS-v2 / F5-TTS / ElevenLabs / Cartesia
│ (Zero-Shot Voice Cloning) │ ── Synthesises WAV using reference voice
└─────────────┬─────────────┘
              │ Audio Files (WAV)
              ▼
┌───────────────────────────┐
│    AudioPlaybackQueue     │ ── Sequential Discord voice playback
└─────────────┬─────────────┘
              │ FFmpeg Audio Stream
              ▼
┌───────────────────────────┐
│   Discord Voice Channel   │ ◄── Users hear Aria speak!
└───────────────────────────┘
```

---

## 📂 Project Structure

```
├── assets/                       # Audio assets & reference voice samples
│   ├── reference_voice.wav       # Default reference audio for voice cloning
│   └── reference_voice_6s.wav
├── bot/
│   ├── brain/                    # Intelligence layer
│   │   ├── llm_client.py         # Gemini & OpenAI sentence-streaming clients
│   │   ├── memory.py             # Sliding-window conversation context memory
│   │   └── prompts.py            # Dynamic persona prompts & voice-output guardrails
│   ├── cogs/                     # Discord bot extensions
│   │   ├── text_listener.py      # Hybrid text-channel listener
│   │   ├── voice_commands.py     # Slash commands (/join, /leave, /setvoice, etc.)
│   │   └── voice_listener.py     # Voice state updates & speech processing loop
│   ├── voice/                    # Audio processing & synthesis layer
│   │   ├── playback.py           # Thread-safe audio playback queue
│   │   ├── sink.py               # Py-Cord VoiceActivitySink with WebRTC VAD
│   │   ├── transcriber.py        # faster-whisper & OpenAI STT adapters
│   │   └── tts_adapter.py        # XTTS-v2, F5-TTS, ElevenLabs, Cartesia TTS adapters
│   └── config.py                 # Pydantic BaseSettings environment loader
├── scripts/
│   └── prepare_reference.py      # Audio cleaner (noise reduction, silence trim, normalize)
├── tests/                        # Pytest suite for memory, audio queue, TTS, and VAD
├── .env.example                  # Environment configuration template
├── main.py                       # Application entry point
├── pytest.ini                    # Pytest configuration
└── requirements.txt              # Project dependencies
```

---

## 🚀 Getting Started

### 1. Prerequisites

- **Python 3.10+** (Python 3.11 recommended)
- **FFmpeg**: Must be installed and added to your system `PATH`.
  - **Windows**: `winget install Gyan.FFmpeg` or `choco install ffmpeg`
  - **Linux (Ubuntu/Debian)**: `sudo apt update && sudo apt install ffmpeg`
  - **macOS**: `brew install ffmpeg`
- **Discord Bot Token**: Create an application at [Discord Developer Portal](https://discord.com/developers/applications).
  - Enable **Message Content Intent** and **Server Members Intent** under the **Bot** tab.
  - Required OAuth2 Scopes: `bot`, `applications.commands`.
  - Required Bot Permissions: `Connect`, `Speak`, `Use Voice Activity`, `Send Messages`, `Read Message History`.
- **API Keys** (depending on chosen backends):
  - [Google AI Studio](https://aistudio.google.com/) for Gemini API key (Free tier available)
  - [OpenAI Platform](https://platform.openai.com/) for OpenAI API key
  - [ElevenLabs](https://elevenlabs.io/) for ElevenLabs API key
  - [Cartesia](https://cartesia.ai/) for Cartesia API key

---

### 2. Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/aria-voice-bot.git
   cd aria-voice-bot
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate

   # Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

> [!TIP]
> **GPU Acceleration (Optional for local STT & TTS)**:
> If you have an NVIDIA GPU, install PyTorch with CUDA support matching your system drivers before running the bot:
> ```bash
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

---

### 3. Configuration

Copy `.env.example` to `.env` and fill in your keys and preferences:

```bash
cp .env.example .env
```

Edit `.env`:

```ini
# Discord
DISCORD_TOKEN=your_bot_token_here
DISCORD_GUILD_ID=                      # Optional: lock slash commands to single guild for instant sync

# LLM Backend: gemini | openai
LLM_BACKEND=gemini
LLM_MODEL=gemini-3.8-flash
GEMINI_API_KEY=your_gemini_api_key_here

# TTS Backend: xtts | f5 | elevenlabs | cartesia
TTS_BACKEND=xtts

# STT Backend: faster-whisper | openai
STT_BACKEND=faster-whisper
WHISPER_MODEL_SIZE=base                # tiny, base, small, medium, large-v3
WHISPER_DEVICE=auto                    # cpu | cuda | auto

# Reference Voice Audio
REFERENCE_AUDIO_PATH=assets/reference_voice.wav

# Bot Persona
BOT_NAME=Aria
BOT_PERSONA=You are Aria, a witty, concise, and friendly AI assistant. Keep responses short and punchy.
```

---

### 4. Prepare Voice Reference (Optional)

Provide a 6–30 second clean sample of speech you want the bot to clone. Place it in `assets/reference_voice.wav` or run the audio preparation script:

```bash
python scripts/prepare_reference.py path/to/your_voice_sample.mp3 --output assets/reference_voice.wav
```

The script automatically:
1. Strips leading and trailing silence.
2. Removes background hiss with spectral noise reduction (`noisereduce`).
3. Downmixes to mono and resamples to 24 kHz (optimal for XTTS-v2).
4. Normalizes audio levels to -3 dBFS.

---

### 5. Run the Bot

```bash
python main.py
```

Once connected, you will see output similar to:
```text
2026-09-15 02:35:00 [INFO] discord-voice-bot: Loaded cog: bot.cogs.voice_commands
2026-09-15 02:35:01 [INFO] discord-voice-bot: Loaded cog: bot.cogs.voice_listener
2026-09-15 02:35:01 [INFO] discord-voice-bot: Loaded cog: bot.cogs.text_listener
2026-09-15 02:35:02 [INFO] discord-voice-bot: [OK] Logged in as Aria#1234
2026-09-15 02:35:02 [INFO] discord-voice-bot: TTS backend: xtts | STT backend: faster-whisper | LLM: gemini (gemini-3.8-flash)
```

---

## 🎮 Slash Commands

| Command | Arguments | Description |
|---|---|---|
| `/join` | *None* | Connects the bot to your current voice channel and starts listening. |
| `/leave` | *None* | Stops playback, releases the voice sink, and disconnects from the channel. |
| `/setvoice` | `attachment` (file) | Upload a `.wav`, `.mp3`, `.m4a`, `.flac`, or `.ogg` file to update the reference voice instantly. |
| `/mode` | `mode` (`voice` \| `text`) | Toggle between real-time voice-to-voice and text-to-voice hybrid mode in the current channel. |
| `/skip` | *None* | Immediately skips and halts the audio response currently playing. |
| `/clearctx` | *None* | Wipes the active conversation memory buffer to start fresh. |

---

## ⚙️ Configuration Options

Here is the complete reference of all environment variables supported in `.env`:

| Variable | Type / Options | Default | Description |
|---|---|---|---|
| `DISCORD_TOKEN` | `string` | *(Required)* | Bot token from Discord Developer Portal. |
| `DISCORD_GUILD_ID` | `integer` | `None` | Restricts slash commands to a single server for instant updates during development. |
| `LLM_BACKEND` | `gemini` \| `openai` | `gemini` | LLM service provider. |
| `LLM_MODEL` | `string` | `gemini-3.8-flash` | Model identifier (e.g. `gemini-3.8-flash`, `gpt-4o`, `gpt-4o-mini`). |
| `GEMINI_API_KEY` | `string` | `None` | Google AI Studio API key. |
| `OPENAI_API_KEY` | `string` | `None` | OpenAI platform API key. |
| `TTS_BACKEND` | `xtts` \| `f5` \| `elevenlabs` \| `cartesia` | `xtts` | Text-to-Speech synthesis backend. |
| `ELEVENLABS_API_KEY` | `string` | `None` | ElevenLabs API key (when using `elevenlabs`). |
| `ELEVENLABS_VOICE_ID`| `string` | `None` | Pre-existing voice ID (or leave blank to use reference audio cloning). |
| `CARTESIA_API_KEY` | `string` | `None` | Cartesia API key (when using `cartesia`). |
| `CARTESIA_VOICE_ID` | `string` | `None` | Cartesia voice ID to speak with. |
| `STT_BACKEND` | `faster-whisper` \| `openai` | `faster-whisper` | Speech-to-Text transcription backend. |
| `WHISPER_MODEL_SIZE` | `tiny` \| `base` \| `small` \| `medium` \| `large-v3` | `base` | Model size for `faster-whisper`. |
| `WHISPER_DEVICE` | `cpu` \| `cuda` \| `auto` | `auto` | Compute device for local Whisper inference. |
| `REFERENCE_AUDIO_PATH`| `filepath` | `assets/reference_voice.wav` | Path to reference WAV file used for zero-shot voice cloning. |
| `CONTEXT_WINDOW_SIZE` | `integer` (1–100) | `20` | Number of previous conversational turns kept in memory. |
| `VAD_SILENCE_THRESHOLD_MS` | `integer` (300–5000) | `1200` | Silence duration (ms) after speech before triggering generation. |
| `VAD_AGGRESSIVENESS` | `0` to `3` | `2` | WebRTC VAD filter intensity (3 is most aggressive at filtering noise). |
| `BOT_NAME` | `string` | `Aria` | The name the bot identifies as in conversations. |
| `BOT_PERSONA` | `string` | *(See .env.example)* | Persona guidelines and voice persona instructions for the LLM. |
| `HYBRID_TEXT_CHANNEL_ID` | `integer` | `None` | ID of the text channel monitored for text-to-voice mode. |
| `PLAYBACK_VOLUME` | `float` (0.0–2.0) | `0.9` | Output audio playback volume multiplier. |

---

## 🎯 Recommended Deployment Profiles

### 🖥️ Local / GPU Profile (Self-Hosted, Maximum Privacy)
Ideal for machines with an NVIDIA GPU (RTX 3060+ with 8GB+ VRAM):
- **STT**: `STT_BACKEND=faster-whisper`, `WHISPER_MODEL_SIZE=small` (or `medium`), `WHISPER_DEVICE=cuda`
- **TTS**: `TTS_BACKEND=xtts` (local Coqui XTTS-v2 zero-shot cloning)
- **LLM**: `LLM_BACKEND=gemini`, `LLM_MODEL=gemini-3.8-flash`

### ☁️ Cloud / Low-Resource Profile (Ultra Fast, Runs on any VPS / CPU)
Ideal for standard cloud instances without dedicated GPUs:
- **STT**: `STT_BACKEND=openai` (or `faster-whisper` with `WHISPER_MODEL_SIZE=tiny` on CPU)
- **TTS**: `TTS_BACKEND=elevenlabs` or `TTS_BACKEND=cartesia`
- **LLM**: `LLM_BACKEND=gemini` (fastest time-to-first-token)

---

## 🧪 Testing

The repository includes a comprehensive test suite using `pytest`:

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run only memory or playback tests
pytest tests/test_memory.py
pytest tests/test_playback.py
```

---

## 🔧 Troubleshooting

<details>
<summary><b>1. "ffmpeg is not recognized as an internal or external command"</b></summary>
Ensure FFmpeg is installed and accessible in your system's PATH. Open a terminal and run `ffmpeg -version`. If it fails, reinstall FFmpeg or add its <code>bin</code> folder to your system environment variables.
</details>

<details>
<summary><b>2. Voice bot connects but doesn't hear anything or respond</b></summary>
- Ensure the bot has <b>Connect</b>, <b>Speak</b>, and <b>Use Voice Activity</b> permissions in the voice channel.
- Check <code>VAD_AGGRESSIVENESS</code> in your <code>.env</code>. If set to <code>3</code>, it may filter out quiet microphones. Try setting it to <code>1</code> or <code>2</code>.
- Verify that your input microphone is not muted or below Discord's voice threshold.
</details>

<details>
<summary><b>3. Voice cloning sounds muffled or robotic</b></summary>
Voice cloning quality directly depends on the cleanliness of the reference audio:
- Use 6–15 seconds of clean, isolated speech with no background music, reverb, or secondary voices.
- Run <code>python scripts/prepare_reference.py your_audio.wav</code> to automatically denoise and normalize the audio.
</details>

<details>
<summary><b>4. Discord DAVE voice encryption errors</b></summary>
Ensure both <code>PyNaCl</code> and <code>davey</code> are installed in your environment:
```bash
pip install PyNaCl>=1.5.0 davey>=0.1.6
```
</details>

---

## 🤝 Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (`git commit -m 'Add amazing feature'`).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) — see the LICENSE file for details.
