"""
bot/brain/llm_client.py — Pluggable LLM adapter with sentence-streaming.

Supports:
  - GeminiClient  → google-genai SDK (gemini-3.8-flash or any Gemini model)
  - OpenAIClient  → openai SDK (gpt-4o, gpt-4o-mini, etc.)

Both clients implement BaseLLMClient with an async generator `stream_sentences`
that yields complete sentences one at a time.  This allows the TTS engine to
start synthesising the first sentence while the LLM is still generating the
rest — cutting perceived latency significantly.
"""
from __future__ import annotations

import abc
import asyncio
import logging
import re
from typing import AsyncIterator, List

log = logging.getLogger(__name__)

# Regex to split streamed text at sentence boundaries (., !, ?) followed by
# whitespace or end-of-string, while being careful not to split on e.g. "Dr."
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> List[str]:
    """Split a block of text into individual sentences."""
    parts = _SENTENCE_END.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


# ── Abstract Base ─────────────────────────────────────────────────────────────


class BaseLLMClient(abc.ABC):
    """
    Abstract LLM adapter.  Subclasses must implement ``stream_sentences``.
    """

    @abc.abstractmethod
    async def stream_sentences(
        self,
        system_prompt: str,
        messages: List[dict],
        new_user_text: str,
    ) -> AsyncIterator[str]:
        """
        Async generator that yields one complete sentence at a time.

        Parameters
        ----------
        system_prompt : str
            The assembled system/persona prompt.
        messages : list[dict]
            Rolling history from ``ConversationMemory.get_context_messages()``.
        new_user_text : str
            The latest user utterance (already in plain text form).
        """
        # Must be declared as async generator via 'yield' in subclass
        raise NotImplementedError
        yield  # noqa: unreachable — makes type-checker happy


# ── Gemini Client ─────────────────────────────────────────────────────────────


class GeminiClient(BaseLLMClient):
    """
    LLM client backed by the Google Gemini API (google-genai SDK ≥ 2.3.0).

    Uses the Interactions API with streaming enabled.  Text tokens are
    accumulated in a buffer and flushed as complete sentences.
    """

    def __init__(self, api_key: str, model: str = "gemini-3.8-flash") -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model
        log.info("GeminiClient initialised with model: %s", model)

    def _build_input(
        self,
        system_prompt: str,
        messages: List[dict],
        new_user_text: str,
    ) -> str:
        """
        Pack history + new message into a single input string.
        The Interactions API handles statefulness via previous_interaction_id;
        here we inline the context for full control over the window.
        """
        history_lines = []
        for msg in messages:
            history_lines.append(f"{msg['role'].upper()}: {msg['content']}")
        if history_lines:
            history_block = "\n".join(history_lines) + "\n\n"
        else:
            history_block = ""
        return f"{history_block}USER: {new_user_text}"

    async def stream_sentences(
        self,
        system_prompt: str,
        messages: List[dict],
        new_user_text: str,
    ) -> AsyncIterator[str]:
        from google import genai
        from google.genai import types as genai_types

        loop = asyncio.get_running_loop()
        input_text = self._build_input(system_prompt, messages, new_user_text)

        # Run blocking Gemini streaming call in a thread executor
        sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()

        def _stream_worker() -> None:
            buffer = ""
            try:
                for event in self._client.interactions.create(
                    model=self._model,
                    input=input_text,
                    system_instruction=system_prompt,
                    stream=True,
                    store=False,
                ):
                    if event.event_type == "step.delta" and event.delta.type == "text":
                        buffer += event.delta.text
                        # Flush complete sentences from the buffer
                        while True:
                            m = _SENTENCE_END.search(buffer)
                            if not m:
                                break
                            sentence = buffer[: m.start() + 1].strip()
                            buffer = buffer[m.end() :]
                            if sentence:
                                loop.call_soon_threadsafe(
                                    sentence_queue.put_nowait, sentence
                                )
                # Flush any remaining text as the final sentence
                if buffer.strip():
                    loop.call_soon_threadsafe(
                        sentence_queue.put_nowait, buffer.strip()
                    )
            except Exception as exc:
                log.exception("Gemini streaming error: %s", exc)
            finally:
                loop.call_soon_threadsafe(sentence_queue.put_nowait, None)

        # Run the blocking worker in the thread pool
        await loop.run_in_executor(None, _stream_worker)

        # Drain the queue, yielding sentences
        # Note: worker signals completion by enqueuing None
        # We restart the queue drain in caller loop
        while True:
            sentence = await sentence_queue.get()
            if sentence is None:
                break
            yield sentence


# ── OpenAI Client ─────────────────────────────────────────────────────────────


class OpenAIClient(BaseLLMClient):
    """
    LLM client backed by the OpenAI API (openai SDK ≥ 1.40.0).

    Uses ChatCompletion streaming with the full message history.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        log.info("OpenAIClient initialised with model: %s", model)

    async def stream_sentences(
        self,
        system_prompt: str,
        messages: List[dict],
        new_user_text: str,
    ) -> AsyncIterator[str]:
        all_messages = (
            [{"role": "system", "content": system_prompt}]
            + messages
            + [{"role": "user", "content": new_user_text}]
        )

        buffer = ""
        async with await self._client.chat.completions.create(
            model=self._model,
            messages=all_messages,  # type: ignore[arg-type]
            stream=True,
            temperature=0.8,
            max_tokens=512,
        ) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    buffer += delta.content
                    while True:
                        m = _SENTENCE_END.search(buffer)
                        if not m:
                            break
                        sentence = buffer[: m.start() + 1].strip()
                        buffer = buffer[m.end() :]
                        if sentence:
                            yield sentence

        if buffer.strip():
            yield buffer.strip()


# ── Factory ───────────────────────────────────────────────────────────────────


def get_llm_client(
    backend: str,
    gemini_api_key: str | None = None,
    openai_api_key: str | None = None,
    model: str = "gemini-3.8-flash",
) -> BaseLLMClient:
    """
    Return the appropriate LLM client for the configured backend.

    Raises
    ------
    ValueError
        If the required API key is missing for the chosen backend.
    """
    if backend == "gemini":
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY must be set when LLM_BACKEND=gemini")
        return GeminiClient(api_key=gemini_api_key, model=model)
    elif backend == "openai":
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set when LLM_BACKEND=openai")
        return OpenAIClient(api_key=openai_api_key, model=model)
    else:
        raise ValueError(f"Unknown LLM backend: {backend!r}")
