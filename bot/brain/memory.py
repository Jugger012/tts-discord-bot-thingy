"""
bot/brain/memory.py — Rolling conversational context buffer.

Maintains a fixed-size deque of message turns, evicting the oldest when the
window is full.  Each entry stores the Discord user ID, display name, role
(user | assistant), and text content so the LLM always receives structured
history.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Literal


@dataclass
class MemoryEntry:
    role: Literal["user", "assistant"]
    username: str
    user_id: int | None          # None for assistant turns
    text: str
    timestamp: float = field(default_factory=time.time)

    def to_llm_dict(self) -> Dict[str, str]:
        """Convert to the dict format expected by most LLM SDKs."""
        if self.role == "user":
            content = f"[{self.username}]: {self.text}"
        else:
            content = self.text
        return {"role": self.role, "content": content}


class ConversationMemory:
    """
    Thread-safe rolling sliding-window conversation buffer.

    Parameters
    ----------
    max_turns : int
        Maximum number of entries to keep.  When exceeded, the oldest entry is
        silently evicted.
    """

    def __init__(self, max_turns: int = 20) -> None:
        self._max_turns = max_turns
        self._buffer: Deque[MemoryEntry] = deque(maxlen=max_turns)

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add_user_message(self, user_id: int, username: str, text: str) -> None:
        """Append a user utterance to the buffer."""
        self._buffer.append(
            MemoryEntry(role="user", username=username, user_id=user_id, text=text)
        )

    def add_assistant_message(self, text: str, bot_name: str = "Bot") -> None:
        """Append an assistant response to the buffer."""
        self._buffer.append(
            MemoryEntry(
                role="assistant",
                username=bot_name,
                user_id=None,
                text=text,
            )
        )

    def clear(self) -> None:
        """Wipe all history (e.g., when bot re-joins a channel)."""
        self._buffer.clear()

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_context_messages(self) -> List[Dict[str, str]]:
        """
        Return all buffered messages as a list of dicts with 'role' and
        'content' keys — compatible with OpenAI chat-completions and Gemini
        multi-turn formats.
        """
        return [entry.to_llm_dict() for entry in self._buffer]

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"<ConversationMemory turns={len(self._buffer)}/{self._max_turns}>"
