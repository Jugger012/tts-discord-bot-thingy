"""
tests/test_memory.py — Unit tests for ConversationMemory.

Tests:
  - Messages are stored and retrievable as LLM dicts.
  - Rolling eviction when buffer exceeds max_turns.
  - User vs assistant role assignment.
  - clear() wipes all entries.
  - __len__ tracks buffer size accurately.
"""
from __future__ import annotations

import pytest

from bot.brain.memory import ConversationMemory


class TestConversationMemory:
    def test_add_user_message(self) -> None:
        mem = ConversationMemory(max_turns=10)
        mem.add_user_message(user_id=123, username="Alice", text="Hello!")
        msgs = mem.get_context_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert "Alice" in msgs[0]["content"]
        assert "Hello!" in msgs[0]["content"]

    def test_add_assistant_message(self) -> None:
        mem = ConversationMemory(max_turns=10)
        mem.add_assistant_message(text="Hi there!", bot_name="Aria")
        msgs = mem.get_context_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["content"] == "Hi there!"

    def test_rolling_eviction(self) -> None:
        """Oldest message must be evicted when max_turns is exceeded."""
        mem = ConversationMemory(max_turns=3)
        mem.add_user_message(1, "A", "first")
        mem.add_user_message(2, "B", "second")
        mem.add_user_message(3, "C", "third")
        # Buffer is now full — adding a 4th should evict the first
        mem.add_user_message(4, "D", "fourth")

        assert len(mem) == 3
        contents = [m["content"] for m in mem.get_context_messages()]
        # "first" should have been evicted
        assert not any("first" in c for c in contents)
        assert any("fourth" in c for c in contents)

    def test_clear(self) -> None:
        mem = ConversationMemory(max_turns=10)
        mem.add_user_message(1, "X", "something")
        mem.add_assistant_message("reply")
        mem.clear()
        assert len(mem) == 0
        assert mem.get_context_messages() == []

    def test_len_tracking(self) -> None:
        mem = ConversationMemory(max_turns=5)
        assert len(mem) == 0
        for i in range(5):
            mem.add_user_message(i, f"User{i}", f"msg{i}")
        assert len(mem) == 5
        # Adding one more should keep length at 5 (eviction)
        mem.add_user_message(99, "Extra", "overflow")
        assert len(mem) == 5

    def test_mixed_conversation(self) -> None:
        """Interleaved user/assistant messages maintain correct order."""
        mem = ConversationMemory(max_turns=10)
        mem.add_user_message(1, "Alice", "What time is it?")
        mem.add_assistant_message("I don't have a clock, but it's probably late.")
        mem.add_user_message(1, "Alice", "Fair enough.")

        msgs = mem.get_context_messages()
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert msgs[2]["role"] == "user"

    def test_repr(self) -> None:
        mem = ConversationMemory(max_turns=20)
        mem.add_user_message(1, "Test", "hi")
        r = repr(mem)
        assert "ConversationMemory" in r
        assert "1/20" in r
