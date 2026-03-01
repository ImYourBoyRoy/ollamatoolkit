# ./ollamatoolkit/agents/memory.py
"""
Ollama Toolkit - Agent Memory Management
========================================
Intelligent conversation memory with sliding-window summarization.

This module provides persistent, context-aware memory for agents that:
- Summarizes older conversations before truncating
- Persists memory across sessions
- Integrates with the agent's LLM for summarization

Usage:
    from ollamatoolkit.agents.memory import ConversationMemory, MemoryConfig

    memory = ConversationMemory(MemoryConfig(max_messages=50))
    memory.add({"role": "user", "content": "Hello"})
    memory.add({"role": "assistant", "content": "Hi there!"})

    context = memory.get_context()  # Returns messages with summary prepended

Key Features:
    - Automatic summarization when approaching message limit
    - Persistent storage to JSON file
    - Token-aware context building
    - Summary chaining for long conversations
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for conversation memory."""

    max_messages: int = 50
    """Maximum messages to keep in active memory."""

    summarize_threshold: int = 40
    """Trigger summarization when reaching this count."""

    summary_max_length: int = 500
    """Maximum characters for each summary segment."""

    persist_path: Optional[str] = None
    """Path to persist memory (JSON file). None = in-memory only."""

    auto_persist: bool = True
    """Automatically save after each add() if persist_path is set."""

    summary_prompt: str = (
        "Summarize the following conversation in 2-3 sentences, "
        "capturing the key topics, decisions, and any important context:\n\n"
        "{conversation}"
    )
    """Prompt template for summarization. Use {conversation} placeholder."""


@dataclass
class MemoryStats:
    """Statistics about memory usage."""

    total_messages: int
    active_messages: int
    summarized_messages: int
    summary_count: int
    estimated_tokens: int


class ConversationMemory:
    """
    Manages conversation history with intelligent summarization.

    When history exceeds threshold:
    1. Summarize older messages into a condensed form
    2. Keep recent messages verbatim
    3. Prepend summary as system context

    This prevents context overflow while preserving important information.
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize conversation memory.

        Args:
            config: Memory configuration. Uses defaults if None.
        """
        self.config = config or MemoryConfig()
        self.messages: List[Dict[str, Any]] = []
        self.summaries: List[str] = []
        self._summarized_count: int = 0
        self._created_at: str = datetime.now().isoformat()

        # Load existing memory if persist path exists
        if self.config.persist_path:
            self._load()

    def add(self, message: Dict[str, Any]) -> None:
        """
        Add a message to memory.

        Args:
            message: Dict with 'role' and 'content' keys (and optional others)
        """
        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.now().isoformat()

        self.messages.append(message)

        # Auto-persist if enabled
        if self.config.auto_persist and self.config.persist_path:
            self._save()

    def add_user(self, content: str, **metadata) -> None:
        """Convenience method to add a user message."""
        self.add({"role": "user", "content": content, **metadata})

    def add_assistant(self, content: str, **metadata) -> None:
        """Convenience method to add an assistant message."""
        self.add({"role": "assistant", "content": content, **metadata})

    def add_system(self, content: str, **metadata) -> None:
        """Convenience method to add a system message."""
        self.add({"role": "system", "content": content, **metadata})

    def add_tool(self, tool_name: str, result: str, **metadata) -> None:
        """Convenience method to add a tool result message."""
        self.add({"role": "tool", "name": tool_name, "content": result, **metadata})

    def add_message(self, role: str, content: str, **metadata) -> None:
        """Generic method to add a message with any role."""
        self.add({"role": role, "content": content, **metadata})

    def get_context(self, include_summary: bool = True) -> List[Dict[str, Any]]:
        """
        Get messages for LLM context.

        Args:
            include_summary: If True, prepend summary as system message

        Returns:
            List of messages suitable for LLM context
        """
        context = []

        # Add combined summary as system context
        if include_summary and self.summaries:
            combined_summary = " ".join(self.summaries)
            context.append(
                {
                    "role": "system",
                    "content": f"[Previous conversation summary: {combined_summary}]",
                }
            )

        # Return non-summarized messages
        context.extend(self.messages[self._summarized_count :])
        return context

    def get_context_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get context messages with optional limit (for backwards compatibility)."""
        context = self.get_context(include_summary=False)
        if limit:
            return context[-limit:]
        return context

    def get_all_messages(self) -> List[Dict[str, Any]]:
        """Get all messages including summarized ones (for debugging)."""
        return self.messages.copy()

    def needs_summarization(self) -> bool:
        """Check if summarization is needed based on threshold."""
        active_count = len(self.messages) - self._summarized_count
        return active_count >= self.config.summarize_threshold

    def summarize(
        self, summarizer: Callable[[str], str], force: bool = False
    ) -> Optional[str]:
        """
        Summarize older messages to free up context space.

        Args:
            summarizer: Function that takes conversation text and returns summary
            force: If True, summarize even if below threshold

        Returns:
            The new summary, or None if no summarization was performed
        """
        if not force and not self.needs_summarization():
            return None

        # Determine messages to summarize
        # Keep the last 10 messages unsummarized for continuity
        keep_recent = 10
        messages_to_summarize = self.messages[
            self._summarized_count : len(self.messages) - keep_recent
        ]

        if not messages_to_summarize:
            return None

        # Build conversation text
        conversation_parts = []
        for msg in messages_to_summarize:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            if content:
                conversation_parts.append(f"{role}: {content}")

        conversation_text = "\n".join(conversation_parts)

        # Generate summary using provided function
        try:
            prompt = self.config.summary_prompt.format(conversation=conversation_text)
            new_summary = summarizer(prompt)

            # Truncate if too long
            if len(new_summary) > self.config.summary_max_length:
                new_summary = new_summary[: self.config.summary_max_length] + "..."

            self.summaries.append(new_summary)
            self._summarized_count = len(self.messages) - keep_recent

            logger.info(
                f"Summarized {len(messages_to_summarize)} messages. "
                f"Active: {len(self.messages) - self._summarized_count}"
            )

            # Persist after summarization
            if self.config.auto_persist and self.config.persist_path:
                self._save()

            return new_summary

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return None

    def get_stats(self) -> MemoryStats:
        """Get statistics about memory usage."""
        active_count = len(self.messages) - self._summarized_count

        # Rough token estimate (4 chars per token)
        total_chars = sum(
            len(m.get("content", "")) for m in self.messages[self._summarized_count :]
        )
        total_chars += sum(len(s) for s in self.summaries)
        estimated_tokens = total_chars // 4

        return MemoryStats(
            total_messages=len(self.messages),
            active_messages=active_count,
            summarized_messages=self._summarized_count,
            summary_count=len(self.summaries),
            estimated_tokens=estimated_tokens,
        )

    def clear(self, keep_summaries: bool = False) -> None:
        """
        Clear all memory.

        Args:
            keep_summaries: If True, preserve existing summaries
        """
        self.messages = []
        self._summarized_count = 0

        if not keep_summaries:
            self.summaries = []

        if self.config.auto_persist and self.config.persist_path:
            self._save()

    def _save(self) -> None:
        """Persist memory to disk."""
        if not self.config.persist_path:
            return

        try:
            path = Path(self.config.persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "version": "1.0",
                "created_at": self._created_at,
                "updated_at": datetime.now().isoformat(),
                "messages": self.messages,
                "summaries": self.summaries,
                "summarized_count": self._summarized_count,
                "config": {
                    "max_messages": self.config.max_messages,
                    "summarize_threshold": self.config.summarize_threshold,
                },
            }

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Memory saved to {path}")

        except Exception as e:
            logger.error(f"Failed to save memory: {e}")

    def _load(self) -> None:
        """Load memory from disk."""
        if not self.config.persist_path:
            return

        path = Path(self.config.persist_path)
        if not path.exists():
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.messages = data.get("messages", [])
            self.summaries = data.get("summaries", [])
            self._summarized_count = data.get("summarized_count", 0)
            self._created_at = data.get("created_at", datetime.now().isoformat())

            logger.info(
                f"Loaded memory from {path}: "
                f"{len(self.messages)} messages, {len(self.summaries)} summaries"
            )

        except Exception as e:
            logger.error(f"Failed to load memory from {path}: {e}")

    def export(self) -> Dict[str, Any]:
        """Export memory state as a dictionary."""
        return {
            "messages": self.messages,
            "summaries": self.summaries,
            "summarized_count": self._summarized_count,
            "stats": {
                "total": len(self.messages),
                "active": len(self.messages) - self._summarized_count,
            },
        }

    def __len__(self) -> int:
        """Return number of active (non-summarized) messages."""
        return len(self.messages) - self._summarized_count

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"ConversationMemory("
            f"active={stats.active_messages}, "
            f"summarized={stats.summarized_messages}, "
            f"summaries={stats.summary_count})"
        )
