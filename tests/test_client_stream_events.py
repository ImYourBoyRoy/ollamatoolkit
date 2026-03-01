# ./tests/test_client_stream_events.py
"""
Tests for structured stream event helpers on OllamaClient and AsyncOllamaClient.
"""

from __future__ import annotations

import json
from typing import AsyncIterator, Iterable, List
from unittest.mock import MagicMock, patch

import httpx
import pytest

from ollamatoolkit.client import AsyncOllamaClient, OllamaClient


def _stream_context(lines: Iterable[str]) -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.raise_for_status.return_value = None
    response.iter_lines.return_value = list(lines)

    context = MagicMock()
    context.__enter__.return_value = response
    context.__exit__.return_value = None
    return context


async def _aiter(lines: List[str]) -> AsyncIterator[str]:
    for line in lines:
        yield line


class _AsyncContext:
    def __init__(self, lines: List[str]):
        self._response = MagicMock(spec=httpx.Response)
        self._response.raise_for_status.return_value = None
        self._response.aiter_lines = lambda: _aiter(lines)

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class TestSyncStreamEvents:
    """Validate sync stream event wrappers."""

    def test_generate_events(self) -> None:
        context = _stream_context(
            [
                json.dumps({"model": "test", "response": "Hel", "done": False}),
                json.dumps({"model": "test", "response": "lo", "done": True}),
            ]
        )

        with patch.object(httpx.Client, "stream", return_value=context):
            with OllamaClient() as client:
                events = list(
                    client.stream_generate_events(
                        "test",
                        "say hello",
                    )
                )

        token_text = [event.text for event in events if event.event == "token"]
        assert "Hel" in token_text
        assert "lo" in token_text
        assert events[-1].event == "done"

    def test_chat_events_with_tool_call(self) -> None:
        context = _stream_context(
            [
                json.dumps(
                    {
                        "model": "test",
                        "message": {
                            "role": "assistant",
                            "content": "Calling tool",
                            "tool_calls": [
                                {
                                    "id": "call_123",
                                    "index": 0,
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": {"company": "B2X"},
                                    },
                                }
                            ],
                        },
                        "done": True,
                    }
                )
            ]
        )

        with patch.object(httpx.Client, "stream", return_value=context):
            with OllamaClient() as client:
                events = list(
                    client.stream_chat_events(
                        "test",
                        [{"role": "user", "content": "lookup B2X"}],
                    )
                )

        tool_events = [event for event in events if event.event == "tool_call"]
        assert len(tool_events) == 1
        assert tool_events[0].tool_calls[0].id == "call_123"


class TestAsyncStreamEvents:
    """Validate async stream event wrappers."""

    @pytest.mark.asyncio
    async def test_async_generate_events(self) -> None:
        context = _AsyncContext(
            [
                json.dumps({"model": "test", "response": "A", "done": False}),
                json.dumps({"model": "test", "response": "B", "done": True}),
            ]
        )

        with patch.object(httpx.AsyncClient, "stream", return_value=context):
            async with AsyncOllamaClient() as client:
                events = [
                    event
                    async for event in client.stream_generate_events(
                        "test",
                        "say AB",
                    )
                ]

        assert [e.text for e in events if e.event == "token"] == ["A", "B"]
        assert events[-1].event == "done"
