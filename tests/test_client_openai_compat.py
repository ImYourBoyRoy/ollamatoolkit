# ./tests/test_client_openai_compat.py
"""
Tests for OpenAI-compatible client methods (`/v1/*`).
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Iterable, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ollamatoolkit.client import AsyncOllamaClient, OllamaClient
from ollamatoolkit.openai_types import (
    OpenAICompatChatCompletionsResponse,
    OpenAICompatEmbeddingsResponse,
    OpenAICompatModelsResponse,
)


def make_mock_response(status_code: int = 200, json_data: dict[str, Any] | None = None):
    """Create a properly mocked httpx.Response."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.text = json.dumps(json_data or {})
    response.json.return_value = json_data or {}

    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="Error",
            request=MagicMock(),
            response=response,
        )
    else:
        response.raise_for_status.return_value = None

    return response


def make_sync_stream_context(lines: Iterable[str]) -> MagicMock:
    """Create context manager mock for sync `httpx.Client.stream`."""
    response = MagicMock(spec=httpx.Response)
    response.raise_for_status.return_value = None
    response.iter_lines.return_value = list(lines)

    context = MagicMock()
    context.__enter__.return_value = response
    context.__exit__.return_value = None
    return context


async def _aiter_lines(lines: List[str]) -> AsyncIterator[str]:
    for line in lines:
        yield line


class _AsyncStreamContext:
    """Async context manager wrapper for mocked stream responses."""

    def __init__(self, lines: List[str]):
        self._response = MagicMock(spec=httpx.Response)
        self._response.raise_for_status.return_value = None
        self._response.aiter_lines = lambda: _aiter_lines(lines)

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class TestSyncOpenAICompat:
    """Validate sync `/v1/*` helpers."""

    def test_openai_list_models(self) -> None:
        mock_response = make_mock_response(
            200,
            {
                "object": "list",
                "data": [{"id": "qwen3:8b", "object": "model"}],
            },
        )

        with patch.object(httpx.Client, "request", return_value=mock_response):
            with OllamaClient() as client:
                response = client.openai_list_models()

        assert isinstance(response, OpenAICompatModelsResponse)
        assert response.data[0].id == "qwen3:8b"

    def test_openai_chat_completions_non_stream(self) -> None:
        mock_response = make_mock_response(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "qwen3:8b",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "I will call a tool.",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "index": 0,
                                    "type": "function",
                                    "function": {
                                        "name": "lookup_company",
                                        "arguments": '{"company":"B2X"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )

        with patch.object(httpx.Client, "request", return_value=mock_response):
            with OllamaClient() as client:
                response = client.openai_chat_completions(
                    model="qwen3:8b",
                    messages=[{"role": "user", "content": "Lookup B2X"}],
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": "lookup_company",
                                "description": "Lookup company details.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "company": {"type": "string"},
                                    },
                                    "required": ["company"],
                                },
                            },
                        }
                    ],
                )

        assert isinstance(response, OpenAICompatChatCompletionsResponse)
        tool_call = response.choices[0].message.tool_calls[0]
        assert tool_call.id == "call_1"
        assert tool_call.index == 0
        assert tool_call.function.name == "lookup_company"

    def test_openai_chat_completions_stream(self) -> None:
        stream_context = make_sync_stream_context(
            [
                'data: {"id":"chunk-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}',
                'data: {"id":"chunk-2","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )

        with patch.object(httpx.Client, "stream", return_value=stream_context):
            with OllamaClient() as client:
                chunks = client.openai_chat_completions(
                    model="qwen3:8b",
                    messages=[{"role": "user", "content": "say hello"}],
                    stream=True,
                )
                parsed = list(chunks)

        assert len(parsed) == 2
        assert parsed[0].choices[0].delta.content == "Hel"
        assert parsed[1].choices[0].delta.content == "lo"

    def test_openai_embeddings(self) -> None:
        mock_response = make_mock_response(
            200,
            {
                "object": "list",
                "model": "qwen3-embedding:4b",
                "data": [
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": [0.1, 0.2, 0.3],
                    }
                ],
            },
        )

        with patch.object(httpx.Client, "request", return_value=mock_response):
            with OllamaClient() as client:
                response = client.openai_embeddings(
                    model="qwen3-embedding:4b",
                    input="B2X qualification",
                )

        assert isinstance(response, OpenAICompatEmbeddingsResponse)
        assert len(response.data) == 1
        assert response.data[0].index == 0


class TestAsyncOpenAICompat:
    """Validate async `/v1/*` helpers."""

    @pytest.mark.asyncio
    async def test_async_openai_list_models(self) -> None:
        mock_response = make_mock_response(
            200,
            {
                "object": "list",
                "data": [{"id": "qwen3:8b", "object": "model"}],
            },
        )

        with patch.object(
            httpx.AsyncClient,
            "request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            async with AsyncOllamaClient() as client:
                response = await client.openai_list_models()

        assert response.data[0].id == "qwen3:8b"

    @pytest.mark.asyncio
    async def test_async_openai_chat_stream(self) -> None:
        async_context = _AsyncStreamContext(
            [
                'data: {"id":"chunk-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"A"}}]}',
                'data: {"id":"chunk-2","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"B"},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )

        with patch.object(httpx.AsyncClient, "stream", return_value=async_context):
            async with AsyncOllamaClient() as client:
                stream = await client.openai_chat_completions(
                    model="qwen3:8b",
                    messages=[{"role": "user", "content": "say AB"}],
                    stream=True,
                )
                chunks = [chunk async for chunk in stream]

        assert len(chunks) == 2
        assert chunks[0].choices[0].delta.content == "A"
        assert chunks[1].choices[0].delta.content == "B"

    @pytest.mark.asyncio
    async def test_async_openai_embeddings(self) -> None:
        mock_response = make_mock_response(
            200,
            {
                "object": "list",
                "model": "qwen3-embedding:4b",
                "data": [
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": [0.2, 0.3, 0.4],
                    }
                ],
            },
        )

        with patch.object(
            httpx.AsyncClient,
            "request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            async with AsyncOllamaClient() as client:
                response = await client.openai_embeddings(
                    model="qwen3-embedding:4b",
                    input="B2X embedding",
                )

        assert response.data[0].embedding[0] == 0.2
