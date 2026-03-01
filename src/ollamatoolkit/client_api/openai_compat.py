# ./src/ollamatoolkit/client_api/openai_compat.py
"""
OpenAI-compatible (`/v1/*`) endpoint adapters for OllamaToolkit clients.
Run: imported by sync/async client façades.
Inputs: OpenAI-style payloads for chat/completions/embeddings/model listing.
Outputs: typed OpenAI-compatible response models and streaming chunk iterators.
Side effects: network requests to Ollama's OpenAI-compatible API routes.
Operational notes: SSE stream parser handles `data: ...` chunk frames and `[DONE]` sentinel.
"""

from __future__ import annotations

import json
from typing import (
    Any,
    AsyncIterator,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

from ollamatoolkit.client_api.common import normalize_tools
from ollamatoolkit.client_api.transport import AsyncTransport, SyncTransport
from ollamatoolkit.openai_types import (
    OpenAICompatChatCompletionsRequest,
    OpenAICompatChatCompletionsResponse,
    OpenAICompatCompletionsRequest,
    OpenAICompatCompletionsResponse,
    OpenAICompatEmbeddingsRequest,
    OpenAICompatEmbeddingsResponse,
    OpenAICompatMessage,
    OpenAICompatModelsResponse,
)
from ollamatoolkit.types import ResponseError, Tool

T = TypeVar("T")


class SyncOpenAICompatAPI:
    """Synchronous adapter for OpenAI-compatible Ollama endpoints."""

    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def list_models(self) -> OpenAICompatModelsResponse:
        """List models from `/v1/models`."""
        return self._transport.request(OpenAICompatModelsResponse, "GET", "/v1/models")

    def chat_completions(
        self,
        *,
        model: str,
        messages: Sequence[Union[OpenAICompatMessage, Mapping[str, Any]]],
        tools: Optional[Sequence[Union[Tool, Mapping[str, Any]]]] = None,
        tool_choice: Optional[Union[str, Mapping[str, Any]]] = None,
        parallel_tool_calls: Optional[bool] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[
        OpenAICompatChatCompletionsResponse,
        Iterator[OpenAICompatChatCompletionsResponse],
    ]:
        """Call `/v1/chat/completions`."""
        request = OpenAICompatChatCompletionsRequest(
            model=model,
            messages=messages,
            tools=normalize_tools(tools) or None,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            stream=stream,
            **kwargs,
        )
        payload = request.model_dump(exclude_none=True)

        if stream:
            return self._stream_sse(
                OpenAICompatChatCompletionsResponse,
                "POST",
                "/v1/chat/completions",
                json=payload,
            )

        return self._transport.request(
            OpenAICompatChatCompletionsResponse,
            "POST",
            "/v1/chat/completions",
            json=payload,
        )

    def completions(
        self,
        *,
        model: str,
        prompt: Union[str, Sequence[str]],
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[
        OpenAICompatCompletionsResponse,
        Iterator[OpenAICompatCompletionsResponse],
    ]:
        """Call `/v1/completions`."""
        request = OpenAICompatCompletionsRequest(
            model=model,
            prompt=prompt,
            stream=stream,
            **kwargs,
        )
        payload = request.model_dump(exclude_none=True)

        if stream:
            return self._stream_sse(
                OpenAICompatCompletionsResponse,
                "POST",
                "/v1/completions",
                json=payload,
            )

        return self._transport.request(
            OpenAICompatCompletionsResponse,
            "POST",
            "/v1/completions",
            json=payload,
        )

    def embeddings(
        self,
        *,
        model: str,
        input: Union[str, Sequence[str]],
        **kwargs: Any,
    ) -> OpenAICompatEmbeddingsResponse:
        """Call `/v1/embeddings`."""
        request = OpenAICompatEmbeddingsRequest(
            model=model,
            input=input,
            **kwargs,
        )
        return self._transport.request(
            OpenAICompatEmbeddingsResponse,
            "POST",
            "/v1/embeddings",
            json=request.model_dump(exclude_none=True),
        )

    def _stream_sse(
        self,
        cls: Type[T],
        method: str,
        url: str,
        **kwargs: Any,
    ) -> Iterator[T]:
        """Parse SSE (`data:`) stream payloads into typed models."""
        for line in self._transport.stream_lines(method, url, **kwargs):
            data = _extract_sse_payload(line)
            if data is None:
                continue
            if data == "[DONE]":
                break
            payload = json.loads(data)
            if isinstance(payload, dict) and payload.get("error"):
                raise ResponseError(str(payload["error"]))
            yield cls(**payload)


class AsyncOpenAICompatAPI:
    """Asynchronous adapter for OpenAI-compatible Ollama endpoints."""

    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def list_models(self) -> OpenAICompatModelsResponse:
        """List models from `/v1/models`."""
        return await self._transport.request(
            OpenAICompatModelsResponse,
            "GET",
            "/v1/models",
        )

    async def chat_completions(
        self,
        *,
        model: str,
        messages: Sequence[Union[OpenAICompatMessage, Mapping[str, Any]]],
        tools: Optional[Sequence[Union[Tool, Mapping[str, Any]]]] = None,
        tool_choice: Optional[Union[str, Mapping[str, Any]]] = None,
        parallel_tool_calls: Optional[bool] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[
        OpenAICompatChatCompletionsResponse,
        AsyncIterator[OpenAICompatChatCompletionsResponse],
    ]:
        """Call `/v1/chat/completions` asynchronously."""
        request = OpenAICompatChatCompletionsRequest(
            model=model,
            messages=messages,
            tools=normalize_tools(tools) or None,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            stream=stream,
            **kwargs,
        )
        payload = request.model_dump(exclude_none=True)

        if stream:
            return self._stream_sse(
                OpenAICompatChatCompletionsResponse,
                "POST",
                "/v1/chat/completions",
                json=payload,
            )

        return await self._transport.request(
            OpenAICompatChatCompletionsResponse,
            "POST",
            "/v1/chat/completions",
            json=payload,
        )

    async def completions(
        self,
        *,
        model: str,
        prompt: Union[str, Sequence[str]],
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[
        OpenAICompatCompletionsResponse,
        AsyncIterator[OpenAICompatCompletionsResponse],
    ]:
        """Call `/v1/completions` asynchronously."""
        request = OpenAICompatCompletionsRequest(
            model=model,
            prompt=prompt,
            stream=stream,
            **kwargs,
        )
        payload = request.model_dump(exclude_none=True)

        if stream:
            return self._stream_sse(
                OpenAICompatCompletionsResponse,
                "POST",
                "/v1/completions",
                json=payload,
            )

        return await self._transport.request(
            OpenAICompatCompletionsResponse,
            "POST",
            "/v1/completions",
            json=payload,
        )

    async def embeddings(
        self,
        *,
        model: str,
        input: Union[str, Sequence[str]],
        **kwargs: Any,
    ) -> OpenAICompatEmbeddingsResponse:
        """Call `/v1/embeddings` asynchronously."""
        request = OpenAICompatEmbeddingsRequest(
            model=model,
            input=input,
            **kwargs,
        )
        return await self._transport.request(
            OpenAICompatEmbeddingsResponse,
            "POST",
            "/v1/embeddings",
            json=request.model_dump(exclude_none=True),
        )

    async def _stream_sse(
        self,
        cls: Type[T],
        method: str,
        url: str,
        **kwargs: Any,
    ) -> AsyncIterator[T]:
        """Parse SSE (`data:`) stream payloads into typed models."""
        async for line in self._transport.stream_lines(method, url, **kwargs):
            data = _extract_sse_payload(line)
            if data is None:
                continue
            if data == "[DONE]":
                break
            payload = json.loads(data)
            if isinstance(payload, dict) and payload.get("error"):
                raise ResponseError(str(payload["error"]))
            yield cls(**payload)


def _extract_sse_payload(raw_line: str) -> Optional[str]:
    """Extract payload content from a single SSE line."""
    line = raw_line.strip()
    if not line or line.startswith(":"):
        return None
    if line.lower().startswith("data:"):
        return line[5:].strip()
    return line
