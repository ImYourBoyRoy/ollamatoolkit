# ./src/ollamatoolkit/client_api/inference.py
"""
Inference endpoint adapters for OllamaToolkit clients.
Run: imported by sync/async client façades.
Inputs: generation/chat/embedding request arguments.
Outputs: typed responses or streaming events/chunks.
Side effects: network requests to `/api/generate`, `/api/chat`, `/api/embed`.
Operational notes: stream event helpers provide richer chunk-level telemetry hooks.
"""

from __future__ import annotations

from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
    cast,
)

from ollamatoolkit.client_api.common import normalize_messages, normalize_tools
from ollamatoolkit.client_api.transport import AsyncTransport, SyncTransport
from ollamatoolkit.types import (
    ChatRequest,
    ChatResponse,
    EmbedRequest,
    EmbedResponse,
    GenerateRequest,
    GenerateResponse,
    Message,
    Options,
    StreamEvent,
    Tool,
)


class SyncInferenceAPI:
    """Synchronous adapter for inference endpoints."""

    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def generate(
        self,
        model: str,
        prompt: str = "",
        *,
        suffix: str = "",
        system: str = "",
        template: str = "",
        context: Optional[Sequence[int]] = None,
        stream: bool = False,
        raw: bool = False,
        format: Optional[Union[Literal["", "json"], Dict[str, Any], str]] = None,
        images: Optional[Sequence[str]] = None,
        options: Optional[Union[Mapping[str, Any], Options]] = None,
        keep_alive: Optional[Union[float, str]] = None,
        think: Optional[Union[bool, Literal["low", "medium", "high"], str]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
    ) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Call `/api/generate` with typed request/response handling."""
        request = GenerateRequest(
            model=model,
            prompt=prompt or None,
            suffix=suffix or None,
            system=system or None,
            template=template or None,
            context=context,
            stream=stream,
            raw=raw or None,
            format=format,
            images=images,
            options=options,
            keep_alive=keep_alive,
            think=think,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
        payload = request.model_dump(exclude_none=True)

        if stream:
            return self._transport.stream(
                GenerateResponse,
                "POST",
                "/api/generate",
                json=payload,
            )

        return self._transport.request(
            GenerateResponse,
            "POST",
            "/api/generate",
            json=payload,
        )

    def chat(
        self,
        model: str,
        messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
        *,
        tools: Optional[
            Sequence[Union[Mapping[str, Any], Tool, Callable[..., Any]]]
        ] = None,
        stream: bool = False,
        format: Optional[Union[Literal["", "json"], Dict[str, Any], str]] = None,
        options: Optional[Union[Mapping[str, Any], Options]] = None,
        keep_alive: Optional[Union[float, str]] = None,
        think: Optional[Union[bool, Literal["low", "medium", "high"], str]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
    ) -> Union[ChatResponse, Iterator[ChatResponse]]:
        """Call `/api/chat` with typed request/response handling."""
        request = ChatRequest(
            model=model,
            messages=normalize_messages(messages) or None,
            tools=normalize_tools(tools) or None,
            stream=stream,
            format=format,
            options=options,
            keep_alive=keep_alive,
            think=think,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
        payload = request.model_dump(exclude_none=True)

        if stream:
            return self._transport.stream(
                ChatResponse, "POST", "/api/chat", json=payload
            )

        return self._transport.request(ChatResponse, "POST", "/api/chat", json=payload)

    def embed(
        self,
        model: str,
        input: Union[str, Sequence[str]],
        *,
        truncate: Optional[bool] = None,
        options: Optional[Union[Mapping[str, Any], Options]] = None,
        keep_alive: Optional[Union[float, str]] = None,
        dimensions: Optional[int] = None,
    ) -> EmbedResponse:
        """Call `/api/embed` for single or batched embedding generation."""
        request = EmbedRequest(
            model=model,
            input=input,
            truncate=truncate,
            options=options,
            keep_alive=keep_alive,
            dimensions=dimensions,
        )
        return self._transport.request(
            EmbedResponse,
            "POST",
            "/api/embed",
            json=request.model_dump(exclude_none=True),
        )

    def stream_generate_events(
        self, *args: Any, **kwargs: Any
    ) -> Iterator[StreamEvent]:
        """Emit structured stream events from `/api/generate` chunks."""
        kwargs["stream"] = True
        chunks = cast(Iterator[GenerateResponse], self.generate(*args, **kwargs))

        for index, chunk in enumerate(chunks):
            if chunk.thinking:
                yield StreamEvent(
                    event="thinking",
                    chunk_index=index,
                    thinking=chunk.thinking,
                    done=bool(chunk.done),
                    raw_chunk=chunk.model_dump(exclude_none=True),
                )
            if chunk.response:
                yield StreamEvent(
                    event="token",
                    chunk_index=index,
                    text=chunk.response,
                    done=bool(chunk.done),
                    raw_chunk=chunk.model_dump(exclude_none=True),
                )
            if chunk.done:
                yield StreamEvent(
                    event="done",
                    chunk_index=index,
                    done=True,
                    raw_chunk=chunk.model_dump(exclude_none=True),
                )

    def stream_chat_events(self, *args: Any, **kwargs: Any) -> Iterator[StreamEvent]:
        """Emit structured stream events from `/api/chat` chunks."""
        kwargs["stream"] = True
        chunks = cast(Iterator[ChatResponse], self.chat(*args, **kwargs))

        for index, chunk in enumerate(chunks):
            message = chunk.message
            if message.thinking:
                yield StreamEvent(
                    event="thinking",
                    chunk_index=index,
                    thinking=message.thinking,
                    done=bool(chunk.done),
                    raw_chunk=chunk.model_dump(exclude_none=True),
                )
            if message.content:
                yield StreamEvent(
                    event="token",
                    chunk_index=index,
                    text=message.content,
                    done=bool(chunk.done),
                    raw_chunk=chunk.model_dump(exclude_none=True),
                )
            if message.tool_calls:
                yield StreamEvent(
                    event="tool_call",
                    chunk_index=index,
                    tool_calls=message.tool_calls,
                    done=bool(chunk.done),
                    raw_chunk=chunk.model_dump(exclude_none=True),
                )
            if chunk.done:
                yield StreamEvent(
                    event="done",
                    chunk_index=index,
                    done=True,
                    raw_chunk=chunk.model_dump(exclude_none=True),
                )


class AsyncInferenceAPI:
    """Asynchronous adapter for inference endpoints."""

    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def generate(
        self,
        model: str,
        prompt: str = "",
        *,
        suffix: str = "",
        system: str = "",
        template: str = "",
        context: Optional[Sequence[int]] = None,
        stream: bool = False,
        raw: bool = False,
        format: Optional[Union[Literal["", "json"], Dict[str, Any], str]] = None,
        images: Optional[Sequence[str]] = None,
        options: Optional[Union[Mapping[str, Any], Options]] = None,
        keep_alive: Optional[Union[float, str]] = None,
        think: Optional[Union[bool, Literal["low", "medium", "high"], str]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
    ) -> Union[GenerateResponse, AsyncIterator[GenerateResponse]]:
        """Call `/api/generate` asynchronously."""
        request = GenerateRequest(
            model=model,
            prompt=prompt or None,
            suffix=suffix or None,
            system=system or None,
            template=template or None,
            context=context,
            stream=stream,
            raw=raw or None,
            format=format,
            images=images,
            options=options,
            keep_alive=keep_alive,
            think=think,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
        payload = request.model_dump(exclude_none=True)

        if stream:
            return self._transport.stream(
                GenerateResponse,
                "POST",
                "/api/generate",
                json=payload,
            )

        return await self._transport.request(
            GenerateResponse,
            "POST",
            "/api/generate",
            json=payload,
        )

    async def chat(
        self,
        model: str,
        messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
        *,
        tools: Optional[
            Sequence[Union[Mapping[str, Any], Tool, Callable[..., Any]]]
        ] = None,
        stream: bool = False,
        format: Optional[Union[Literal["", "json"], Dict[str, Any], str]] = None,
        options: Optional[Union[Mapping[str, Any], Options]] = None,
        keep_alive: Optional[Union[float, str]] = None,
        think: Optional[Union[bool, Literal["low", "medium", "high"], str]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
    ) -> Union[ChatResponse, AsyncIterator[ChatResponse]]:
        """Call `/api/chat` asynchronously."""
        request = ChatRequest(
            model=model,
            messages=normalize_messages(messages) or None,
            tools=normalize_tools(tools) or None,
            stream=stream,
            format=format,
            options=options,
            keep_alive=keep_alive,
            think=think,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
        payload = request.model_dump(exclude_none=True)

        if stream:
            return self._transport.stream(
                ChatResponse, "POST", "/api/chat", json=payload
            )

        return await self._transport.request(
            ChatResponse,
            "POST",
            "/api/chat",
            json=payload,
        )

    async def embed(
        self,
        model: str,
        input: Union[str, Sequence[str]],
        *,
        truncate: Optional[bool] = None,
        options: Optional[Union[Mapping[str, Any], Options]] = None,
        keep_alive: Optional[Union[float, str]] = None,
        dimensions: Optional[int] = None,
    ) -> EmbedResponse:
        """Call `/api/embed` asynchronously for embedding generation."""
        request = EmbedRequest(
            model=model,
            input=input,
            truncate=truncate,
            options=options,
            keep_alive=keep_alive,
            dimensions=dimensions,
        )
        return await self._transport.request(
            EmbedResponse,
            "POST",
            "/api/embed",
            json=request.model_dump(exclude_none=True),
        )

    async def stream_generate_events(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[StreamEvent]:
        """Emit structured stream events from async `/api/generate` chunks."""
        kwargs["stream"] = True
        chunks = cast(
            AsyncIterator[GenerateResponse],
            await self.generate(*args, **kwargs),
        )
        async for index, chunk in _aenumerate(chunks):
            if chunk.thinking:
                yield StreamEvent(
                    event="thinking",
                    chunk_index=index,
                    thinking=chunk.thinking,
                    done=bool(chunk.done),
                    raw_chunk=chunk.model_dump(exclude_none=True),
                )
            if chunk.response:
                yield StreamEvent(
                    event="token",
                    chunk_index=index,
                    text=chunk.response,
                    done=bool(chunk.done),
                    raw_chunk=chunk.model_dump(exclude_none=True),
                )
            if chunk.done:
                yield StreamEvent(
                    event="done",
                    chunk_index=index,
                    done=True,
                    raw_chunk=chunk.model_dump(exclude_none=True),
                )

    async def stream_chat_events(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[StreamEvent]:
        """Emit structured stream events from async `/api/chat` chunks."""
        kwargs["stream"] = True
        chunks = cast(
            AsyncIterator[ChatResponse],
            await self.chat(*args, **kwargs),
        )
        async for index, chunk in _aenumerate(chunks):
            message = chunk.message
            if message.thinking:
                yield StreamEvent(
                    event="thinking",
                    chunk_index=index,
                    thinking=message.thinking,
                    done=bool(chunk.done),
                    raw_chunk=chunk.model_dump(exclude_none=True),
                )
            if message.content:
                yield StreamEvent(
                    event="token",
                    chunk_index=index,
                    text=message.content,
                    done=bool(chunk.done),
                    raw_chunk=chunk.model_dump(exclude_none=True),
                )
            if message.tool_calls:
                yield StreamEvent(
                    event="tool_call",
                    chunk_index=index,
                    tool_calls=message.tool_calls,
                    done=bool(chunk.done),
                    raw_chunk=chunk.model_dump(exclude_none=True),
                )
            if chunk.done:
                yield StreamEvent(
                    event="done",
                    chunk_index=index,
                    done=True,
                    raw_chunk=chunk.model_dump(exclude_none=True),
                )


async def _aenumerate(
    async_iterable: AsyncIterator[Any],
) -> AsyncIterator[tuple[int, Any]]:
    """Async equivalent of enumerate()."""
    index = 0
    async for item in async_iterable:
        yield index, item
        index += 1
