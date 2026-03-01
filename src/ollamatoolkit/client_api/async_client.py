# ./src/ollamatoolkit/client_api/async_client.py
"""
Public asynchronous Ollama client composed from endpoint-domain adapters.
Run: imported by `ollamatoolkit.client.AsyncOllamaClient`.
Inputs: Ollama endpoint arguments and optional transport/config overrides.
Outputs: typed async responses/stream generators for Ollama `/api/*` and `/v1/*` APIs.
Side effects: network calls and optional model-management mutations.
Operational notes: async method parity mirrors the synchronous client surface.
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Sequence, Union

from ollamatoolkit.client_api import (
    AsyncInferenceAPI,
    AsyncModelAPI,
    AsyncOpenAICompatAPI,
    AsyncTransport,
    AsyncWebAPI,
    default_headers,
    merge_headers,
    parse_host,
)
from ollamatoolkit.openai_types import (
    OpenAICompatChatCompletionsResponse,
    OpenAICompatCompletionsResponse,
    OpenAICompatEmbeddingsResponse,
    OpenAICompatMessage,
    OpenAICompatModelsResponse,
)
from ollamatoolkit.types import (
    ChatResponse,
    EmbedResponse,
    GenerateResponse,
    ListResponse,
    Message,
    ProcessResponse,
    ProgressResponse,
    ShowResponse,
    StatusResponse,
    StreamEvent,
    VersionResponse,
    WebFetchResponse,
    WebSearchResponse,
)


class AsyncOllamaClient:
    """Asynchronous Ollama client with `/api/*` and `/v1/*` support."""

    def __init__(
        self,
        host: Optional[str] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        resolved_host = parse_host(host or os.getenv("OLLAMA_HOST"))
        merged_headers = merge_headers(default_headers(), headers)

        self._transport = AsyncTransport(
            base_url=resolved_host,
            timeout=timeout,
            headers=merged_headers,
            **kwargs,
        )
        self._client = self._transport.client

        self._inference = AsyncInferenceAPI(self._transport)
        self._models = AsyncModelAPI(self._transport)
        self._web = AsyncWebAPI(self._transport)
        self._openai = AsyncOpenAICompatAPI(self._transport)

    async def __aenter__(self) -> "AsyncOllamaClient":
        return self

    async def __aexit__(
        self, exc_type: object, exc_val: object, exc_tb: object
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP transport."""
        await self._transport.close()

    # ---------------------------------------------------------------------
    # Inference (/api)
    # ---------------------------------------------------------------------

    async def generate(
        self, *args: Any, **kwargs: Any
    ) -> Union[GenerateResponse, AsyncIterator[GenerateResponse]]:
        """Proxy to async `/api/generate`."""
        return await self._inference.generate(*args, **kwargs)

    async def chat(
        self, *args: Any, **kwargs: Any
    ) -> Union[ChatResponse, AsyncIterator[ChatResponse]]:
        """Proxy to async `/api/chat`."""
        return await self._inference.chat(*args, **kwargs)

    async def embed(self, *args: Any, **kwargs: Any) -> EmbedResponse:
        """Proxy to async `/api/embed`."""
        return await self._inference.embed(*args, **kwargs)

    async def stream_generate_events(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[StreamEvent]:
        """Yield structured async generation stream events."""
        async for event in self._inference.stream_generate_events(*args, **kwargs):
            yield event

    async def stream_chat_events(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[StreamEvent]:
        """Yield structured async chat stream events."""
        async for event in self._inference.stream_chat_events(*args, **kwargs):
            yield event

    # ---------------------------------------------------------------------
    # Model management (/api)
    # ---------------------------------------------------------------------

    async def list(self) -> ListResponse:
        """List installed models."""
        return await self._models.list()

    async def version(self) -> VersionResponse:
        """Get Ollama server version."""
        return await self._models.version()

    async def show(self, model: str) -> ShowResponse:
        """Get model details."""
        return await self._models.show(model)

    async def ps(self) -> ProcessResponse:
        """List running models."""
        return await self._models.ps()

    async def pull(
        self,
        model: str,
        *,
        insecure: bool = False,
        stream: bool = False,
    ) -> Union[ProgressResponse, AsyncIterator[ProgressResponse]]:
        """Pull model data."""
        return await self._models.pull(model, insecure=insecure, stream=stream)

    async def push(
        self,
        model: str,
        *,
        insecure: bool = False,
        stream: bool = False,
    ) -> Union[ProgressResponse, AsyncIterator[ProgressResponse]]:
        """Push model data."""
        return await self._models.push(model, insecure=insecure, stream=stream)

    async def create(
        self,
        model: str,
        *,
        quantize: Optional[str] = None,
        from_: Optional[str] = None,
        files: Optional[dict[str, str]] = None,
        adapters: Optional[dict[str, str]] = None,
        template: Optional[str] = None,
        license: Optional[Union[str, List[str]]] = None,
        system: Optional[str] = None,
        parameters: Optional[Mapping[str, Any]] = None,
        messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
        stream: bool = False,
    ) -> Union[ProgressResponse, AsyncIterator[ProgressResponse]]:
        """Create a model."""
        return await self._models.create(
            model,
            quantize=quantize,
            from_=from_,
            files=files,
            adapters=adapters,
            template=template,
            license=license,
            system=system,
            parameters=parameters,
            messages=messages,
            stream=stream,
        )

    async def delete(self, model: str) -> StatusResponse:
        """Delete a model."""
        return await self._models.delete(model)

    async def copy(self, source: str, destination: str) -> StatusResponse:
        """Copy a model."""
        return await self._models.copy(source, destination)

    async def ensure_model_available(
        self,
        model: str,
        auto_pull: bool = True,
        stream_progress: bool = False,
    ) -> bool:
        """Ensure a model exists locally."""
        return await self._models.ensure_model_available(
            model,
            auto_pull=auto_pull,
            stream_progress=stream_progress,
        )

    async def get_model_details(self, model: str) -> Dict[str, Any]:
        """Get merged model metadata."""
        return await self._models.get_model_details(model)

    async def get_model_capabilities(self, model: str) -> List[str]:
        """Infer model capability tags."""
        return await self._models.get_model_capabilities(model)

    async def get_model_context_length(self, model: str) -> int:
        """Return model context length."""
        return await self._models.get_model_context_length(model)

    # ---------------------------------------------------------------------
    # Hosted web helpers
    # ---------------------------------------------------------------------

    async def web_search(self, query: str, max_results: int = 3) -> WebSearchResponse:
        """Search web via Ollama hosted web API."""
        return await self._web.web_search(query, max_results=max_results)

    async def web_fetch(self, url: str) -> WebFetchResponse:
        """Fetch URL via Ollama hosted web API."""
        return await self._web.web_fetch(url)

    # ---------------------------------------------------------------------
    # OpenAI-compatible (/v1)
    # ---------------------------------------------------------------------

    async def openai_list_models(self) -> OpenAICompatModelsResponse:
        """List models from `/v1/models`."""
        return await self._openai.list_models()

    async def openai_chat_completions(
        self,
        *,
        model: str,
        messages: Sequence[Union[OpenAICompatMessage, Mapping[str, Any]]],
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[
        OpenAICompatChatCompletionsResponse,
        AsyncIterator[OpenAICompatChatCompletionsResponse],
    ]:
        """Call `/v1/chat/completions` with typed responses."""
        return await self._openai.chat_completions(
            model=model,
            messages=messages,
            stream=stream,
            **kwargs,
        )

    async def openai_completions(
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
        """Call `/v1/completions` with typed responses."""
        return await self._openai.completions(
            model=model,
            prompt=prompt,
            stream=stream,
            **kwargs,
        )

    async def openai_embeddings(
        self,
        *,
        model: str,
        input: Union[str, Sequence[str]],
        **kwargs: Any,
    ) -> OpenAICompatEmbeddingsResponse:
        """Call `/v1/embeddings` with typed responses."""
        return await self._openai.embeddings(model=model, input=input, **kwargs)

    # Compatibility aliases
    chat_completions = openai_chat_completions
    completions = openai_completions
    embeddings_openai = openai_embeddings
    list_openai_models = openai_list_models
