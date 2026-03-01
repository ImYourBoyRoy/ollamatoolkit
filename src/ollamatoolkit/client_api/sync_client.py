# ./src/ollamatoolkit/client_api/sync_client.py
"""
Public synchronous Ollama client composed from endpoint-domain adapters.
Run: imported by `ollamatoolkit.client.OllamaClient`.
Inputs: Ollama endpoint arguments and optional transport/config overrides.
Outputs: typed responses/iterators for Ollama `/api/*` and `/v1/*` APIs.
Side effects: network calls and optional model-management mutations.
Operational notes: keeps backward-compatible method signatures while avoiding monolith code.
"""

from __future__ import annotations

import os
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
    overload,
)

from ollamatoolkit.client_api import (
    SyncInferenceAPI,
    SyncModelAPI,
    SyncOpenAICompatAPI,
    SyncTransport,
    SyncWebAPI,
    default_headers,
    merge_headers,
    parse_host,
)
from ollamatoolkit.openai_types import (
    OpenAICompatEmbeddingsResponse,
    OpenAICompatModelsResponse,
)
from ollamatoolkit.types import (
    ChatResponse,
    EmbedResponse,
    GenerateResponse,
    ListResponse,
    Message,
    Options,
    ProcessResponse,
    ProgressResponse,
    ShowResponse,
    StatusResponse,
    StreamEvent,
    Tool,
    VersionResponse,
    WebFetchResponse,
    WebSearchResponse,
)


class OllamaClient:
    """Synchronous Ollama client with `/api/*` and `/v1/*` support."""

    def __init__(
        self,
        host: Optional[str] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        resolved_host = parse_host(host or os.getenv("OLLAMA_HOST"))
        merged_headers = merge_headers(default_headers(), headers)

        self._transport = SyncTransport(
            base_url=resolved_host,
            timeout=timeout,
            headers=merged_headers,
            **kwargs,
        )
        self._client = self._transport.client

        self._inference = SyncInferenceAPI(self._transport)
        self._models = SyncModelAPI(self._transport)
        self._web = SyncWebAPI(self._transport)
        self._openai = SyncOpenAICompatAPI(self._transport)

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP transport."""
        self._transport.close()

    # ---------------------------------------------------------------------
    # Inference (/api)
    # ---------------------------------------------------------------------

    @overload
    def generate(
        self,
        model: str,
        prompt: str = "",
        *,
        suffix: str = "",
        system: str = "",
        template: str = "",
        context: Optional[Sequence[int]] = None,
        stream: Literal[False] = False,
        raw: bool = False,
        format: Optional[Union[str, Dict[str, Any]]] = None,
        images: Optional[Sequence[str]] = None,
        options: Optional[Union[Mapping[str, Any], Options]] = None,
        keep_alive: Optional[Union[float, str]] = None,
        think: Optional[Union[bool, str]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
    ) -> GenerateResponse: ...

    @overload
    def generate(
        self,
        model: str,
        prompt: str = "",
        *,
        suffix: str = "",
        system: str = "",
        template: str = "",
        context: Optional[Sequence[int]] = None,
        stream: Literal[True],
        raw: bool = False,
        format: Optional[Union[str, Dict[str, Any]]] = None,
        images: Optional[Sequence[str]] = None,
        options: Optional[Union[Mapping[str, Any], Options]] = None,
        keep_alive: Optional[Union[float, str]] = None,
        think: Optional[Union[bool, str]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
    ) -> Iterator[GenerateResponse]: ...

    def generate(
        self, *args: Any, **kwargs: Any
    ) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Proxy to `/api/generate`."""
        return self._inference.generate(*args, **kwargs)

    @overload
    def chat(
        self,
        model: str,
        messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
        *,
        tools: Optional[
            Sequence[Union[Mapping[str, Any], Tool, Callable[..., Any]]]
        ] = None,
        stream: Literal[False] = False,
        format: Optional[Union[str, Dict[str, Any]]] = None,
        options: Optional[Union[Mapping[str, Any], Options]] = None,
        keep_alive: Optional[Union[float, str]] = None,
        think: Optional[Union[bool, str]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
    ) -> ChatResponse: ...

    @overload
    def chat(
        self,
        model: str,
        messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
        *,
        tools: Optional[
            Sequence[Union[Mapping[str, Any], Tool, Callable[..., Any]]]
        ] = None,
        stream: Literal[True],
        format: Optional[Union[str, Dict[str, Any]]] = None,
        options: Optional[Union[Mapping[str, Any], Options]] = None,
        keep_alive: Optional[Union[float, str]] = None,
        think: Optional[Union[bool, str]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
    ) -> Iterator[ChatResponse]: ...

    def chat(
        self, *args: Any, **kwargs: Any
    ) -> Union[ChatResponse, Iterator[ChatResponse]]:
        """Proxy to `/api/chat`."""
        return self._inference.chat(*args, **kwargs)

    def embed(self, *args: Any, **kwargs: Any) -> EmbedResponse:
        """Proxy to `/api/embed`."""
        return self._inference.embed(*args, **kwargs)

    def stream_generate_events(
        self, *args: Any, **kwargs: Any
    ) -> Iterator[StreamEvent]:
        """Yield structured generation stream events."""
        return self._inference.stream_generate_events(*args, **kwargs)

    def stream_chat_events(self, *args: Any, **kwargs: Any) -> Iterator[StreamEvent]:
        """Yield structured chat stream events."""
        return self._inference.stream_chat_events(*args, **kwargs)

    # ---------------------------------------------------------------------
    # Model management (/api)
    # ---------------------------------------------------------------------

    def list(self) -> ListResponse:
        """List installed models."""
        return self._models.list()

    def version(self) -> VersionResponse:
        """Get Ollama server version."""
        return self._models.version()

    def show(self, model: str) -> ShowResponse:
        """Get model details."""
        return self._models.show(model)

    def ps(self) -> ProcessResponse:
        """List running models."""
        return self._models.ps()

    @overload
    def pull(
        self,
        model: str,
        *,
        insecure: bool = False,
        stream: Literal[False] = False,
    ) -> ProgressResponse: ...

    @overload
    def pull(
        self,
        model: str,
        *,
        insecure: bool = False,
        stream: Literal[True],
    ) -> Iterator[ProgressResponse]: ...

    def pull(
        self,
        model: str,
        *,
        insecure: bool = False,
        stream: bool = False,
    ) -> Union[ProgressResponse, Iterator[ProgressResponse]]:
        """Pull model data."""
        return self._models.pull(model, insecure=insecure, stream=stream)

    @overload
    def push(
        self,
        model: str,
        *,
        insecure: bool = False,
        stream: Literal[False] = False,
    ) -> ProgressResponse: ...

    @overload
    def push(
        self,
        model: str,
        *,
        insecure: bool = False,
        stream: Literal[True],
    ) -> Iterator[ProgressResponse]: ...

    def push(
        self,
        model: str,
        *,
        insecure: bool = False,
        stream: bool = False,
    ) -> Union[ProgressResponse, Iterator[ProgressResponse]]:
        """Push model data."""
        return self._models.push(model, insecure=insecure, stream=stream)

    @overload
    def create(
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
        parameters: Optional[Union[Mapping[str, Any], Options]] = None,
        messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
        stream: Literal[False] = False,
    ) -> ProgressResponse: ...

    @overload
    def create(
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
        parameters: Optional[Union[Mapping[str, Any], Options]] = None,
        messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
        stream: Literal[True],
    ) -> Iterator[ProgressResponse]: ...

    def create(
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
        parameters: Optional[Union[Mapping[str, Any], Options]] = None,
        messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
        stream: bool = False,
    ) -> Union[ProgressResponse, Iterator[ProgressResponse]]:
        """Create a model."""
        return self._models.create(
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

    def delete(self, model: str) -> StatusResponse:
        """Delete a model."""
        return self._models.delete(model)

    def copy(self, source: str, destination: str) -> StatusResponse:
        """Copy a model."""
        return self._models.copy(source, destination)

    def ensure_model_available(
        self,
        model: str,
        auto_pull: bool = True,
        stream_progress: bool = False,
    ) -> bool:
        """Ensure a model exists locally."""
        return self._models.ensure_model_available(
            model,
            auto_pull=auto_pull,
            stream_progress=stream_progress,
        )

    def get_model_details(self, model: str) -> Dict[str, Any]:
        """Get merged model metadata."""
        return self._models.get_model_details(model)

    def get_model_capabilities(self, model: str) -> List[str]:
        """Infer model capability tags."""
        return self._models.get_model_capabilities(model)

    def get_model_context_length(self, model: str) -> int:
        """Return model context length."""
        return self._models.get_model_context_length(model)

    # ---------------------------------------------------------------------
    # Hosted web helpers
    # ---------------------------------------------------------------------

    def web_search(self, query: str, max_results: int = 3) -> WebSearchResponse:
        """Search web via Ollama hosted web API."""
        return self._web.web_search(query, max_results=max_results)

    def web_fetch(self, url: str) -> WebFetchResponse:
        """Fetch URL via Ollama hosted web API."""
        return self._web.web_fetch(url)

    # ---------------------------------------------------------------------
    # OpenAI-compatible (/v1)
    # ---------------------------------------------------------------------

    def openai_list_models(self) -> OpenAICompatModelsResponse:
        """List models from `/v1/models`."""
        return self._openai.list_models()

    def openai_chat_completions(self, *args: Any, **kwargs: Any):
        """Call `/v1/chat/completions` with typed responses."""
        return self._openai.chat_completions(*args, **kwargs)

    def openai_completions(self, *args: Any, **kwargs: Any):
        """Call `/v1/completions` with typed responses."""
        return self._openai.completions(*args, **kwargs)

    def openai_embeddings(
        self, *args: Any, **kwargs: Any
    ) -> OpenAICompatEmbeddingsResponse:
        """Call `/v1/embeddings` with typed responses."""
        return self._openai.embeddings(*args, **kwargs)

    # Compatibility aliases
    chat_completions = openai_chat_completions
    completions = openai_completions
    embeddings_openai = openai_embeddings
    list_openai_models = openai_list_models
