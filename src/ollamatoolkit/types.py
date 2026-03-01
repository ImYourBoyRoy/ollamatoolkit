# ./src/ollamatoolkit/types.py
"""
Ollama Toolkit - Type Definitions
=================================
Pydantic models for the Ollama API based on the official ollama-python client.

These types provide:
- Strong type safety for API interactions
- Automatic validation of requests/responses
- Serialization with `model_dump(exclude_none=True)`
- Subscript access for backward compatibility

Inputs: None (pure type definitions)
Outputs: Pydantic model classes for Ollama API
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Union

from pydantic import BaseModel, ByteSize, ConfigDict, Field, model_serializer


class SubscriptableBaseModel(BaseModel):
    """Base model with dict-like access for backward compatibility."""

    def __getitem__(self, key: str) -> Any:
        if key in self:
            return getattr(self, key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        if key in self.model_fields_set:
            return True
        if value := self.__class__.model_fields.get(key):
            return value.default is not None
        return False

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key) if hasattr(self, key) else default


# =============================================================================
# Options & Configuration
# =============================================================================


class Options(SubscriptableBaseModel):
    """Model inference options for Ollama API calls."""

    # Load time options
    numa: Optional[bool] = None
    num_ctx: Optional[int] = None
    num_batch: Optional[int] = None
    num_gpu: Optional[int] = None
    main_gpu: Optional[int] = None
    low_vram: Optional[bool] = None
    f16_kv: Optional[bool] = None
    logits_all: Optional[bool] = None
    vocab_only: Optional[bool] = None
    use_mmap: Optional[bool] = None
    use_mlock: Optional[bool] = None
    embedding_only: Optional[bool] = None
    num_thread: Optional[int] = None

    # Runtime options
    num_keep: Optional[int] = None
    seed: Optional[int] = None
    num_predict: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    tfs_z: Optional[float] = None
    typical_p: Optional[float] = None
    repeat_last_n: Optional[int] = None
    temperature: Optional[float] = None
    repeat_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    mirostat: Optional[int] = None
    mirostat_tau: Optional[float] = None
    mirostat_eta: Optional[float] = None
    penalize_newline: Optional[bool] = None
    stop: Optional[Sequence[str]] = None


# =============================================================================
# Messages & Tools
# =============================================================================


class Message(SubscriptableBaseModel):
    """Chat message for the Ollama API."""

    role: str
    "Role: 'user', 'assistant', 'system', or 'tool'."

    content: Optional[str] = None
    "Message content. May be fragments when streaming."

    thinking: Optional[str] = None
    "Thinking content for reasoning models."

    images: Optional[Sequence[str]] = None
    "Base64-encoded image data for multimodal models."

    tool_name: Optional[str] = None
    "Name of executed tool (for tool role)."

    tool_call_id: Optional[str] = None
    "Call id correlation value for tool-role follow-up messages."

    index: Optional[int] = None
    "Stream chunk/tool call index when provided by upstream responses."

    class ToolCall(SubscriptableBaseModel):
        """Tool call made by the model."""

        class Function(SubscriptableBaseModel):
            """Function details for a tool call."""

            name: str
            arguments: Union[str, Mapping[str, Any]]

        id: Optional[str] = None
        "Tool call id from Ollama/OpenAI-compatible responses."

        index: Optional[int] = None
        "Tool call index used for parallel-call chunking."

        type: Optional[str] = "function"
        "Tool call type. Defaults to `function`."

        function: "Message.ToolCall.Function"

    tool_calls: Optional[Sequence[ToolCall]] = None
    "Tool calls requested by the model."


class Tool(SubscriptableBaseModel):
    """Tool definition for function calling."""

    type: Optional[str] = "function"

    class Function(SubscriptableBaseModel):
        """Function definition within a tool."""

        name: Optional[str] = None
        description: Optional[str] = None

        class Parameters(SubscriptableBaseModel):
            """JSON Schema parameters for a function."""

            model_config = ConfigDict(populate_by_name=True)
            type: Optional[Literal["object"]] = "object"
            defs: Optional[Any] = Field(None, alias="$defs")
            items: Optional[Any] = None
            required: Optional[Sequence[str]] = None

            class Property(SubscriptableBaseModel):
                """Property definition within parameters."""

                model_config = ConfigDict(arbitrary_types_allowed=True)
                type: Optional[Union[str, Sequence[str]]] = None
                items: Optional[Any] = None
                description: Optional[str] = None
                enum: Optional[Sequence[Any]] = None

            properties: Optional[Mapping[str, Property]] = None

        parameters: Optional[Parameters] = None

    function: Optional[Function] = None


# =============================================================================
# Requests
# =============================================================================


class BaseRequest(SubscriptableBaseModel):
    """Base request with required model field."""

    model: str = Field(..., min_length=1)
    "Model to use for the request."


class BaseStreamableRequest(BaseRequest):
    """Request that supports streaming."""

    stream: Optional[bool] = None


class BaseGenerateRequest(BaseStreamableRequest):
    """Base for generate/chat requests."""

    options: Optional[Union[Mapping[str, Any], Options]] = None
    format: Optional[Union[Literal["", "json"], Dict[str, Any], str]] = None
    keep_alive: Optional[Union[float, str]] = None


class GenerateRequest(BaseGenerateRequest):
    """Request for /api/generate endpoint."""

    prompt: Optional[str] = None
    suffix: Optional[str] = None
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[Sequence[int]] = None
    raw: Optional[bool] = None
    images: Optional[Sequence[str]] = None
    think: Optional[Union[bool, Literal["low", "medium", "high"], str]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None


class ChatRequest(BaseGenerateRequest):
    """Request for /api/chat endpoint."""

    @model_serializer(mode="wrap")
    def serialize_model(self, nxt):
        output = nxt(self)
        if output.get("tools"):
            for tool in output["tools"]:
                if (
                    "function" in tool
                    and "parameters" in tool["function"]
                    and "defs" in tool["function"]["parameters"]
                ):
                    tool["function"]["parameters"]["$defs"] = tool["function"][
                        "parameters"
                    ].pop("defs")
        return output

    messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None
    tools: Optional[Sequence[Union[Tool, Mapping[str, Any]]]] = None
    think: Optional[Union[bool, Literal["low", "medium", "high"], str]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None


class EmbedRequest(BaseRequest):
    """Request for /api/embed endpoint."""

    input: Union[str, Sequence[str]]
    truncate: Optional[bool] = None
    options: Optional[Union[Mapping[str, Any], Options]] = None
    keep_alive: Optional[Union[float, str]] = None
    dimensions: Optional[int] = None


class PullRequest(BaseStreamableRequest):
    """Request for /api/pull endpoint."""

    insecure: Optional[bool] = None


class PushRequest(BaseStreamableRequest):
    """Request for /api/push endpoint."""

    insecure: Optional[bool] = None


class CreateRequest(BaseStreamableRequest):
    """Request for /api/create endpoint."""

    @model_serializer(mode="wrap")
    def serialize_model(self, nxt):
        output = nxt(self)
        if "from_" in output:
            output["from"] = output.pop("from_")
        return output

    quantize: Optional[str] = None
    from_: Optional[str] = None
    files: Optional[Dict[str, str]] = None
    adapters: Optional[Dict[str, str]] = None
    template: Optional[str] = None
    license: Optional[Union[str, List[str]]] = None
    system: Optional[str] = None
    parameters: Optional[Union[Mapping[str, Any], Options]] = None
    messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None


class CopyRequest(BaseModel):
    """Request to copy a model."""

    source: str
    destination: str


class DeleteRequest(BaseRequest):
    """Request to delete a model."""

    pass


class ShowRequest(BaseRequest):
    """Request to show model info."""

    pass


class WebSearchRequest(SubscriptableBaseModel):
    """Request for web search."""

    query: str
    max_results: Optional[int] = None


class WebFetchRequest(SubscriptableBaseModel):
    """Request to fetch a URL."""

    url: str


# =============================================================================
# Responses
# =============================================================================


class BaseGenerateResponse(SubscriptableBaseModel):
    """Base response for generate/chat."""

    model: Optional[str] = None
    created_at: Optional[str] = None
    done: Optional[bool] = None
    done_reason: Optional[str] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class TokenLogprob(SubscriptableBaseModel):
    """Token log probability."""

    token: str
    logprob: float


class Logprob(TokenLogprob):
    """Log probability with alternatives."""

    top_logprobs: Optional[Sequence[TokenLogprob]] = None


class GenerateResponse(BaseGenerateResponse):
    """Response from /api/generate."""

    response: str = ""
    thinking: Optional[str] = None
    context: Optional[Sequence[int]] = None
    logprobs: Optional[Sequence[Logprob]] = None


class ChatResponse(BaseGenerateResponse):
    """Response from /api/chat."""

    message: Message = Field(default_factory=lambda: Message(role="assistant"))
    logprobs: Optional[Sequence[Logprob]] = None


class EmbedResponse(BaseGenerateResponse):
    """Response from /api/embed."""

    embeddings: Sequence[Sequence[float]] = Field(default_factory=list)


class StreamEvent(SubscriptableBaseModel):
    """Structured stream event emitted from streaming helper APIs."""

    event: Literal["token", "thinking", "tool_call", "done"]
    chunk_index: int
    text: Optional[str] = None
    thinking: Optional[str] = None
    tool_calls: Optional[Sequence[Message.ToolCall]] = None
    done: bool = False
    raw_chunk: Optional[Dict[str, Any]] = None


class ModelDetails(SubscriptableBaseModel):
    """Details about a model."""

    parent_model: Optional[str] = None
    format: Optional[str] = None
    family: Optional[str] = None
    families: Optional[Sequence[str]] = None
    parameter_size: Optional[str] = None
    quantization_level: Optional[str] = None


class ListResponse(SubscriptableBaseModel):
    """Response listing available models."""

    class Model(SubscriptableBaseModel):
        """Model in list response."""

        model: Optional[str] = None
        name: Optional[str] = None
        modified_at: Optional[datetime] = None
        digest: Optional[str] = None
        size: Optional[ByteSize] = None
        details: Optional[ModelDetails] = None

    models: Sequence[Model] = Field(default_factory=list)


class ShowResponse(SubscriptableBaseModel):
    """Response from /api/show."""

    modified_at: Optional[datetime] = None
    template: Optional[str] = None
    modelfile: Optional[str] = None
    license: Optional[str] = None
    details: Optional[ModelDetails] = None
    modelinfo: Optional[Mapping[str, Any]] = Field(None, alias="model_info")
    parameters: Optional[str] = None
    capabilities: Optional[List[str]] = None


class StatusResponse(SubscriptableBaseModel):
    """Generic status response."""

    status: Optional[str] = None


class ProgressResponse(StatusResponse):
    """Progress response for pull/push/create."""

    completed: Optional[int] = None
    total: Optional[int] = None
    digest: Optional[str] = None


class ProcessResponse(SubscriptableBaseModel):
    """Response from /api/ps (running models)."""

    class Model(SubscriptableBaseModel):
        """Running model details."""

        model: Optional[str] = None
        name: Optional[str] = None
        digest: Optional[str] = None
        expires_at: Optional[datetime] = None
        size: Optional[ByteSize] = None
        size_vram: Optional[ByteSize] = None
        details: Optional[ModelDetails] = None
        context_length: Optional[int] = None

    models: Sequence[Model] = Field(default_factory=list)


class VersionResponse(SubscriptableBaseModel):
    """Response from /api/version."""

    version: str = ""


class WebSearchResult(SubscriptableBaseModel):
    """Single web search result."""

    content: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None


class WebSearchResponse(SubscriptableBaseModel):
    """Response from web search."""

    results: Sequence[WebSearchResult] = Field(default_factory=list)


class WebFetchResponse(SubscriptableBaseModel):
    """Response from web fetch."""

    title: Optional[str] = None
    content: Optional[str] = None
    links: Optional[Sequence[str]] = None


# =============================================================================
# Errors
# =============================================================================


class RequestError(Exception):
    """Error when making a request."""

    def __init__(self, error: str):
        super().__init__(error)
        self.error = error


class ResponseError(Exception):
    """Error in API response."""

    def __init__(self, error: str, status_code: int = -1):
        import json as json_module

        try:
            parsed = json_module.loads(error)
            error = parsed.get("error", error)
        except (json_module.JSONDecodeError, TypeError):
            pass

        super().__init__(error)
        self.error = error
        self.status_code = status_code

    def __str__(self) -> str:
        return f"{self.error} (status code: {self.status_code})"
