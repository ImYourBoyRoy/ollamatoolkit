# ./src/ollamatoolkit/openai_types.py
"""
Typed models for Ollama OpenAI-compatible (`/v1/*`) endpoints.
Run: imported by OpenAI compatibility client adapters.
Inputs: OpenAI-style request payloads.
Outputs: validated typed request/response objects.
Side effects: none (schema definitions only).
Operational notes: models allow extra fields for forward compatibility with Ollama updates.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Union

from pydantic import ConfigDict, Field

from ollamatoolkit.types import SubscriptableBaseModel, Tool


class OpenAICompatFunctionCall(SubscriptableBaseModel):
    """Function payload for tool calls."""

    name: Optional[str] = None
    arguments: Optional[Union[str, Mapping[str, Any]]] = None


class OpenAICompatToolCall(SubscriptableBaseModel):
    """Tool call structure used by chat completion responses."""

    id: Optional[str] = None
    index: Optional[int] = None
    type: Optional[str] = "function"
    function: OpenAICompatFunctionCall


class OpenAICompatMessage(SubscriptableBaseModel):
    """OpenAI-compatible chat message."""

    model_config = ConfigDict(extra="allow")

    role: Optional[Literal["system", "user", "assistant", "tool", "developer"]] = None
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[OpenAICompatToolCall]] = None
    tool_call_id: Optional[str] = None
    refusal: Optional[str] = None


class OpenAICompatUsage(SubscriptableBaseModel):
    """Token usage metrics for OpenAI-compatible responses."""

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class OpenAICompatChatCompletionChoice(SubscriptableBaseModel):
    """Choice object for chat completion responses."""

    index: int = 0
    message: Optional[OpenAICompatMessage] = None
    delta: Optional[OpenAICompatMessage] = None
    finish_reason: Optional[str] = None


class OpenAICompatChatCompletionsRequest(SubscriptableBaseModel):
    """Request schema for `/v1/chat/completions`."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: Sequence[Union[OpenAICompatMessage, Mapping[str, Any]]]
    stream: Optional[bool] = None
    tools: Optional[Sequence[Union[Tool, Mapping[str, Any]]]] = None
    tool_choice: Optional[Union[str, Mapping[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, Sequence[str]]] = None


class OpenAICompatChatCompletionsResponse(SubscriptableBaseModel):
    """Response schema for `/v1/chat/completions`."""

    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    object: str = "chat.completion"
    created: Optional[int] = None
    model: Optional[str] = None
    choices: List[OpenAICompatChatCompletionChoice] = Field(default_factory=list)
    usage: Optional[OpenAICompatUsage] = None


class OpenAICompatCompletionsRequest(SubscriptableBaseModel):
    """Request schema for `/v1/completions`."""

    model_config = ConfigDict(extra="allow")

    model: str
    prompt: Union[str, Sequence[str]]
    stream: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, Sequence[str]]] = None


class OpenAICompatCompletionChoice(SubscriptableBaseModel):
    """Choice object for completion responses."""

    index: int = 0
    text: str = ""
    finish_reason: Optional[str] = None


class OpenAICompatCompletionsResponse(SubscriptableBaseModel):
    """Response schema for `/v1/completions`."""

    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    object: str = "text_completion"
    created: Optional[int] = None
    model: Optional[str] = None
    choices: List[OpenAICompatCompletionChoice] = Field(default_factory=list)
    usage: Optional[OpenAICompatUsage] = None


class OpenAICompatEmbeddingsRequest(SubscriptableBaseModel):
    """Request schema for `/v1/embeddings`."""

    model_config = ConfigDict(extra="allow")

    model: str
    input: Union[str, Sequence[str]]
    dimensions: Optional[int] = None
    encoding_format: Optional[str] = None


class OpenAICompatEmbeddingData(SubscriptableBaseModel):
    """Single embedding datum in `/v1/embeddings` response."""

    object: str = "embedding"
    embedding: Sequence[float] = Field(default_factory=list)
    index: int = 0


class OpenAICompatEmbeddingsResponse(SubscriptableBaseModel):
    """Response schema for `/v1/embeddings`."""

    model_config = ConfigDict(extra="allow")

    object: str = "list"
    model: Optional[str] = None
    data: List[OpenAICompatEmbeddingData] = Field(default_factory=list)
    usage: Optional[OpenAICompatUsage] = None


class OpenAICompatModel(SubscriptableBaseModel):
    """Single model entry in `/v1/models` response."""

    model_config = ConfigDict(extra="allow")

    id: str
    object: Optional[str] = "model"
    created: Optional[int] = None
    owned_by: Optional[str] = None


class OpenAICompatModelsResponse(SubscriptableBaseModel):
    """Response schema for `/v1/models`."""

    model_config = ConfigDict(extra="allow")

    object: str = "list"
    data: List[OpenAICompatModel] = Field(default_factory=list)
