# ./src/ollamatoolkit/client_api/common.py
"""
Shared helpers for OllamaToolkit API clients.
Run: imported by sync/async client domain modules.
Inputs: host/header/function metadata and message/tool payloads.
Outputs: normalized URLs/headers and serialized message/tool structures.
Side effects: none (pure helper utilities).
Operational notes: helpers keep sync and async client behavior aligned.
"""

from __future__ import annotations

import os
import platform
from importlib import metadata
from inspect import Parameter, Signature, getdoc, signature
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Union,
)

from ollamatoolkit.types import Message, Tool

try:
    PACKAGE_VERSION = metadata.version("ollamatoolkit")
except metadata.PackageNotFoundError:
    PACKAGE_VERSION = "0.2.0"


def parse_host(host: Optional[str]) -> str:
    """Parse and normalize an Ollama host URL."""
    normalized = (host or "http://localhost:11434").strip().rstrip("/")
    if not normalized.startswith(("http://", "https://")):
        normalized = f"http://{normalized}"
    return normalized


def default_headers() -> Dict[str, str]:
    """Return default HTTP headers for Ollama API requests."""
    headers: Dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": (
            f"ollamatoolkit/{PACKAGE_VERSION} "
            f"({platform.machine()} {platform.system().lower()}) "
            f"Python/{platform.python_version()}"
        ),
    }

    api_key = os.getenv("OLLAMA_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    return headers


def has_authorization_header(headers: Mapping[str, str]) -> bool:
    """Check whether a header mapping includes Authorization."""
    return any(key.lower() == "authorization" for key in headers)


def function_to_tool_schema(func: Callable[..., Any]) -> Dict[str, Any]:
    """Convert a Python callable into an Ollama/OpenAI-compatible tool schema."""
    function_signature: Signature = signature(func)

    properties: Dict[str, Dict[str, str]] = {}
    required: list[str] = []

    for name, parameter in function_signature.parameters.items():
        if name == "self":
            continue

        param_type = _python_annotation_to_json_type(parameter.annotation)
        properties[name] = {"type": param_type}

        if parameter.default is Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": getdoc(func) or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def normalize_messages(
    messages: Optional[Sequence[Union[Mapping[str, Any], Message]]],
) -> list[dict[str, Any]]:
    """Normalize typed/raw message payloads for request serialization."""
    output: list[dict[str, Any]] = []
    for message in messages or []:
        if isinstance(message, Message):
            output.append(message.model_dump(exclude_none=True))
        else:
            output.append(dict(message))
    return output


def normalize_tools(
    tools: Optional[Sequence[Union[Mapping[str, Any], Tool, Callable[..., Any]]]],
) -> list[dict[str, Any]]:
    """Normalize typed/raw/callable tools for request serialization."""
    output: list[dict[str, Any]] = []
    for tool in tools or []:
        if isinstance(tool, Tool):
            output.append(tool.model_dump(exclude_none=True))
        elif callable(tool):
            output.append(function_to_tool_schema(tool))
        else:
            output.append(dict(tool))
    return output


def merge_headers(
    base_headers: Mapping[str, str],
    extra_headers: Optional[Mapping[str, str]],
) -> Dict[str, str]:
    """Merge two header mappings into a mutable dictionary."""
    merged: MutableMapping[str, str] = dict(base_headers)
    if extra_headers:
        merged.update(extra_headers)
    return dict(merged)


def _python_annotation_to_json_type(annotation: Any) -> str:
    """Convert a Python type annotation into a JSON schema primitive type name."""
    if annotation is bool:
        return "boolean"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    return "string"
