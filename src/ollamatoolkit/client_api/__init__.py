# ./src/ollamatoolkit/client_api/__init__.py
"""
Composable client-domain adapters used by OllamaToolkit public client façades.
Run: imported internally by `ollamatoolkit.client`.
Inputs: request payloads routed through transport/domain adapters.
Outputs: typed sync/async responses for Ollama `/api/*` and `/v1/*` endpoints.
Side effects: delegated network activity through shared HTTP transports.
"""

from ollamatoolkit.client_api.common import (
    default_headers,
    function_to_tool_schema,
    merge_headers,
    parse_host,
)
from ollamatoolkit.client_api.inference import AsyncInferenceAPI, SyncInferenceAPI
from ollamatoolkit.client_api.models import AsyncModelAPI, SyncModelAPI
from ollamatoolkit.client_api.openai_compat import (
    AsyncOpenAICompatAPI,
    SyncOpenAICompatAPI,
)
from ollamatoolkit.client_api.transport import AsyncTransport, SyncTransport
from ollamatoolkit.client_api.web import AsyncWebAPI, SyncWebAPI

__all__ = [
    "AsyncInferenceAPI",
    "AsyncModelAPI",
    "AsyncOpenAICompatAPI",
    "AsyncTransport",
    "AsyncWebAPI",
    "SyncInferenceAPI",
    "SyncModelAPI",
    "SyncOpenAICompatAPI",
    "SyncTransport",
    "SyncWebAPI",
    "default_headers",
    "function_to_tool_schema",
    "merge_headers",
    "parse_host",
]
