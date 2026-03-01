# ./src/ollamatoolkit/client.py
"""
Public client module for OllamaToolkit.
Run: `from ollamatoolkit.client import OllamaClient, AsyncOllamaClient`.
Inputs: host/timeout/headers and endpoint-specific request args.
Outputs: typed sync/async responses for Ollama `/api/*` and `/v1/*` endpoints.
Side effects: network requests and optional model-management actions on target Ollama host.
Operational notes: implementation is split into endpoint-domain modules under `client_api/`.
"""

from ollamatoolkit.client_api.common import default_headers as _default_headers
from ollamatoolkit.client_api.common import parse_host as _parse_host
from ollamatoolkit.client_api.sync_client import OllamaClient
from ollamatoolkit.client_api.async_client import AsyncOllamaClient

__all__ = ["OllamaClient", "AsyncOllamaClient", "_parse_host", "_default_headers"]
