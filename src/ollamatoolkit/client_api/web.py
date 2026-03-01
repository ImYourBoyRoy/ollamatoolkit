# ./src/ollamatoolkit/client_api/web.py
"""
Web endpoint adapters for OllamaToolkit clients.
Run: imported by sync/async client façades.
Inputs: web search or web fetch payloads.
Outputs: typed web search/fetch responses.
Side effects: network calls to hosted Ollama web APIs.
Operational notes: requires Authorization header via `OLLAMA_API_KEY` or custom headers.
"""

from __future__ import annotations

from ollamatoolkit.client_api.common import has_authorization_header
from ollamatoolkit.client_api.transport import AsyncTransport, SyncTransport
from ollamatoolkit.types import (
    WebFetchRequest,
    WebFetchResponse,
    WebSearchRequest,
    WebSearchResponse,
)

AUTH_ERROR = (
    "Authorization header with Bearer token is required for web endpoints. "
    "Set OLLAMA_API_KEY environment variable or pass Authorization header explicitly."
)


class SyncWebAPI:
    """Synchronous adapter for hosted Ollama web APIs."""

    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def web_search(self, query: str, max_results: int = 3) -> WebSearchResponse:
        """Call `https://ollama.com/api/web_search`."""
        if not has_authorization_header(self._transport.headers):
            raise ValueError(AUTH_ERROR)

        request = WebSearchRequest(query=query, max_results=max_results)
        return self._transport.request(
            WebSearchResponse,
            "POST",
            "https://ollama.com/api/web_search",
            json=request.model_dump(exclude_none=True),
        )

    def web_fetch(self, url: str) -> WebFetchResponse:
        """Call `https://ollama.com/api/web_fetch`."""
        if not has_authorization_header(self._transport.headers):
            raise ValueError(AUTH_ERROR)

        request = WebFetchRequest(url=url)
        return self._transport.request(
            WebFetchResponse,
            "POST",
            "https://ollama.com/api/web_fetch",
            json=request.model_dump(exclude_none=True),
        )


class AsyncWebAPI:
    """Asynchronous adapter for hosted Ollama web APIs."""

    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def web_search(self, query: str, max_results: int = 3) -> WebSearchResponse:
        """Call `https://ollama.com/api/web_search`."""
        if not has_authorization_header(self._transport.headers):
            raise ValueError(AUTH_ERROR)

        request = WebSearchRequest(query=query, max_results=max_results)
        return await self._transport.request(
            WebSearchResponse,
            "POST",
            "https://ollama.com/api/web_search",
            json=request.model_dump(exclude_none=True),
        )

    async def web_fetch(self, url: str) -> WebFetchResponse:
        """Call `https://ollama.com/api/web_fetch`."""
        if not has_authorization_header(self._transport.headers):
            raise ValueError(AUTH_ERROR)

        request = WebFetchRequest(url=url)
        return await self._transport.request(
            WebFetchResponse,
            "POST",
            "https://ollama.com/api/web_fetch",
            json=request.model_dump(exclude_none=True),
        )
