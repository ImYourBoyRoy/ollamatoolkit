# ./src/ollamatoolkit/client_api/transport.py
"""
HTTP transport layer used by OllamaToolkit client domain modules.
Run: imported by sync/async domain adapters.
Inputs: HTTP method/path, pydantic response class, and request payload.
Outputs: typed pydantic responses or streaming iterators/generators.
Side effects: network calls against Ollama REST/OpenAI-compatible APIs.
Operational notes: centralizes error handling for consistent behavior.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Iterator, Mapping, Optional, Type, TypeVar

import httpx

from ollamatoolkit.types import ResponseError

CONNECTION_ERROR_MESSAGE = (
    "Failed to connect to Ollama. Please check that Ollama is downloaded, "
    "running and accessible. https://ollama.com/download"
)

T = TypeVar("T")


class SyncTransport:
    """Synchronous HTTP transport with typed response helpers."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout: Optional[float],
        headers: Mapping[str, str],
        **kwargs: Any,
    ) -> None:
        self.client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers=headers,
            follow_redirects=True,
            **kwargs,
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self.client.close()

    def request_raw(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Issue a raw HTTP request and raise toolkit-specific errors."""
        try:
            response = self.client.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            raise ResponseError(exc.response.text, exc.response.status_code) from None
        except httpx.ConnectError:
            raise ConnectionError(CONNECTION_ERROR_MESSAGE) from None

    def request(self, cls: Type[T], method: str, url: str, **kwargs: Any) -> T:
        """Issue a JSON request and parse into a typed pydantic response model."""
        payload = self.request_raw(method, url, **kwargs).json()
        if isinstance(payload, dict) and payload.get("error"):
            raise ResponseError(str(payload["error"]))
        return cls(**payload)

    def stream(self, cls: Type[T], method: str, url: str, **kwargs: Any) -> Iterator[T]:
        """Stream JSON line-delimited payloads and parse into typed models."""
        try:
            with self.client.stream(method, url, **kwargs) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    part = json.loads(line)
                    if part.get("error"):
                        raise ResponseError(str(part["error"]))
                    yield cls(**part)
        except httpx.HTTPStatusError as exc:
            raise ResponseError(exc.response.text, exc.response.status_code) from None
        except httpx.ConnectError:
            raise ConnectionError(CONNECTION_ERROR_MESSAGE) from None

    def stream_lines(self, method: str, url: str, **kwargs: Any) -> Iterator[str]:
        """Stream raw line payloads for SSE-compatible endpoints (/v1/*)."""
        try:
            with self.client.stream(method, url, **kwargs) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line is None:
                        continue
                    yield line
        except httpx.HTTPStatusError as exc:
            raise ResponseError(exc.response.text, exc.response.status_code) from None
        except httpx.ConnectError:
            raise ConnectionError(CONNECTION_ERROR_MESSAGE) from None

    @property
    def headers(self) -> Dict[str, str]:
        """Expose normalized request headers for compatibility checks."""
        return dict(self.client.headers)


class AsyncTransport:
    """Asynchronous HTTP transport with typed response helpers."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout: Optional[float],
        headers: Mapping[str, str],
        **kwargs: Any,
    ) -> None:
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            headers=headers,
            follow_redirects=True,
            **kwargs,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self.client.aclose()

    async def request_raw(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Issue a raw HTTP request and raise toolkit-specific errors."""
        try:
            response = await self.client.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            raise ResponseError(exc.response.text, exc.response.status_code) from None
        except httpx.ConnectError:
            raise ConnectionError(CONNECTION_ERROR_MESSAGE) from None

    async def request(self, cls: Type[T], method: str, url: str, **kwargs: Any) -> T:
        """Issue a JSON request and parse into a typed pydantic response model."""
        payload = (await self.request_raw(method, url, **kwargs)).json()
        if isinstance(payload, dict) and payload.get("error"):
            raise ResponseError(str(payload["error"]))
        return cls(**payload)

    async def stream(
        self, cls: Type[T], method: str, url: str, **kwargs: Any
    ) -> AsyncIterator[T]:
        """Stream JSON line-delimited payloads and parse into typed models."""
        try:
            async with self.client.stream(method, url, **kwargs) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    part = json.loads(line)
                    if part.get("error"):
                        raise ResponseError(str(part["error"]))
                    yield cls(**part)
        except httpx.HTTPStatusError as exc:
            raise ResponseError(exc.response.text, exc.response.status_code) from None
        except httpx.ConnectError:
            raise ConnectionError(CONNECTION_ERROR_MESSAGE) from None

    async def stream_lines(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream raw line payloads for SSE-compatible endpoints (/v1/*)."""
        try:
            async with self.client.stream(method, url, **kwargs) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line is None:
                        continue
                    yield line
        except httpx.HTTPStatusError as exc:
            raise ResponseError(exc.response.text, exc.response.status_code) from None
        except httpx.ConnectError:
            raise ConnectionError(CONNECTION_ERROR_MESSAGE) from None

    @property
    def headers(self) -> Dict[str, str]:
        """Expose normalized request headers for compatibility checks."""
        return dict(self.client.headers)
