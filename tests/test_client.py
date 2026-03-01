# ./tests/test_client.py
"""
Tests for OllamaClient and AsyncOllamaClient.
Uses pytest fixtures for proper mocking.
"""

import json
from unittest.mock import MagicMock, patch, AsyncMock

import httpx
import pytest

from ollamatoolkit.client import (
    OllamaClient,
    AsyncOllamaClient,
    _parse_host,
    _default_headers,
)
from ollamatoolkit.types import (
    GenerateResponse,
    ChatResponse,
    EmbedResponse,
    ListResponse,
    ShowResponse,
    ProcessResponse,
    StatusResponse,
    ResponseError,
    VersionResponse,
)


# =============================================================================
# Helper for mocking httpx responses
# =============================================================================


def make_mock_response(status_code: int = 200, json_data: dict = None, text: str = ""):
    """Create a properly mocked httpx.Response."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.text = text or json.dumps(json_data or {})
    response.json.return_value = json_data or {}

    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="Error",
            request=MagicMock(),
            response=response,
        )
    else:
        response.raise_for_status.return_value = None

    return response


# =============================================================================
# Host Parsing Tests
# =============================================================================


class TestHostParsing:
    """Test host URL parsing."""

    def test_default_host(self):
        assert _parse_host(None) == "http://localhost:11434"

    def test_strips_trailing_slash(self):
        assert _parse_host("http://localhost:11434/") == "http://localhost:11434"

    def test_adds_scheme(self):
        assert _parse_host("localhost:11434") == "http://localhost:11434"

    def test_preserves_https(self):
        assert _parse_host("https://ollama.example.com") == "https://ollama.example.com"

    def test_custom_port(self):
        assert (
            _parse_host("ollama-server.local:11434")
            == "http://ollama-server.local:11434"
        )


# =============================================================================
# Header Tests
# =============================================================================


class TestDefaultHeaders:
    """Test default headers generation."""

    def test_content_type(self):
        headers = _default_headers()
        assert headers["Content-Type"] == "application/json"

    def test_accept(self):
        headers = _default_headers()
        assert headers["Accept"] == "application/json"

    def test_user_agent_present(self):
        headers = _default_headers()
        assert "User-Agent" in headers
        assert "ollamatoolkit" in headers["User-Agent"]


# =============================================================================
# Client Initialization Tests
# =============================================================================


class TestClientInit:
    """Test client initialization."""

    def test_default_init(self):
        client = OllamaClient()
        assert client._client.base_url == httpx.URL("http://localhost:11434")
        client.close()

    def test_custom_host(self):
        client = OllamaClient(host="http://ollama-server.local:11434")
        assert (
            str(client._client.base_url).rstrip("/")
            == "http://ollama-server.local:11434"
        )
        client.close()

    def test_context_manager(self):
        with OllamaClient() as client:
            assert client._client is not None


# =============================================================================
# Generate Endpoint Tests
# =============================================================================


class TestClientGenerate:
    """Test generate endpoint."""

    def test_generate_basic(self):
        mock_response = make_mock_response(
            200,
            {
                "model": "llama2",
                "response": "Hello!",
                "done": True,
                "total_duration": 1000000,
            },
        )

        with patch.object(httpx.Client, "request", return_value=mock_response):
            client = OllamaClient()
            response = client.generate("llama2", "Hi")

            assert isinstance(response, GenerateResponse)
            assert response.response == "Hello!"
            assert response.done is True
            client.close()

    def test_generate_with_options(self):
        mock_response = make_mock_response(
            200, {"model": "llama2", "response": "Test", "done": True}
        )

        with patch.object(httpx.Client, "request", return_value=mock_response):
            client = OllamaClient()
            response = client.generate(
                "llama2",
                "Test prompt",
                system="You are helpful",
                options={"temperature": 0.5},
            )
            assert response.response == "Test"
            client.close()


# =============================================================================
# Chat Endpoint Tests
# =============================================================================


class TestClientChat:
    """Test chat endpoint."""

    def test_chat_basic(self):
        mock_response = make_mock_response(
            200,
            {
                "model": "llama2",
                "message": {"role": "assistant", "content": "Hi there!"},
                "done": True,
            },
        )

        with patch.object(httpx.Client, "request", return_value=mock_response):
            client = OllamaClient()
            response = client.chat(
                "llama2",
                messages=[{"role": "user", "content": "Hello"}],
            )

            assert isinstance(response, ChatResponse)
            assert response.message.content == "Hi there!"
            client.close()


# =============================================================================
# Embed Endpoint Tests
# =============================================================================


class TestClientEmbed:
    """Test embed endpoint."""

    def test_embed_single(self):
        mock_response = make_mock_response(
            200, {"model": "nomic-embed-text", "embeddings": [[0.1, 0.2, 0.3]]}
        )

        with patch.object(httpx.Client, "request", return_value=mock_response):
            client = OllamaClient()
            response = client.embed("nomic-embed-text", "Hello world")

            assert isinstance(response, EmbedResponse)
            assert len(response.embeddings) == 1
            assert len(response.embeddings[0]) == 3
            client.close()

    def test_embed_batch(self):
        mock_response = make_mock_response(
            200,
            {"model": "nomic-embed-text", "embeddings": [[0.1, 0.2], [0.3, 0.4]]},
        )

        with patch.object(httpx.Client, "request", return_value=mock_response):
            client = OllamaClient()
            response = client.embed("nomic-embed-text", ["Hello", "World"])

            assert len(response.embeddings) == 2
            client.close()


# =============================================================================
# Model Management Tests
# =============================================================================


class TestClientModelManagement:
    """Test model management endpoints."""

    def test_list_models(self):
        mock_response = make_mock_response(
            200,
            {
                "models": [
                    {"model": "llama2", "name": "llama2"},
                    {"model": "mistral", "name": "mistral"},
                ]
            },
        )

        with patch.object(httpx.Client, "request", return_value=mock_response):
            client = OllamaClient()
            response = client.list()

            assert isinstance(response, ListResponse)
            assert len(response.models) == 2
            client.close()

    def test_version(self):
        mock_response = make_mock_response(200, {"version": "0.17.4"})

        with patch.object(httpx.Client, "request", return_value=mock_response):
            client = OllamaClient()
            response = client.version()

            assert isinstance(response, VersionResponse)
            assert response.version == "0.17.4"
            client.close()

    def test_show_model(self):
        mock_response = make_mock_response(
            200,
            {
                "template": "{{ .System }}",
                "parameters": "stop: [STOP]",
                "details": {"family": "llama"},
                "capabilities": ["completion", "vision", "tools"],
            },
        )

        with patch.object(httpx.Client, "request", return_value=mock_response):
            client = OllamaClient()
            response = client.show("llama2")

            assert isinstance(response, ShowResponse)
            assert response.template == "{{ .System }}"
            assert response.capabilities == ["completion", "vision", "tools"]
            client.close()

    def test_ps(self):
        mock_response = make_mock_response(
            200, {"models": [{"model": "llama2", "size": 1000000}]}
        )

        with patch.object(httpx.Client, "request", return_value=mock_response):
            client = OllamaClient()
            response = client.ps()

            assert isinstance(response, ProcessResponse)
            assert len(response.models) == 1
            client.close()

    def test_delete(self):
        mock_response = make_mock_response(200, {})

        with patch.object(httpx.Client, "request", return_value=mock_response):
            client = OllamaClient()
            response = client.delete("test-model")

            assert isinstance(response, StatusResponse)
            assert response.status == "success"
            client.close()

    def test_copy(self):
        mock_response = make_mock_response(200, {})

        with patch.object(httpx.Client, "request", return_value=mock_response):
            client = OllamaClient()
            response = client.copy("llama2", "my-llama2")

            assert isinstance(response, StatusResponse)
            assert response.status == "success"
            client.close()

    def test_push(self):
        mock_response = make_mock_response(
            200, {"status": "success", "completed": 10, "total": 10}
        )

        with patch.object(httpx.Client, "request", return_value=mock_response):
            client = OllamaClient()
            response = client.push("llama2")

            assert response.status == "success"
            client.close()

    def test_create(self):
        mock_response = make_mock_response(
            200, {"status": "success", "completed": 1, "total": 1}
        )

        with patch.object(httpx.Client, "request", return_value=mock_response):
            client = OllamaClient()
            response = client.create("my-llama2", from_="llama2")

            assert response.status == "success"
            client.close()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestClientErrors:
    """Test error handling."""

    def test_response_error(self):
        mock_response = make_mock_response(404, {"error": "model not found"})

        with patch.object(httpx.Client, "request", return_value=mock_response):
            client = OllamaClient()
            with pytest.raises(ResponseError) as exc_info:
                client.generate("nonexistent", "test")

            assert exc_info.value.status_code == 404
            client.close()


# =============================================================================
# Async Client Tests
# =============================================================================


class TestAsyncClient:
    """Test async client basics."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        async with AsyncOllamaClient() as client:
            assert client._client is not None

    @pytest.mark.asyncio
    async def test_async_list(self):
        mock_response = make_mock_response(200, {"models": [{"model": "llama2"}]})

        with patch.object(
            httpx.AsyncClient,
            "request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            async with AsyncOllamaClient() as client:
                response = await client.list()
                assert len(response.models) == 1

    @pytest.mark.asyncio
    async def test_async_version(self):
        mock_response = make_mock_response(200, {"version": "0.17.4"})

        with patch.object(
            httpx.AsyncClient,
            "request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            async with AsyncOllamaClient() as client:
                response = await client.version()
                assert response.version == "0.17.4"

    @pytest.mark.asyncio
    async def test_async_push(self):
        mock_response = make_mock_response(
            200, {"status": "success", "completed": 10, "total": 10}
        )

        with patch.object(
            httpx.AsyncClient,
            "request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            async with AsyncOllamaClient() as client:
                response = await client.push("llama2")
                assert response.status == "success"

    @pytest.mark.asyncio
    async def test_async_create(self):
        mock_response = make_mock_response(
            200, {"status": "success", "completed": 1, "total": 1}
        )

        with patch.object(
            httpx.AsyncClient,
            "request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            async with AsyncOllamaClient() as client:
                response = await client.create("my-llama2", from_="llama2")
                assert response.status == "success"
