# ./tests/test_connector.py
"""
Unit tests for OllamaConnector health/model helper behavior.
Run: `python -m pytest tests/test_connector.py`.
Inputs: mocked OllamaClient responses for version/tags endpoints.
Outputs: assertions on health payloads and model name extraction.
Side effects: none.
Operational notes: fully offline-safe using mocks.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ollamatoolkit.connector import OllamaConnector
from ollamatoolkit.types import ListResponse, VersionResponse


def test_check_health_online_payload_contains_version_and_model_count() -> None:
    with patch("ollamatoolkit.connector.OllamaClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.version.return_value = VersionResponse(version="0.17.4")
        mock_client.list.return_value = ListResponse(
            models=[
                ListResponse.Model(name="qwen3:8b"),
                ListResponse.Model(name="llama3.1:latest"),
            ]
        )
        mock_client_cls.return_value = mock_client

        payload = OllamaConnector.check_health("http://ollama-server.local:11434")

        assert payload["online"] is True
        assert payload["version"] == "0.17.4"
        assert payload["model_count"] == 2


def test_check_health_offline_payload_contains_error() -> None:
    with patch("ollamatoolkit.connector.OllamaClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.version.side_effect = RuntimeError("connection refused")
        mock_client_cls.return_value = mock_client

        payload = OllamaConnector.check_health("http://offline.local:11434")

        assert payload["online"] is False
        assert payload["version"] == "unknown"
        assert "connection refused" in payload["error"]


def test_get_available_models_uses_name_or_model_fallback() -> None:
    with patch("ollamatoolkit.connector.OllamaClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.list.return_value = ListResponse(
            models=[
                ListResponse.Model(name="qwen3:8b"),
                ListResponse.Model(model="llama3.1:latest"),
            ]
        )
        mock_client_cls.return_value = mock_client

        models = OllamaConnector.get_available_models(
            "http://ollama-server.local:11434"
        )
        assert models == ["qwen3:8b", "llama3.1:latest"]


def test_check_ollama_health_returns_boolean() -> None:
    with patch("ollamatoolkit.connector.OllamaConnector.check_health") as check_health:
        check_health.return_value = {"online": True}
        assert OllamaConnector.check_ollama_health("http://ollama.local:11434") is True

        check_health.return_value = {"online": False}
        assert OllamaConnector.check_ollama_health("http://ollama.local:11434") is False
