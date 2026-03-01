# ./src/ollamatoolkit/connector.py
"""
High-level Ollama connector façade used by toolkit consumers and dependent apps.
Run: imported via `from ollamatoolkit import OllamaConnector`.
Inputs: Ollama host URL and model/task arguments.
Outputs: health/model metadata and model-management API responses.
Side effects: network requests to remote/local Ollama server.
Operational notes: health checks use `/api/version` first, then `/api/tags` for model stats.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Union

from ollamatoolkit.client import OllamaClient
from ollamatoolkit.types import ProcessResponse, ProgressResponse, StatusResponse

logger = logging.getLogger(__name__)


class OllamaConnector:
    """
    High-level connector for common Ollama API tasks.
    Acts as a stable compatibility layer over `OllamaClient`.
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.client = OllamaClient(host=base_url)

    @staticmethod
    def _health_payload(
        *,
        base_url: str,
        online: bool,
        version: str,
        model_count: int,
        error: str = "",
    ) -> Dict[str, Any]:
        return {
            "host": base_url,
            "online": online,
            "version": version,
            "model_count": model_count,
            "modern_api": online,
            "error": error,
        }

    @staticmethod
    def check_health(base_url: str) -> Dict[str, Any]:
        """
        Check connectivity and server metadata.

        Returns:
            {
              "host": "...",
              "online": bool,
              "version": "x.y.z" | "unknown",
              "model_count": int,
              "modern_api": bool,
              "error": str
            }
        """
        client = OllamaClient(host=base_url, timeout=10.0)
        try:
            version_resp = client.version()
            tags_resp = client.list()
            models = list(tags_resp.models or [])
            return OllamaConnector._health_payload(
                base_url=base_url,
                online=True,
                version=version_resp.version or "unknown",
                model_count=len(models),
            )
        except Exception as exc:
            logger.error("Health check failed for %s: %s", base_url, exc)
            return OllamaConnector._health_payload(
                base_url=base_url,
                online=False,
                version="unknown",
                model_count=0,
                error=str(exc),
            )
        finally:
            client.close()

    @staticmethod
    def check_ollama_health(base_url: str) -> bool:
        """
        Compatibility helper used by dependent applications.
        Returns True if server is online.
        """
        return bool(OllamaConnector.check_health(base_url).get("online", False))

    @staticmethod
    def check_capabilities(base_url: str) -> Dict[str, Any]:
        """Backward-compatible alias for `check_health`."""
        return OllamaConnector.check_health(base_url)

    @staticmethod
    def get_available_models(base_url: str) -> List[str]:
        """Get available model names from Ollama `/api/tags`."""
        client = OllamaClient(host=base_url, timeout=15.0)
        try:
            resp = client.list()
            models = []
            for model in resp.models:
                if model.name:
                    models.append(model.name)
                elif model.model:
                    models.append(model.model)
            return models
        except Exception as exc:
            logger.error("Failed to fetch models from %s: %s", base_url, exc)
            return []
        finally:
            client.close()

    def close(self) -> None:
        """Close underlying HTTP client."""
        self.client.close()

    # --- Instance Methods ---

    def get_models(self) -> List[str]:
        """Get model names using existing client session."""
        resp = self.client.list()
        model_names: List[str] = []
        for model in resp.models:
            if model.name:
                model_names.append(model.name)
            elif model.model:
                model_names.append(model.model)
        return model_names

    def get_model_details(self, model: str) -> Dict[str, Any]:
        """Get detailed metadata for a model."""
        return self.client.get_model_details(model)

    def pull_model(
        self, model: str, stream: bool = False
    ) -> Union[ProgressResponse, Iterator[ProgressResponse]]:
        """Pull a model from remote registry."""
        if stream:
            return self.client.pull(model, stream=True)
        return self.client.pull(model, stream=False)

    def push_model(
        self, model: str, stream: bool = False
    ) -> Union[ProgressResponse, Iterator[ProgressResponse]]:
        """Push a model to remote registry."""
        if stream:
            return self.client.push(model, stream=True)
        return self.client.push(model, stream=False)

    def create_model(
        self,
        model: str,
        from_: str,
        system: str | None = None,
        stream: bool = False,
    ) -> Union[ProgressResponse, Iterator[ProgressResponse]]:
        """Create a model from a base model."""
        if stream:
            return self.client.create(
                model=model,
                from_=from_,
                system=system,
                stream=True,
            )
        return self.client.create(
            model=model,
            from_=from_,
            system=system,
            stream=False,
        )

    def delete_model(self, model: str) -> StatusResponse:
        """Delete a model from local Ollama store."""
        return self.client.delete(model)

    def copy_model(self, source: str, destination: str) -> StatusResponse:
        """Copy an existing model to a new tag."""
        return self.client.copy(source, destination)

    def list_running(self) -> ProcessResponse:
        """List currently loaded/running models."""
        return self.client.ps()
