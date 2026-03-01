# ./src/ollamatoolkit/tools/server.py
"""
Ollama Toolkit - Server Tools
=============================
Tools for managing and inspecting the remote Ollama server.
"""

import logging
import json
from typing import Any, Dict, Union

import requests

logger = logging.getLogger(__name__)


class OllamaServerTools:
    """
    Interacts with the Ollama API directly.
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    def _get(self, endpoint: str) -> Union[Dict[str, Any], str]:
        try:
            resp = requests.get(f"{self.base_url}{endpoint}", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return f"Error GET {endpoint}: {e}"

    def _post(
        self, endpoint: str, payload: Dict[str, Any], stream: bool = False
    ) -> Union[Dict[str, Any], str]:
        try:
            resp = requests.post(
                f"{self.base_url}{endpoint}",
                json=payload,
                stream=stream,
                timeout=30,  # longer timeout for ops
            )
            resp.raise_for_status()

            if stream:
                # If streaming, we just return a success message or iterate?
                # For an agent tool, returning a generator is tricky.
                # We'll just read it all for now or check status.
                # Pulling can be huge.
                pass

            return resp.json()
        except Exception as e:
            return f"Error POST {endpoint}: {e}"

    def server_status(self) -> str:
        """Checks if the server is reachable."""
        try:
            requests.get(self.base_url, timeout=2)  # Root usually returns 200
            return "Online"
        except Exception as e:
            return f"Offline or Unreachable: {e}"

    def list_models(self) -> str:
        """Lists available models on the server."""
        # API: /api/tags
        res = self._get("/api/tags")
        if isinstance(res, str):
            return res

        models = res.get("models", [])
        if not models:
            return "No models found."

        summary = []
        for m in models:
            name = m.get("name", "unknown")
            size = m.get("size", 0) / (1024**3)  # GB
            summary.append(f"- {name} ({size:.2f} GB)")
        return "\n".join(summary)

    def show_model_info(self, model_name: str) -> str:
        """Shows details for a specific model."""
        # API: /api/show
        res = self._post("/api/show", {"name": model_name})
        if isinstance(res, str):
            return res

        # Parse key info
        details = res.get("details", {})
        info = [
            f"Model: {model_name}",
            f"Family: {details.get('family', 'N/A')}",
            f"Parameter Size: {details.get('parameter_size', 'N/A')}",
            f"Quantization: {details.get('quantization_level', 'N/A')}",
            f"Modelfile:\n{res.get('modelfile', '')[:200]}...",  # Truncate
        ]
        return "\n".join(info)

    def pull_model(self, model_name: str) -> str:
        """
        Triggers a model pull.
        WARNING: This is a long-running operation.
        """
        # API: /api/pull
        # Note: This usually streams. We will fire and forget? No, user wants feedback.
        # We can start it and return "Pull started".
        try:
            # We use stream=True but don't iterate to avoid blocking forever if it's huge?
            # Or we iterate a bit to see if it works.
            with requests.post(
                f"{self.base_url}/api/pull", json={"name": model_name}, stream=True
            ) as r:
                r.raise_for_status()
                # Read first chunk to confirm start
                for line in r.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "error" in data:
                            return f"Error pulling model: {data['error']}"
                        status = data.get("status", "")
                        return f"Pull started successfully. Initial status: {status}"
                return "Pull request sent; waiting for progress updates."
        except Exception as e:
            return f"Failed to initiate pull: {e}"
