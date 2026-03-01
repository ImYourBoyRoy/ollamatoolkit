# ./ollamatoolkit/tools/system_health.py
"""
Ollama Toolkit - System Health Tool
===================================
Allows agents to check the health and capabilities of the underlying Ollama system.
"""

from typing import Dict, Any
from ..connector import OllamaConnector
from ..config import ToolkitConfig


class SystemHealthTool:
    def __init__(self, config: ToolkitConfig):
        self.config = config
        self.base_url = config.agent.base_url

    def get_status(self) -> Dict[str, Any]:
        """
        Returns the current status of the Ollama system.
        """
        caps = OllamaConnector.check_capabilities(self.base_url)
        available_models = OllamaConnector.get_available_models(self.base_url)

        return {
            "status": "online" if caps.get("online") else "offline",
            "capabilities": caps,
            "config": {
                "base_url": self.base_url,
                "current_model": self.config.agent.model,
                "strict_mode": self.config.agent.strict_mode,
            },
            "available_models": available_models,
        }

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Returns detailed information about a specific model.
        """
        connector = OllamaConnector(base_url=self.base_url)
        try:
            return connector.get_model_details(model_name)
        finally:
            connector.close()
