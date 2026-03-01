# ./ollamatoolkit/config/presets.py
"""
Ollama Toolkit - Model Presets
==============================
Dynamic model defaults derived from model metadata.

Usage:
    presets = ModelPresets(base_url="http://localhost:11434")

    # Get optimal settings for a task
    settings = presets.get_settings("ministral-3:latest", TaskType.VISION)

    # Get recommended model for task
    model = presets.recommend_model(TaskType.EMBEDDING)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import ollama

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks for model selection."""

    COMPLETION = "completion"
    VISION = "vision"
    EMBEDDING = "embedding"
    REASONING = "reasoning"
    TOOLS = "tools"
    CHAT = "chat"


@dataclass
class ModelSettings:
    """Optimal settings for a model/task combination."""

    model: str
    temperature: float
    max_tokens: int
    context_length: int
    capabilities: List[str]
    family: str
    parameter_size: str
    quantization: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "context_length": self.context_length,
            "capabilities": self.capabilities,
            "family": self.family,
            "parameter_size": self.parameter_size,
            "quantization": self.quantization,
        }


class ModelPresets:
    """
    Dynamic model presets derived from model metadata.
    Provides optimal temperature, max_tokens, and context settings.
    """

    # Temperature recommendations by task type
    TEMPERATURE_DEFAULTS = {
        TaskType.COMPLETION: 0.0,
        TaskType.VISION: 0.2,
        TaskType.EMBEDDING: 0.0,
        TaskType.REASONING: 0.7,
        TaskType.TOOLS: 0.0,
        TaskType.CHAT: 0.7,
    }

    # Max tokens recommendations by task type
    MAX_TOKENS_DEFAULTS = {
        TaskType.COMPLETION: 4096,
        TaskType.VISION: 1024,
        TaskType.EMBEDDING: 512,
        TaskType.REASONING: 8192,
        TaskType.TOOLS: 4096,
        TaskType.CHAT: 2048,
    }

    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize ModelPresets.

        Args:
            base_url: Ollama server URL
        """
        self.base_url = base_url
        self.client = ollama.Client(host=base_url)
        self._cache: Dict[str, Dict] = {}
        self._capabilities_cache: Dict[str, List[str]] = {}

    def get_model_info(self, model_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get model metadata from Ollama.

        Args:
            model_name: Model name
            use_cache: Whether to use cached data

        Returns:
            Model metadata dictionary
        """
        if use_cache and model_name in self._cache:
            return self._cache[model_name]

        try:
            response = self.client.show(model_name)

            if hasattr(response, "model_dump"):
                data = response.model_dump()
            elif hasattr(response, "__dict__"):
                data = dict(response.__dict__)
            else:
                data = dict(response)

            data["model"] = model_name
            data["capabilities"] = self._detect_capabilities(data)

            self._cache[model_name] = data
            return data

        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            return {"model": model_name, "error": str(e)}

    def _detect_capabilities(self, model_data: Dict) -> List[str]:
        """Detect model capabilities from metadata."""
        caps = set(["completion"])

        details = model_data.get("details", {}) or {}
        family = (details.get("family", "") or "").lower()
        families = details.get("families", []) or []
        modelinfo = model_data.get("modelinfo", {}) or {}
        template = model_data.get("template", "") or ""

        # Vision detection
        vision_indicators = [
            "vision" in family,
            "llava" in family,
            any("vision" in f.lower() for f in families if f),
            ".vision" in str(modelinfo).lower(),
            "image" in template.lower(),
        ]
        if any(vision_indicators):
            caps.add("vision")

        # Embedding detection
        embed_indicators = [
            "embed" in family,
            "nomic" in family,
            "bert" in family,
            any("embed" in f.lower() for f in families if f),
        ]
        if any(embed_indicators):
            caps.add("embedding")

        # Tools detection
        if ".tools" in template.lower() or "tool_calls" in template.lower():
            caps.add("tools")

        # Reasoning detection
        if "thinking" in template.lower() or "<think>" in template.lower():
            caps.add("reasoning")

        return sorted(list(caps))

    def get_context_length(self, model_name: str) -> int:
        """Get model's context length from metadata."""
        info = self.get_model_info(model_name)
        modelinfo = info.get("modelinfo", {}) or {}

        # Try common context length keys
        for key in modelinfo:
            if "context_length" in key.lower():
                try:
                    return int(modelinfo[key])
                except (ValueError, TypeError):
                    pass

        # Default based on family
        details = info.get("details", {}) or {}
        family = (details.get("family", "") or "").lower()

        if "bert" in family:
            return 512
        elif "mistral" in family:
            return 32768
        elif "llama" in family:
            return 8192

        return 4096  # Safe default

    def get_settings(
        self, model_name: str, task_type: TaskType = TaskType.COMPLETION
    ) -> ModelSettings:
        """
        Get optimal settings for a model/task combination.

        Args:
            model_name: Model name
            task_type: Type of task

        Returns:
            ModelSettings with optimal configuration
        """
        info = self.get_model_info(model_name)
        details = info.get("details", {}) or {}

        # Get context length
        context_length = self.get_context_length(model_name)

        # Get recommended temperature
        temp = self.TEMPERATURE_DEFAULTS.get(task_type, 0.0)

        # Get recommended max_tokens (capped at context length)
        max_tokens = min(
            self.MAX_TOKENS_DEFAULTS.get(task_type, 4096),
            context_length // 2,  # Leave room for prompt
        )

        return ModelSettings(
            model=model_name,
            temperature=temp,
            max_tokens=max_tokens,
            context_length=context_length,
            capabilities=info.get("capabilities", ["completion"]),
            family=details.get("family", "unknown"),
            parameter_size=details.get("parameter_size", "unknown"),
            quantization=details.get("quantization_level", "unknown"),
        )

    def has_capability(self, model_name: str, capability: str) -> bool:
        """Check if model has a specific capability."""
        info = self.get_model_info(model_name)
        caps = info.get("capabilities", [])
        return capability in caps

    def list_models_with_capability(self, capability: str) -> List[str]:
        """List all models with a specific capability."""
        try:
            response = self.client.list()
            models = [m.model for m in response.models if m.model]

            result = []
            for model in models:
                if self.has_capability(model, capability):
                    result.append(model)

            return result

        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def recommend_model(
        self, task_type: TaskType, prefer_smaller: bool = False
    ) -> Optional[str]:
        """
        Recommend best model for a task type.

        Args:
            task_type: Type of task
            prefer_smaller: Prefer smaller/faster models

        Returns:
            Recommended model name or None
        """
        capability = task_type.value
        models = self.list_models_with_capability(capability)

        if not models:
            # Fall back to completion models
            models = self.list_models_with_capability("completion")

        if not models:
            return None

        # Sort by parameter size
        def get_size(model_name):
            info = self.get_model_info(model_name)
            details = info.get("details", {}) or {}
            size_str = details.get("parameter_size", "0")
            try:
                num = float(size_str.replace("B", "").replace("M", ""))
                if "B" in size_str:
                    num *= 1000
                return num
            except (ValueError, AttributeError):
                return 0

        sorted_models = sorted(models, key=get_size, reverse=not prefer_smaller)
        return sorted_models[0] if sorted_models else None

    def get_all_presets(self) -> Dict[str, ModelSettings]:
        """Get settings for all installed models."""
        try:
            response = self.client.list()
            models = [m.model for m in response.models if m.model]

            return {model: self.get_settings(model) for model in models}
        except Exception as e:
            logger.error(f"Failed to get presets: {e}")
            return {}
