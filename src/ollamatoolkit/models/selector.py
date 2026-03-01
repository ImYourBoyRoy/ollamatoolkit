# ./src/ollamatoolkit/models/selector.py
"""
Ollama Toolkit - Smart Model Selector
=====================================
Automatically selects the best available model for a given task.

Uses the Ollama API to query installed models and their capabilities,
then picks the most appropriate one based on task requirements.

Usage:
    from ollamatoolkit.models import ModelSelector

    selector = ModelSelector()

    # Get best model for vision tasks
    vision_model = selector.get_for_capability("vision")

    # Get model with multiple capabilities
    agent_model = selector.get_for_capabilities("tools", "completion")

    # Get smallest model for fast responses
    fast_model = selector.get_for_capability("completion", prefer_small=True)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import ollama

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Common task types for model selection."""

    CHAT = "chat"
    VISION = "vision"
    EMBEDDING = "embedding"
    REASONING = "reasoning"
    CODE = "code"
    OCR = "ocr"


@dataclass
class ModelInfo:
    """Information about a single model."""

    name: str
    family: str
    parameter_size: str
    parameter_count: int
    capabilities: List[str]
    quantization: str
    context_length: int

    @property
    def size_in_billions(self) -> float:
        """Parse parameter size to float (e.g., '8.0B' -> 8.0, '595.78M' -> 0.596)."""
        size = self.parameter_size
        try:
            if "B" in size:
                return float(size.replace("B", ""))
            elif "M" in size:
                return float(size.replace("M", "")) / 1000
            return 0.0
        except (ValueError, TypeError):
            return 0.0


class ModelSelector:
    """
    Smart model selector that auto-picks the best model for each task.

    Queries the Ollama server for available models and their capabilities,
    then provides recommendations based on task requirements.
    """

    # Known model family preferences for certain tasks
    FAMILY_PREFERENCES = {
        TaskType.CODE: ["qwen", "codellama", "deepseek", "starcoder"],
        TaskType.VISION: ["qwen-vl", "llava", "bakllava", "mistral3"],
        TaskType.REASONING: ["qwen3", "gemma", "phi"],
        TaskType.OCR: ["deepseek-ocr", "qwen-vl"],
    }

    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize ModelSelector.

        Args:
            base_url: Ollama server URL
        """
        self.base_url = base_url
        self.client = ollama.Client(host=base_url)
        self._models: Dict[str, ModelInfo] = {}
        self._last_refresh = None
        self.refresh()

    def refresh(self) -> int:
        """
        Refresh the list of available models from the server.

        Returns:
            Number of models found
        """
        self._models.clear()

        try:
            # Get list of models
            response = self.client.list()
            models = response.models if hasattr(response, "models") else []

            for model in models:
                # Ollama API uses .model attribute, not .name
                name = getattr(model, "model", None) or getattr(model, "name", None)
                if not name:
                    name = str(model)

                # Get detailed info
                try:
                    show_response = self.client.show(name)
                    details: Any = (
                        show_response.details
                        if hasattr(show_response, "details")
                        else {}
                    )

                    # Handle modelinfo - could be dict or object
                    modelinfo_raw = (
                        show_response.modelinfo
                        if hasattr(show_response, "modelinfo")
                        else {}
                    )

                    # Convert to dict if it's an object
                    if isinstance(modelinfo_raw, dict):
                        modelinfo = dict(modelinfo_raw)
                    elif modelinfo_raw is not None and hasattr(modelinfo_raw, "items"):
                        modelinfo = dict(modelinfo_raw.items())
                    elif hasattr(modelinfo_raw, "__dict__"):
                        modelinfo = modelinfo_raw.__dict__
                    else:
                        modelinfo = {}

                    # Extract capabilities
                    capabilities = self._extract_capabilities(show_response)

                    # Get context length from model info
                    context_length = 0
                    for key, value in modelinfo.items():
                        if "context_length" in key:
                            context_length = value
                            break

                    self._models[name] = ModelInfo(
                        name=name,
                        family=getattr(details, "family", "unknown"),
                        parameter_size=getattr(details, "parameter_size", "0B"),
                        parameter_count=modelinfo.get("general.parameter_count", 0)
                        if modelinfo
                        else 0,
                        capabilities=capabilities,
                        quantization=getattr(details, "quantization_level", "unknown"),
                        context_length=context_length,
                    )
                except Exception as e:
                    # Log but don't fail - add with basic info
                    logger.warning(f"Failed to get details for model={name}: {e}")
                    # Still add basic model entry
                    self._models[name] = ModelInfo(
                        name=name,
                        family="unknown",
                        parameter_size="0B",
                        parameter_count=0,
                        capabilities=["completion"],
                        quantization="unknown",
                        context_length=0,
                    )

            logger.info(f"ModelSelector loaded {len(self._models)} models")
            return len(self._models)

        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return 0

    def _extract_capabilities(self, show_response) -> List[str]:
        """Extract capabilities from model show response."""
        capabilities = []

        # Check for capabilities attribute (newer Ollama)
        if hasattr(show_response, "capabilities"):
            caps = show_response.capabilities
            if caps:
                return list(caps) if hasattr(caps, "__iter__") else [str(caps)]

        # Fall back to modelinfo inspection
        modelinfo = (
            show_response.modelinfo if hasattr(show_response, "modelinfo") else {}
        )
        if not modelinfo:
            return ["completion"]  # Default

        # Check for vision capability
        for key in modelinfo.keys():
            if "vision" in key.lower():
                capabilities.append("vision")
                break

        # Check template for tool support
        template = show_response.template if hasattr(show_response, "template") else ""
        if template and ("Tools" in template or "tool_call" in template.lower()):
            capabilities.append("tools")

        # Check for embedding capability
        arch = modelinfo.get("general.architecture", "")
        if "bert" in arch.lower() or "embed" in arch.lower():
            capabilities.append("embedding")

        # Default to completion
        if not capabilities:
            capabilities.append("completion")

        return capabilities

    @property
    def model_names(self) -> List[str]:
        """Get all available model names."""
        return list(self._models.keys())

    @property
    def total_models(self) -> int:
        """Get total number of available models."""
        return len(self._models)

    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Get model info by name."""
        return self._models.get(name)

    def get_models_by_capability(self, capability: str) -> List[ModelInfo]:
        """Get all models with a specific capability."""
        return [m for m in self._models.values() if capability in m.capabilities]

    def get_for_capability(
        self, capability: str, prefer_small: bool = True
    ) -> Optional[str]:
        """
        Get the best model for a capability.

        Args:
            capability: Required capability (vision, tools, embedding, reasoning)
            prefer_small: If True, prefer smaller models for speed

        Returns:
            Model name or None
        """
        candidates = self.get_models_by_capability(capability)
        if not candidates:
            return None

        # Sort by size
        if prefer_small:
            candidates.sort(key=lambda m: m.size_in_billions)
        else:
            candidates.sort(key=lambda m: m.size_in_billions, reverse=True)

        return candidates[0].name

    def get_for_capabilities(
        self, *capabilities: str, prefer_small: bool = True
    ) -> Optional[str]:
        """
        Get a model with ALL specified capabilities.

        Args:
            *capabilities: Required capabilities
            prefer_small: If True, prefer smaller models

        Returns:
            Model name or None
        """
        required = set(capabilities)
        candidates = [
            m for m in self._models.values() if required.issubset(set(m.capabilities))
        ]

        if not candidates:
            return None

        if prefer_small:
            candidates.sort(key=lambda m: m.size_in_billions)
        else:
            candidates.sort(key=lambda m: m.size_in_billions, reverse=True)

        return candidates[0].name

    def get_for_task(self, task: TaskType, prefer_small: bool = True) -> Optional[str]:
        """
        Get the best model for a specific task type.

        Args:
            task: Task type enum
            prefer_small: If True, prefer smaller models

        Returns:
            Model name or None
        """
        capability_map = {
            TaskType.CHAT: "completion",
            TaskType.VISION: "vision",
            TaskType.EMBEDDING: "embedding",
            TaskType.REASONING: "reasoning",
            TaskType.CODE: "tools",  # Code usually benefits from tool capability
            TaskType.OCR: "vision",
        }

        # Get base candidates
        required_cap = capability_map.get(task, "completion")
        candidates = self.get_models_by_capability(required_cap)

        if not candidates:
            return None

        # Apply family preferences if available
        preferred_families = self.FAMILY_PREFERENCES.get(task, [])
        if preferred_families:
            preferred = [
                m
                for m in candidates
                if any(fam in m.family.lower() for fam in preferred_families)
            ]
            if preferred:
                candidates = preferred

        # Sort by size
        if prefer_small:
            candidates.sort(key=lambda m: m.size_in_billions)
        else:
            candidates.sort(key=lambda m: m.size_in_billions, reverse=True)

        return candidates[0].name

    def get_best_chat_model(self) -> Optional[str]:
        """Get the best model for general chat (prefers tools capability)."""
        # Try to get a model with tools
        model = self.get_for_capability("tools", prefer_small=True)
        if model:
            return model

        # Fall back to any completion model
        return self.get_for_capability("completion", prefer_small=True)

    def get_best_embedding_model(self) -> Optional[str]:
        """Get the best embedding model."""
        return self.get_for_capability("embedding", prefer_small=True)

    def get_best_vision_model(self) -> Optional[str]:
        """Get the best vision model."""
        return self.get_for_capability("vision", prefer_small=True)

    def has_capability(self, model_name: str, capability: str) -> bool:
        """Check if a model has a specific capability."""
        model = self._models.get(model_name)
        return model is not None and capability in model.capabilities

    def require_capability(self, capability: str) -> str:
        """
        Get a model with the capability or raise an error.

        Args:
            capability: Required capability

        Returns:
            Model name

        Raises:
            CapabilityNotFoundError: If no model has the capability
        """
        from ollamatoolkit.exceptions import CapabilityNotFoundError

        model = self.get_for_capability(capability)
        if not model:
            available = list(
                {cap for m in self._models.values() for cap in m.capabilities}
            )
            raise CapabilityNotFoundError(capability, available)
        return model

    def summary(self) -> Dict[str, Any]:
        """Get a summary of available models and capabilities."""
        by_capability: Dict[str, List[str]] = {}
        by_family: Dict[str, List[str]] = {}

        for name, model in self._models.items():
            # Group by capability
            for cap in model.capabilities:
                if cap not in by_capability:
                    by_capability[cap] = []
                by_capability[cap].append(name)

            # Group by family
            if model.family not in by_family:
                by_family[model.family] = []
            by_family[model.family].append(name)

        return {
            "total_models": len(self._models),
            "by_capability": by_capability,
            "by_family": by_family,
        }
