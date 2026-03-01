# ./ollamatoolkit/models/__init__.py
"""
Ollama Toolkit - Model Utilities
================================
Tools for model inspection, selection, and management.

Main Components:
- ModelSelector: Smart auto-selection based on task requirements
- ModelInspector: Detailed model capability inspection
- ModelBenchmark: Performance benchmarking

Usage:
    from ollamatoolkit.models import ModelSelector, ModelInspector

    selector = ModelSelector()
    vision_model = selector.get_for_capability("vision")
"""

from .selector import ModelSelector, ModelInfo, TaskType

# Re-export from tools for backwards compatibility
from ollamatoolkit.tools.models import ModelInspector
from ollamatoolkit.tools.benchmark import ModelBenchmark

__all__ = [
    "ModelSelector",
    "ModelInfo",
    "TaskType",
    "ModelInspector",
    "ModelBenchmark",
]
