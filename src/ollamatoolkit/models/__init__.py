# ./src/ollamatoolkit/models/__init__.py
"""
Model inspection, selection, and benchmarking exports for OllamaToolkit.
Run via imports from CLI/operator tooling; no standalone CLI entrypoint.
Inputs: package imports requesting selectors, inventory helpers, or benchmark utilities.
Outputs: stable model utility symbols for installed/running inventory and smart selection.
Side effects: may import optional benchmark/model helper modules but avoids runtime CLI behavior changes.
Operational notes: keep exports additive and lightweight for inventory-oriented workflows.
"""

from .inventory import collect_model_inventory
from .selector import ModelSelector, ModelInfo, TaskType

# Re-export from tools for backwards compatibility
from ollamatoolkit.tools.models import ModelInspector
from ollamatoolkit.tools.benchmark import ModelBenchmark

__all__ = [
    "ModelSelector",
    "ModelInfo",
    "TaskType",
    "collect_model_inventory",
    "ModelInspector",
    "ModelBenchmark",
]
