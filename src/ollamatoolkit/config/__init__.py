# ./ollamatoolkit/config/__init__.py
"""
Ollama Toolkit - Configuration Package
======================================
Centralized configuration with model presets and dynamic defaults.
"""

from .core import (
    AgentConfig,
    ToolConfig,
    WebToolConfig,
    LoggingConfig,
    DashboardConfig,
    VisionConfig,
    VectorConfig,
    ModelsConfig,
    MemoryConfig,
    DocumentConfig,
    BenchmarkConfig,
    TelemetryConfig,
    ToolkitConfig,
)
from .presets import ModelPresets, TaskType

__all__ = [
    "AgentConfig",
    "ToolConfig",
    "WebToolConfig",
    "LoggingConfig",
    "DashboardConfig",
    "VisionConfig",
    "VectorConfig",
    "ModelsConfig",
    "MemoryConfig",
    "DocumentConfig",
    "BenchmarkConfig",
    "TelemetryConfig",
    "ToolkitConfig",
    "ModelPresets",
    "TaskType",
]
