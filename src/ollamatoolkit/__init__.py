# ./src/ollamatoolkit/__init__.py
"""
Ollama Toolkit - A Professional-Grade Agentic Framework
========================================================
Provides agents, tools, and a full-featured Ollama API client.

Quick Start:
    from ollamatoolkit import OllamaClient, ToolkitConfig
    from ollamatoolkit.tools import ModelInspector, ModelBenchmark

    client = OllamaClient()
    response = client.generate("llama2", "Hello!")
"""

__version__ = "0.2.3"

# Core Components
from .agents.simple import SimpleAgent, AgentHooks
from .agents.role import RoleAgent
from .agents.team import AgentTeam, AgentRole, TeamStrategy, TeamResult
from .agents.memory import ConversationMemory, MemoryConfig
from .extractor import SimpleExtractor
from .connector import OllamaConnector
from .common.utils import clean_json_response

# Full-Featured Client
from .client import OllamaClient, AsyncOllamaClient

# Configuration (from new config package)
from .config import (
    ToolkitConfig,
    AgentConfig,
    VisionConfig,
    VectorConfig,
    ModelsConfig,
    ModelPresets,
    TaskType,
)

# Key Types
from .types import (
    Message,
    Tool,
    Options,
    GenerateResponse,
    ChatResponse,
    EmbedResponse,
    StreamEvent,
    ResponseError,
    RequestError,
)
from .openai_types import (
    OpenAICompatMessage,
    OpenAICompatToolCall,
    OpenAICompatChatCompletionsResponse,
    OpenAICompatCompletionsResponse,
    OpenAICompatEmbeddingsResponse,
    OpenAICompatModelsResponse,
)

# Exceptions
from .exceptions import (
    OllamaToolkitError,
    ModelError,
    ModelNotFoundError,
    ModelLoadError,
    CapabilityNotFoundError,
    ConnectionError,
    ToolError,
    ToolNotFoundError,
    ToolExecutionError,
    ConfigurationError,
    VectorStoreError,
    MCPError,
    AgentError,
)

# Callbacks
from .callbacks import ProgressCallback, ProgressEvent, create_console_callback

# Utilities
from .utils import retry, retry_async

# Model Selection
from .models.selector import ModelSelector, ModelInfo

# Tool Registry
from .tool_registry import ToolRegistry, ToolResult, ToolDefinition

__all__ = [
    "__version__",
    # Agents
    "SimpleAgent",
    "RoleAgent",
    "AgentTeam",
    "AgentRole",
    "TeamStrategy",
    "TeamResult",
    "AgentHooks",
    "ConversationMemory",
    "MemoryConfig",
    "SimpleExtractor",
    # Client
    "OllamaClient",
    "AsyncOllamaClient",
    "OllamaConnector",
    # Config
    "ToolkitConfig",
    "AgentConfig",
    "VisionConfig",
    "VectorConfig",
    "ModelsConfig",
    "ModelPresets",
    "TaskType",
    # Types
    "Message",
    "Tool",
    "Options",
    "GenerateResponse",
    "ChatResponse",
    "EmbedResponse",
    "StreamEvent",
    "ResponseError",
    "RequestError",
    "OpenAICompatMessage",
    "OpenAICompatToolCall",
    "OpenAICompatChatCompletionsResponse",
    "OpenAICompatCompletionsResponse",
    "OpenAICompatEmbeddingsResponse",
    "OpenAICompatModelsResponse",
    # Exceptions
    "OllamaToolkitError",
    "ModelError",
    "ModelNotFoundError",
    "ModelLoadError",
    "CapabilityNotFoundError",
    "ConnectionError",
    "ToolError",
    "ToolNotFoundError",
    "ToolExecutionError",
    "ConfigurationError",
    "VectorStoreError",
    "MCPError",
    "AgentError",
    # Utils
    "clean_json_response",
    "retry",
    "retry_async",
    # Model Selection
    "ModelSelector",
    "ModelInfo",
    # Callbacks
    "ProgressCallback",
    "ProgressEvent",
    "create_console_callback",
    # Tool Registry
    "ToolRegistry",
    "ToolResult",
    "ToolDefinition",
]
