# ./ollamatoolkit/exceptions.py
"""
OllamaToolkit Exception Hierarchy
=================================
Custom exceptions for better error handling and debugging.

This module provides a structured exception hierarchy that enables:
- Clear error categorization (model, connection, tool, config errors)
- Proper exception chaining for debugging
- Type-safe error handling with isinstance checks

Usage:
    from ollamatoolkit.exceptions import ModelNotFoundError, ToolExecutionError

    try:
        client.generate("missing-model", "Hello")
    except ModelNotFoundError as e:
        print(f"Model {e.model} not available")
"""


class OllamaToolkitError(Exception):
    """Base exception for all toolkit errors."""

    pass


# =============================================================================
# Model Errors
# =============================================================================


class ModelError(OllamaToolkitError):
    """Base class for model-related errors."""

    pass


class ModelNotFoundError(ModelError):
    """Requested model is not available locally."""

    def __init__(self, model: str, message: str = None):
        self.model = model
        super().__init__(
            message or f"Model '{model}' not found. Run: ollama pull {model}"
        )


class ModelLoadError(ModelError):
    """Failed to load model (GPU memory, corruption, etc.)."""

    def __init__(self, model: str, reason: str = None):
        self.model = model
        self.reason = reason
        msg = f"Failed to load model '{model}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class ModelBusyError(ModelError):
    """Model is currently in use and cannot be unloaded."""

    def __init__(self, model: str):
        self.model = model
        super().__init__(f"Model '{model}' is busy and cannot be unloaded")


class CapabilityNotFoundError(ModelError):
    """No model with the required capability is available."""

    # Suggested models for each capability
    CAPABILITY_MODELS = {
        "vision": ["llava", "qwen2.5-vl", "llava-llama3"],
        "embedding": ["nomic-embed-text", "mxbai-embed-large", "all-minilm"],
        "tools": ["llama3.1", "mistral", "qwen2.5"],
        "reasoning": ["qwen3", "llama3.1", "deepseek-coder"],
    }

    def __init__(self, capability: str, available_capabilities: list = None):
        self.capability = capability
        self.available_capabilities = available_capabilities or []

        # Build helpful message
        msg = f"No model with '{capability}' capability found.\n"

        if available_capabilities:
            msg += f"Available capabilities: {', '.join(available_capabilities)}\n"

        # Suggest models to install
        suggestions = self.CAPABILITY_MODELS.get(capability, [])
        if suggestions:
            msg += f"\nInstall a {capability} model:\n"
            for model in suggestions[:3]:
                msg += f"  ollama pull {model}\n"

        super().__init__(msg.strip())


# =============================================================================
# Connection Errors
# =============================================================================


class ConnectionError(OllamaToolkitError):
    """Failed to connect to Ollama server."""

    def __init__(self, url: str = None, message: str = None):
        self.url = url
        default_msg = "Failed to connect to Ollama server"
        if url:
            default_msg += f" at {url}"
        default_msg += ". Ensure Ollama is running: https://ollama.com/download"
        super().__init__(message or default_msg)


class TimeoutError(OllamaToolkitError):
    """Request timed out."""

    def __init__(self, operation: str = "request", timeout: float = None):
        self.operation = operation
        self.timeout = timeout
        msg = f"Operation '{operation}' timed out"
        if timeout:
            msg += f" after {timeout}s"
        super().__init__(msg)


# =============================================================================
# Tool Errors
# =============================================================================


class ToolError(OllamaToolkitError):
    """Base class for tool-related errors."""

    pass


class ToolNotFoundError(ToolError):
    """Requested tool is not registered."""

    def __init__(self, tool_name: str, available: list = None):
        self.tool_name = tool_name
        self.available = available or []
        msg = f"Tool '{tool_name}' not found"
        if available:
            msg += f". Available tools: {', '.join(available[:5])}"
        super().__init__(msg)


class ToolExecutionError(ToolError):
    """Tool failed during execution."""

    def __init__(self, tool_name: str, cause: Exception = None, message: str = None):
        self.tool_name = tool_name
        self.cause = cause
        msg = message or f"Tool '{tool_name}' execution failed"
        if cause:
            msg += f": {cause}"
        super().__init__(msg)
        if cause:
            self.__cause__ = cause


class ToolValidationError(ToolError):
    """Tool arguments failed validation."""

    def __init__(self, tool_name: str, errors: list = None):
        self.tool_name = tool_name
        self.errors = errors or []
        msg = f"Invalid arguments for tool '{tool_name}'"
        if errors:
            msg += f": {'; '.join(errors)}"
        super().__init__(msg)


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(OllamaToolkitError):
    """Invalid or missing configuration."""

    def __init__(self, field: str = None, message: str = None):
        self.field = field
        msg = message or "Configuration error"
        if field:
            msg = f"Invalid configuration for '{field}': {message}"
        super().__init__(msg)


# =============================================================================
# Vector Store / RAG Errors
# =============================================================================


class VectorStoreError(OllamaToolkitError):
    """Base class for vector store errors."""

    pass


class EmbeddingError(VectorStoreError):
    """Failed to generate embeddings."""

    def __init__(self, model: str = None, message: str = None):
        self.model = model
        msg = message or "Embedding generation failed"
        if model:
            msg += f" (model: {model})"
        super().__init__(msg)


class IndexError(VectorStoreError):
    """Vector index operation failed."""

    pass


# =============================================================================
# MCP Errors
# =============================================================================


class MCPError(OllamaToolkitError):
    """Base class for MCP (Model Context Protocol) errors."""

    pass


class MCPConnectionError(MCPError):
    """Failed to connect to MCP server."""

    def __init__(self, server_name: str, message: str = None):
        self.server_name = server_name
        super().__init__(message or f"Failed to connect to MCP server '{server_name}'")


class MCPProtocolError(MCPError):
    """MCP protocol violation or unexpected response."""

    def __init__(self, message: str, request_id: int = None):
        self.request_id = request_id
        super().__init__(message)


# =============================================================================
# Agent Errors
# =============================================================================


class AgentError(OllamaToolkitError):
    """Base class for agent-related errors."""

    pass


class MaxTurnsExceededError(AgentError):
    """Agent exceeded maximum conversation turns."""

    def __init__(self, max_turns: int, task: str = None):
        self.max_turns = max_turns
        self.task = task
        msg = f"Agent exceeded maximum turns ({max_turns})"
        if task:
            msg += f" while processing: {task[:100]}"
        super().__init__(msg)


class StructuredOutputError(AgentError):
    """Failed to parse structured output from agent."""

    def __init__(self, expected_type: str = None, raw_output: str = None):
        self.expected_type = expected_type
        self.raw_output = raw_output
        msg = "Failed to parse structured output"
        if expected_type:
            msg += f" as {expected_type}"
        super().__init__(msg)
