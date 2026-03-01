# ./tests/conftest.py
"""
OllamaToolkit Test Configuration
================================
Pytest fixtures and mock infrastructure for testing with real Ollama models.

This module provides:
- OllamaModelRegistry: Real model data from test_output JSONs for capability-based qualification
- FakeOllamaClient: Mock client for unit testing
- Fixtures for common test scenarios
- Model selection helpers

Usage:
    # In your test file
    def test_something(ollama_models, temp_workspace):
        # Get a model that supports vision
        vision_model = ollama_models.get_model_for_capability("vision")
        if vision_model:
            # Run test with real model
            pass
        else:
            pytest.skip("No vision model available")
"""

import json
import pytest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import MagicMock, patch


# =============================================================================
# Real Model Registry (from test_output JSONs)
# =============================================================================

# Path to test_output folder (relative to tests/)
TEST_OUTPUT_DIR = Path(__file__).parent.parent / "test_output"
ALL_MODELS_JSON = TEST_OUTPUT_DIR / "all_models.json"
MODEL_SUMMARY_JSON = TEST_OUTPUT_DIR / "model_summary.json"


class OllamaModelRegistry:
    """
    Registry of available Ollama models loaded from test_output JSONs.

    Provides methods to:
    - Query models by capability (vision, tools, embedding, reasoning)
    - Get recommended models for specific tasks
    - Check if a model is installed and has required capabilities
    """

    def __init__(self, load_from_files: bool = True):
        self._models: Dict[str, Dict[str, Any]] = {}
        self._by_capability: Dict[str, List[str]] = {}
        self._by_family: Dict[str, List[str]] = {}

        if load_from_files:
            self._load_model_data()

    def _load_model_data(self):
        """Load model data from test_output JSONs."""
        # Load full model details
        if ALL_MODELS_JSON.exists():
            with open(ALL_MODELS_JSON, "r", encoding="utf-8") as f:
                self._models = json.load(f)

        # Load summary with capability groupings
        if MODEL_SUMMARY_JSON.exists():
            with open(MODEL_SUMMARY_JSON, "r", encoding="utf-8") as f:
                summary = json.load(f)
                self._by_capability = summary.get("by_capability", {})
                self._by_family = summary.get("by_family", {})

    @property
    def model_names(self) -> List[str]:
        """Get all available model names."""
        return list(self._models.keys())

    @property
    def total_models(self) -> int:
        """Get total number of available models."""
        return len(self._models)

    def get_model(self, name: str) -> Optional[Dict[str, Any]]:
        """Get full model details by name."""
        return self._models.get(name)

    def get_models_by_capability(self, capability: str) -> List[str]:
        """
        Get all models with a specific capability.

        Args:
            capability: One of 'vision', 'tools', 'embedding', 'reasoning', 'completion'

        Returns:
            List of model names with that capability
        """
        return self._by_capability.get(capability, [])

    def get_model_for_capability(
        self, capability: str, prefer_small: bool = True
    ) -> Optional[str]:
        """
        Get a single model for a capability, preferring smaller models for speed.

        Args:
            capability: Required capability
            prefer_small: If True, prefer smaller parameter counts

        Returns:
            Model name or None if no model has that capability
        """
        models = self.get_models_by_capability(capability)
        if not models:
            return None

        if prefer_small:
            # Sort by parameter size (extract from model details)
            def get_param_size(name: str) -> float:
                model = self._models.get(name, {})
                details = model.get("details", {})
                param_str = details.get("parameter_size", "0B")
                # Parse "8.0B", "24.0B", "595.78M" etc.
                try:
                    if "B" in param_str:
                        return float(param_str.replace("B", ""))
                    elif "M" in param_str:
                        return float(param_str.replace("M", "")) / 1000
                    return 0
                except (ValueError, TypeError):
                    return 0

            models = sorted(models, key=get_param_size)

        return models[0] if models else None

    def get_models_by_family(self, family: str) -> List[str]:
        """Get all models from a specific family (e.g., 'qwen3', 'llama', 'mistral3')."""
        return self._by_family.get(family, [])

    def has_capability(self, model_name: str, capability: str) -> bool:
        """Check if a specific model has a capability."""
        model = self._models.get(model_name, {})
        capabilities = model.get("capabilities", [])
        return capability in capabilities

    def get_capabilities(self, model_name: str) -> List[str]:
        """Get all capabilities of a model."""
        model = self._models.get(model_name, {})
        return model.get("capabilities", [])

    def get_model_details(self, model_name: str) -> Dict[str, Any]:
        """Get detailed info for a model (family, size, quantization, etc.)."""
        model = self._models.get(model_name, {})
        return model.get("details", {})

    def require_capability(self, capability: str) -> str:
        """
        Get a model with the required capability or raise pytest.skip.

        Usage in tests:
            model = ollama_models.require_capability("vision")
        """
        model = self.get_model_for_capability(capability)
        if not model:
            pytest.skip(f"No model available with '{capability}' capability")
        return model

    def get_best_chat_model(self) -> Optional[str]:
        """Get the best model for general chat (prefers tools + completion)."""
        # Prefer models with both tools and completion
        tool_models = set(self.get_models_by_capability("tools"))
        completion_models = set(self.get_models_by_capability("completion"))
        candidates = tool_models & completion_models

        if candidates:
            # Prefer medium-sized models
            return self.get_model_for_capability("tools", prefer_small=True)

        return self.get_model_for_capability("completion", prefer_small=True)

    def get_best_embedding_model(self) -> Optional[str]:
        """Get the best embedding model available."""
        return self.get_model_for_capability("embedding", prefer_small=True)

    def get_best_vision_model(self) -> Optional[str]:
        """Get the best vision model available."""
        return self.get_model_for_capability("vision", prefer_small=True)


# =============================================================================
# Mock Response Classes
# =============================================================================


@dataclass
class FakeGenerateResponse:
    """Mock response for generate calls."""

    response: str = "This is a mock response."
    model: str = "mock-model"
    done: bool = True
    context: List[int] = field(default_factory=list)
    total_duration: int = 1000000
    load_duration: int = 100000
    prompt_eval_count: int = 10
    prompt_eval_duration: int = 500000
    eval_count: int = 50
    eval_duration: int = 400000


@dataclass
class FakeChatResponse:
    """Mock response for chat calls."""

    model: str = "mock-model"
    done: bool = True
    message: Dict[str, Any] = field(
        default_factory=lambda: {"role": "assistant", "content": "Mock chat response."}
    )
    total_duration: int = 1000000


@dataclass
class FakeEmbedResponse:
    """Mock response for embed calls."""

    model: str = "mock-embed"
    embeddings: List[List[float]] = field(default_factory=lambda: [[0.1] * 384])


@dataclass
class FakeShowResponse:
    """Mock response for show calls."""

    model: str = "mock-model"
    modelfile: str = "FROM base"
    template: str = "{{ .System }} {{ .Prompt }}"
    details: Dict[str, Any] = field(default_factory=dict)
    modelinfo: Dict[str, Any] = field(default_factory=dict)
    parameters: str = ""


@dataclass
class FakeListResponse:
    """Mock response for list calls."""

    models: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "name": "llama3:latest",
                "model": "llama3:latest",
                "size": 4000000000,
                "digest": "abc123",
            },
            {
                "name": "mistral:latest",
                "model": "mistral:latest",
                "size": 3500000000,
                "digest": "def456",
            },
        ]
    )


# =============================================================================
# Fake Ollama Client
# =============================================================================


class FakeOllamaClient:
    """
    Mock Ollama client for unit testing.

    Provides configurable responses for all Ollama API methods without
    requiring a live server connection.

    Attributes:
        responses: Dict mapping prompts/models to custom responses
        call_history: List of all method calls made to this client
        default_response: Default response text when no custom match
    """

    def __init__(
        self,
        responses: Optional[Dict[str, str]] = None,
        default_response: str = "This is a mock response from FakeOllamaClient.",
    ):
        self.responses = responses or {}
        self.call_history: List[Dict[str, Any]] = []
        self.default_response = default_response
        self._installed_models = [
            "llama3:latest",
            "mistral:latest",
            "nomic-embed-text:latest",
        ]

    def generate(
        self,
        model: str,
        prompt: str = "",
        **kwargs,
    ) -> FakeGenerateResponse:
        """Mock generate call."""
        self.call_history.append(
            {
                "method": "generate",
                "model": model,
                "prompt": prompt,
                "kwargs": kwargs,
            }
        )

        # Check for custom response
        response_text = self.responses.get(prompt)
        if response_text is None:
            response_text = self.responses.get(model)
        if response_text is None:
            response_text = self.default_response

        return FakeGenerateResponse(response=response_text, model=model)

    def chat(
        self,
        model: str,
        messages: Optional[Sequence[Dict[str, Any]]] = None,
        **kwargs,
    ) -> FakeChatResponse:
        """Mock chat call."""
        self.call_history.append(
            {
                "method": "chat",
                "model": model,
                "messages": messages,
                "kwargs": kwargs,
            }
        )

        # Check for custom response based on last message
        response_text = self.default_response
        if messages:
            last_content = messages[-1].get("content", "")
            if last_content in self.responses:
                response_text = self.responses[last_content]

        return FakeChatResponse(
            model=model,
            message={"role": "assistant", "content": response_text},
        )

    def embed(
        self,
        model: str,
        input: str,
        **kwargs,
    ) -> FakeEmbedResponse:
        """Mock embed call."""
        self.call_history.append(
            {
                "method": "embed",
                "model": model,
                "input": input,
                "kwargs": kwargs,
            }
        )

        # Generate deterministic embeddings based on input
        import hashlib

        hash_val = int(hashlib.md5(input.encode()).hexdigest()[:8], 16)
        base = (hash_val % 100) / 100.0
        embeddings = [[base + (i * 0.001) for i in range(384)]]

        return FakeEmbedResponse(model=model, embeddings=embeddings)

    def list(self) -> FakeListResponse:
        """Mock list call."""
        self.call_history.append({"method": "list"})
        return FakeListResponse(
            models=[
                {"name": m, "model": m, "size": 1000000000}
                for m in self._installed_models
            ]
        )

    def show(self, model: str) -> FakeShowResponse:
        """Mock show call."""
        self.call_history.append({"method": "show", "model": model})

        if model not in self._installed_models:
            from ollamatoolkit.types import ResponseError

            raise ResponseError(f"model '{model}' not found")

        return FakeShowResponse(model=model)

    def pull(self, model: str, **kwargs) -> MagicMock:
        """Mock pull call."""
        self.call_history.append({"method": "pull", "model": model, "kwargs": kwargs})
        self._installed_models.append(model)
        return MagicMock(status="success")

    def delete(self, model: str) -> MagicMock:
        """Mock delete call."""
        self.call_history.append({"method": "delete", "model": model})
        if model in self._installed_models:
            self._installed_models.remove(model)
        return MagicMock(status="success")

    def copy(self, source: str, destination: str) -> MagicMock:
        """Mock copy call."""
        self.call_history.append(
            {
                "method": "copy",
                "source": source,
                "destination": destination,
            }
        )
        if destination not in self._installed_models:
            self._installed_models.append(destination)
        return MagicMock(status="success")

    def close(self):
        """Mock close."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def get_call_count(self, method: Optional[str] = None) -> int:
        """Get number of calls made, optionally filtered by method."""
        if method:
            return sum(1 for c in self.call_history if c.get("method") == method)
        return len(self.call_history)

    def get_last_call(self, method: Optional[str] = None) -> Optional[Dict]:
        """Get the most recent call, optionally filtered by method."""
        if method:
            calls = [c for c in self.call_history if c.get("method") == method]
            return calls[-1] if calls else None
        return self.call_history[-1] if self.call_history else None

    def clear_history(self):
        """Clear call history."""
        self.call_history = []


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def fake_ollama_client():
    """Provides a FakeOllamaClient for testing."""
    return FakeOllamaClient()


@pytest.fixture
def fake_ollama_with_responses():
    """Factory fixture for FakeOllamaClient with custom responses."""

    def _create(responses: Dict[str, str]):
        return FakeOllamaClient(responses=responses)

    return _create


@pytest.fixture
def mock_ollama_client(fake_ollama_client):
    """Patches OllamaClient globally with FakeOllamaClient."""
    with patch("ollamatoolkit.client.OllamaClient", return_value=fake_ollama_client):
        yield fake_ollama_client


@pytest.fixture
def temp_workspace(tmp_path):
    """Provides a temporary workspace directory for file operations."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Create some test files
    (workspace / "test.txt").write_text("Hello, World!")
    (workspace / "data.json").write_text('{"key": "value"}')
    (workspace / "subdir").mkdir()
    (workspace / "subdir" / "nested.txt").write_text("Nested content")

    return workspace


@pytest.fixture
def sample_config(temp_workspace):
    """Provides a test configuration."""
    from ollamatoolkit.config import ToolkitConfig

    config = ToolkitConfig()
    config.tools.root_dir = str(temp_workspace)
    config.agent.model = "ollama/mock-model"
    config.agent.base_url = "http://localhost:11434"
    return config


@pytest.fixture
def sample_agent(fake_ollama_client):
    """Provides a SimpleAgent configured for testing."""
    from ollamatoolkit.agents.simple import SimpleAgent

    return SimpleAgent(
        name="test_agent",
        system_message="You are a helpful test assistant.",
        model_config={
            "model": "ollama/mock-model",
            "base_url": "http://localhost:11434",
            "temperature": 0.0,
        },
        mock_responses=[{"role": "assistant", "content": "Mock agent response."}],
    )


@pytest.fixture
def sample_file_tools(temp_workspace):
    """Provides FileTools configured for testing."""
    from ollamatoolkit.tools.files import FileTools

    return FileTools(root_dir=str(temp_workspace), read_only=False)


@pytest.fixture
def sample_conversation_memory():
    """Provides a ConversationMemory for testing."""
    from ollamatoolkit.agents.memory import ConversationMemory, MemoryConfig

    return ConversationMemory(MemoryConfig(max_messages=10, summarize_threshold=8))


# =============================================================================
# Real Model Fixtures
# =============================================================================


# Module-level cache for model registry
_model_registry: Optional[OllamaModelRegistry] = None


def get_model_registry() -> OllamaModelRegistry:
    """Get or create the shared model registry."""
    global _model_registry
    if _model_registry is None:
        _model_registry = OllamaModelRegistry()
    return _model_registry


@pytest.fixture(scope="session")
def ollama_models() -> OllamaModelRegistry:
    """
    Provides the OllamaModelRegistry for capability-based model selection.

    Usage:
        def test_vision(ollama_models):
            model = ollama_models.require_capability("vision")
            # Test with real model...

        def test_embedding(ollama_models):
            model = ollama_models.get_best_embedding_model()
            if not model:
                pytest.skip("No embedding model available")
    """
    return get_model_registry()


@pytest.fixture
def vision_model(ollama_models) -> str:
    """Get a vision-capable model or skip the test."""
    return ollama_models.require_capability("vision")


@pytest.fixture
def embedding_model(ollama_models) -> str:
    """Get an embedding-capable model or skip the test."""
    return ollama_models.require_capability("embedding")


@pytest.fixture
def tools_model(ollama_models) -> str:
    """Get a tools-capable model or skip the test."""
    return ollama_models.require_capability("tools")


@pytest.fixture
def reasoning_model(ollama_models) -> str:
    """Get a reasoning-capable model or skip the test."""
    return ollama_models.require_capability("reasoning")


@pytest.fixture
def chat_model(ollama_models) -> str:
    """Get the best chat model or skip the test."""
    model = ollama_models.get_best_chat_model()
    if not model:
        pytest.skip("No chat model available")
    return model


# =============================================================================
# Test Helpers
# =============================================================================


def make_mock_chat_response(content: str, tool_calls: Optional[List[Dict]] = None):
    """Create a mock chat response for testing."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message = MagicMock()
    response.choices[0].message.role = "assistant"
    response.choices[0].message.content = content
    response.choices[0].message.tool_calls = tool_calls
    return response


def make_mock_tool_call(name: str, arguments: Dict) -> Dict:
    """Create a mock tool call structure."""
    return {
        "id": f"call_{name}_{id(arguments)}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


# =============================================================================
# Markers and Hooks
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require Ollama)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "requires_gpu: marks tests that require GPU")
    config.addinivalue_line(
        "markers", "requires_vision: marks tests that require a vision-capable model"
    )
    config.addinivalue_line(
        "markers", "requires_embedding: marks tests that require an embedding model"
    )
    config.addinivalue_line(
        "markers", "requires_tools: marks tests that require a tools-capable model"
    )
    config.addinivalue_line(
        "markers", "requires_reasoning: marks tests that require a reasoning model"
    )


def pytest_collection_modifyitems(config, items):
    """
    Skip tests based on model capability markers.

    Tests marked with @pytest.mark.requires_vision will be skipped
    if no vision model is available.
    """
    registry = get_model_registry()

    capability_markers = {
        "requires_vision": "vision",
        "requires_embedding": "embedding",
        "requires_tools": "tools",
        "requires_reasoning": "reasoning",
    }

    for item in items:
        for marker_name, capability in capability_markers.items():
            if item.get_closest_marker(marker_name):
                if not registry.get_model_for_capability(capability):
                    skip_marker = pytest.mark.skip(
                        reason=f"No '{capability}' model available"
                    )
                    item.add_marker(skip_marker)
