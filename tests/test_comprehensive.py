# ./tests/test_comprehensive.py
"""
OllamaToolkit - Comprehensive Test Suite
========================================
Tests for all major components including:
- ModelSelector
- ConversationMemory
- AgentTeam
- Configuration loading
- DocumentProcessor auto-selection

Run with: pytest tests/test_comprehensive.py -v
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Test Configuration Loading
# =============================================================================


class TestConfigurationLoading:
    """Tests for configuration loading and defaults."""

    def test_default_config_has_all_sections(self):
        """Ensure ToolkitConfig has all expected sections with defaults."""
        from ollamatoolkit.config import ToolkitConfig

        cfg = ToolkitConfig()

        # Check all sections exist
        assert hasattr(cfg, "agent")
        assert hasattr(cfg, "tools")
        assert hasattr(cfg, "web")
        assert hasattr(cfg, "logging")
        assert hasattr(cfg, "dashboard")
        assert hasattr(cfg, "vision")
        assert hasattr(cfg, "vector")
        assert hasattr(cfg, "models")
        assert hasattr(cfg, "memory")
        assert hasattr(cfg, "document")
        assert hasattr(cfg, "benchmark")
        assert hasattr(cfg, "telemetry")

    def test_agent_config_defaults(self):
        """Test AgentConfig default values."""
        from ollamatoolkit.config import AgentConfig

        cfg = AgentConfig()

        assert cfg.temperature == 0.0
        assert cfg.base_url == "http://localhost:11434"
        assert cfg.history_limit == 50
        assert cfg.strict_mode is False

    def test_memory_config_defaults(self):
        """Test MemoryConfig default values."""
        from ollamatoolkit.config import MemoryConfig

        cfg = MemoryConfig()

        assert cfg.max_messages == 100
        assert cfg.summarize_threshold == 80
        assert cfg.storage_path is None
        assert cfg.auto_save is False

    def test_document_config_defaults(self):
        """Test DocumentConfig default values."""
        from ollamatoolkit.config import DocumentConfig

        cfg = DocumentConfig()

        assert cfg.default_vision_model is None  # Auto-select
        assert cfg.dpi == 150
        assert cfg.min_text_chars == 100
        assert cfg.clean_text is True
        assert cfg.save_images is False

    def test_benchmark_config_defaults(self):
        """Test BenchmarkConfig default values."""
        from ollamatoolkit.config import BenchmarkConfig

        cfg = BenchmarkConfig()

        assert cfg.unload_wait_time == 3.0
        assert cfg.max_unload_wait == 30.0
        assert cfg.unload_check_interval == 2.0

    def test_telemetry_config_defaults(self):
        """Test TelemetryConfig default values."""
        from ollamatoolkit.config import TelemetryConfig

        cfg = TelemetryConfig()

        assert cfg.enabled is True
        assert cfg.session_dir == "./logs/sessions"
        assert cfg.max_sessions == 100

    def test_load_from_json_file(self):
        """Test loading config from JSON file."""
        from ollamatoolkit.config import ToolkitConfig

        config_data = {
            "agent": {
                "model": "test/model:latest",
                "temperature": 0.5,
                "base_url": "http://custom:11434",
            },
            "memory": {"max_messages": 200, "summarize_threshold": 150},
            "document": {"dpi": 300, "clean_text": False},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            cfg = ToolkitConfig.load(temp_path)

            # Check loaded values
            assert cfg.agent.model == "test/model:latest"
            assert cfg.agent.temperature == 0.5
            assert cfg.agent.base_url == "http://custom:11434"
            assert cfg.memory.max_messages == 200
            assert cfg.memory.summarize_threshold == 150
            assert cfg.document.dpi == 300
            assert cfg.document.clean_text is False

            # Check defaults for unspecified values
            assert cfg.agent.history_limit == 50  # default
            assert cfg.memory.auto_save is False  # default
        finally:
            Path(temp_path).unlink()

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        from ollamatoolkit.config import ToolkitConfig

        cfg = ToolkitConfig()
        cfg_dict = cfg.to_dict()

        assert isinstance(cfg_dict, dict)
        assert "agent" in cfg_dict
        assert "memory" in cfg_dict
        assert "document" in cfg_dict
        assert "benchmark" in cfg_dict
        assert "telemetry" in cfg_dict

    def test_sample_config_matches_dataclasses(self):
        """Verify sample_config.json has all fields from dataclasses."""
        sample_path = Path(__file__).parent.parent / "sample_config.json"
        if not sample_path.exists():
            pytest.skip("sample_config.json not found")

        with open(sample_path) as f:
            sample = json.load(f)

        # Check all main sections exist
        expected_sections = [
            "agent",
            "models",
            "tools",
            "web",
            "vision",
            "vector",
            "memory",
            "document",
            "benchmark",
            "logging",
            "dashboard",
            "telemetry",
        ]
        for section in expected_sections:
            assert section in sample, f"Missing section: {section}"


# =============================================================================
# Test ModelSelector
# =============================================================================


class TestModelSelector:
    """Tests for ModelSelector utility."""

    def test_model_selector_initialization(self):
        """Test ModelSelector can be instantiated."""
        from ollamatoolkit.models.selector import ModelSelector

        with patch("ollama.Client") as mock_client:
            mock_client.return_value.list.return_value = MagicMock(models=[])
            selector = ModelSelector()
            assert selector.total_models == 0

    def test_model_info_size_parsing(self):
        """Test ModelInfo size parsing."""
        from ollamatoolkit.models.selector import ModelInfo

        # Test various size formats
        info_8b = ModelInfo(
            name="test",
            family="test",
            parameter_size="8.0B",
            parameter_count=0,
            capabilities=[],
            quantization="Q4",
            context_length=4096,
        )
        assert info_8b.size_in_billions == 8.0

        info_595m = ModelInfo(
            name="test",
            family="test",
            parameter_size="595.78M",
            parameter_count=0,
            capabilities=[],
            quantization="Q4",
            context_length=4096,
        )
        assert abs(info_595m.size_in_billions - 0.59578) < 0.001

    def test_task_type_enum(self):
        """Test TaskType enum values."""
        from ollamatoolkit.models.selector import TaskType

        assert TaskType.CHAT.value == "chat"
        assert TaskType.VISION.value == "vision"
        assert TaskType.EMBEDDING.value == "embedding"
        assert TaskType.REASONING.value == "reasoning"
        assert TaskType.CODE.value == "code"
        assert TaskType.OCR.value == "ocr"

    def test_model_selector_graceful_connection_failure(self):
        """Test ModelSelector handles connection failures gracefully."""
        from ollamatoolkit.models.selector import ModelSelector

        with patch("ollama.Client") as mock_client:
            mock_client.return_value.list.side_effect = Exception("Connection refused")
            selector = ModelSelector()

            # Should not raise, just log and return empty
            assert selector.total_models == 0
            assert selector.get_best_vision_model() is None


# =============================================================================
# Test ConversationMemory
# =============================================================================


class TestConversationMemory:
    """Tests for ConversationMemory."""

    def test_memory_initialization(self):
        """Test ConversationMemory can be instantiated."""
        from ollamatoolkit.agents.memory import ConversationMemory, MemoryConfig

        config = MemoryConfig(max_messages=50)
        memory = ConversationMemory(config)

        assert memory.config.max_messages == 50
        assert len(memory.messages) == 0

    def test_memory_add_message(self):
        """Test adding messages to memory."""
        from ollamatoolkit.agents.memory import ConversationMemory, MemoryConfig

        memory = ConversationMemory(MemoryConfig())

        memory.add_message("user", "Hello")
        memory.add_message("assistant", "Hi there!")

        assert len(memory.messages) == 2
        assert memory.messages[0]["role"] == "user"
        assert memory.messages[0]["content"] == "Hello"
        assert memory.messages[1]["role"] == "assistant"

    def test_memory_clear(self):
        """Test clearing memory."""
        from ollamatoolkit.agents.memory import ConversationMemory, MemoryConfig

        memory = ConversationMemory(MemoryConfig())
        memory.add_message("user", "Test")
        memory.add_message("assistant", "Response")

        assert len(memory.messages) == 2

        memory.clear()
        assert len(memory.messages) == 0

    def test_memory_get_context_messages(self):
        """Test getting context messages."""
        from ollamatoolkit.agents.memory import ConversationMemory, MemoryConfig

        memory = ConversationMemory(MemoryConfig(max_messages=5))

        for i in range(10):
            memory.add_message("user", f"Message {i}")
            memory.add_message("assistant", f"Response {i}")

        # Should return limited messages
        context = memory.get_context_messages(limit=4)
        assert len(context) <= 4


# =============================================================================
# Test AgentTeam
# =============================================================================


class TestAgentTeam:
    """Tests for AgentTeam orchestration."""

    def test_agent_role_creation(self):
        """Test creating AgentRole."""
        from ollamatoolkit.agents.team import AgentRole
        from unittest.mock import MagicMock

        mock_agent = MagicMock()
        role = AgentRole(
            name="researcher",
            agent=mock_agent,
            description="Research and gather information",
            capabilities=["search", "summarize"],
        )

        assert role.name == "researcher"
        assert "search" in role.capabilities
        assert role.agent is mock_agent

    def test_team_strategy_enum(self):
        """Test TeamStrategy enum values."""
        from ollamatoolkit.agents.team import TeamStrategy

        assert TeamStrategy.SEQUENTIAL.value == "sequential"
        assert TeamStrategy.PARALLEL.value == "parallel"
        assert TeamStrategy.SUPERVISOR.value == "supervisor"
        assert TeamStrategy.ROUND_ROBIN.value == "round_robin"

    def test_team_result_structure(self):
        """Test TeamResult dataclass."""
        from ollamatoolkit.agents.team import TeamResult, TeamStrategy

        result = TeamResult(
            final_response="Final answer",
            agent_responses={"agent1": "output1", "agent2": "output2"},
            turns_used=5,
            strategy=TeamStrategy.SEQUENTIAL,
        )

        assert result.final_response == "Final answer"
        assert len(result.agent_responses) == 2
        assert result.turns_used == 5


# =============================================================================
# Test SimpleAgent with Memory Integration
# =============================================================================


class TestSimpleAgentMemory:
    """Tests for SimpleAgent memory integration."""

    def test_agent_without_memory(self):
        """Test SimpleAgent works without memory (backwards compatible)."""
        from ollamatoolkit.agents.simple import SimpleAgent

        agent = SimpleAgent(
            name="test",
            system_message="You are helpful.",
            model_config={"model": "test/model"},
            history_limit=50,
        )

        assert agent.memory is None
        assert len(agent.history) == 1  # System message
        assert agent.history[0]["role"] == "system"

    def test_agent_with_memory(self):
        """Test SimpleAgent with ConversationMemory."""
        from ollamatoolkit.agents.simple import SimpleAgent
        from ollamatoolkit.agents.memory import ConversationMemory, MemoryConfig

        memory = ConversationMemory(MemoryConfig(max_messages=100))

        agent = SimpleAgent(
            name="test",
            system_message="You are helpful.",
            model_config={"model": "test/model"},
            memory=memory,
        )

        assert agent.memory is memory
        # History should be memory's messages
        assert agent.history is memory.messages


# =============================================================================
# Test DocumentProcessor Auto-Selection
# =============================================================================


class TestDocumentProcessorAutoSelection:
    """Tests for DocumentProcessor model auto-selection."""

    def test_document_processor_init(self):
        """Test DocumentProcessor initialization."""
        from ollamatoolkit.tools.document import DocumentProcessor

        processor = DocumentProcessor(base_url="http://localhost:11434")

        assert processor.base_url == "http://localhost:11434"
        assert processor._vision_models_cache is None

    def test_get_vision_models_empty(self):
        """Test getting vision models when none available."""
        from ollamatoolkit.tools.document import DocumentProcessor

        processor = DocumentProcessor()

        with patch.object(processor, "_get_vision_models", return_value=[]):
            result = processor._select_vision_model()
            assert result is None

    def test_get_vision_models_selection(self):
        """Test vision model selection returns first available."""
        from ollamatoolkit.tools.document import DocumentProcessor

        processor = DocumentProcessor()

        with patch.object(
            processor, "_get_vision_models", return_value=["model1", "model2"]
        ):
            result = processor._select_vision_model()
            assert result == "model1"


# =============================================================================
# Test Utility Functions
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_retry_decorator(self):
        """Test retry decorator."""
        from ollamatoolkit.utils import retry

        call_count = 0

        @retry(attempts=3, delay=0.01)
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"

        result = failing_function()
        assert result == "success"
        assert call_count == 3

    def test_clean_json_response(self):
        """Test JSON cleaning utility."""
        from ollamatoolkit.common.utils import clean_json_response

        # Test with markdown code block
        raw = '```json\n{"key": "value"}\n```'
        cleaned = clean_json_response(raw)
        assert "```" not in cleaned
        assert '"key"' in cleaned

        # Test with plain JSON
        plain = '{"key": "value"}'
        result = clean_json_response(plain)
        assert result == plain


# =============================================================================
# Test Imports
# =============================================================================


class TestImports:
    """Tests for package imports."""

    def test_main_package_imports(self):
        """Test main package exports are importable."""
        from ollamatoolkit import (
            SimpleAgent,
            AgentTeam,
            OllamaClient,
            ToolkitConfig,
            ModelSelector,
            ConversationMemory,
        )

        assert SimpleAgent is not None
        assert AgentTeam is not None
        assert OllamaClient is not None
        assert ToolkitConfig is not None
        assert ModelSelector is not None
        assert ConversationMemory is not None

    def test_config_imports(self):
        """Test all configs are importable."""
        from ollamatoolkit.config import (
            AgentConfig,
            ToolConfig,
            WebToolConfig,
            VisionConfig,
            VectorConfig,
            MemoryConfig,
            DocumentConfig,
            BenchmarkConfig,
            TelemetryConfig,
            ModelsConfig,
        )

        assert all(
            [
                AgentConfig,
                ToolConfig,
                WebToolConfig,
                VisionConfig,
                VectorConfig,
                MemoryConfig,
                DocumentConfig,
                BenchmarkConfig,
                TelemetryConfig,
                ModelsConfig,
            ]
        )

    def test_agents_package_imports(self):
        """Test agents package exports."""
        from ollamatoolkit.agents import (
            SimpleAgent,
            RoleAgent,
            AgentTeam,
            AgentRole,
            TeamStrategy,
            ConversationMemory,
            MemoryConfig,
        )

        assert all(
            [
                SimpleAgent,
                RoleAgent,
                AgentTeam,
                AgentRole,
                TeamStrategy,
                ConversationMemory,
                MemoryConfig,
            ]
        )

    def test_models_package_imports(self):
        """Test models package exports."""
        from ollamatoolkit.models import (
            ModelSelector,
            ModelInfo,
            TaskType,
            ModelInspector,
        )

        assert all([ModelSelector, ModelInfo, TaskType, ModelInspector])
