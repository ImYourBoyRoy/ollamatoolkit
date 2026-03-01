# ./tests/test_config.py
"""
Tests for ToolkitConfig and related configuration classes.
"""

import json
import tempfile
from pathlib import Path


from ollamatoolkit.config import (
    AgentConfig,
    ToolConfig,
    WebToolConfig,
    VisionConfig,
    VectorConfig,
    ToolkitConfig,
)


class TestAgentConfig:
    """Test AgentConfig defaults and overrides."""

    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.model.startswith("ollama/")
        assert cfg.temperature == 0.0
        assert cfg.base_url == "http://localhost:11434"
        assert cfg.strict_mode is False

    def test_custom_values(self):
        cfg = AgentConfig(model="ollama/llama2", temperature=0.5)
        assert cfg.model == "ollama/llama2"
        assert cfg.temperature == 0.5


class TestToolConfig:
    """Test ToolConfig defaults."""

    def test_defaults(self):
        cfg = ToolConfig()
        assert cfg.root_dir == "./"
        assert cfg.read_only is False
        assert "files" in cfg.allowed_tools
        assert "vision" in cfg.allowed_tools

    def test_custom_tools(self):
        cfg = ToolConfig(custom_tools=["my_tool"])
        assert "my_tool" in cfg.custom_tools


class TestToolkitConfig:
    """Test root ToolkitConfig."""

    def test_defaults(self):
        cfg = ToolkitConfig()
        assert isinstance(cfg.agent, AgentConfig)
        assert isinstance(cfg.tools, ToolConfig)
        assert isinstance(cfg.web, WebToolConfig)
        assert isinstance(cfg.vision, VisionConfig)

    def test_to_dict(self):
        cfg = ToolkitConfig()
        data = cfg.to_dict()
        assert "agent" in data
        assert "tools" in data
        assert data["agent"]["model"].startswith("ollama/")

    def test_generate_sample(self):
        cfg = ToolkitConfig.generate_sample()
        assert cfg.agent.model.startswith("ollama/")

    def test_load_from_file(self):
        # Create temp config
        config_data = {
            "agent": {"model": "ollama/custom", "temperature": 0.7},
            "tools": {"root_dir": "/custom/path"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            f.flush()

            cfg = ToolkitConfig.load(f.name)
            assert cfg.agent.model == "ollama/custom"
            assert cfg.agent.temperature == 0.7
            assert cfg.tools.root_dir == "/custom/path"
            # Other values should be defaults
            assert cfg.web.timeout == WebToolConfig().timeout

        Path(f.name).unlink()

    def test_load_nonexistent_returns_defaults(self):
        cfg = ToolkitConfig.load("/nonexistent/config.json")
        assert cfg.agent.model.startswith("ollama/")

    def test_save_sample(self):
        cfg = ToolkitConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            cfg.save(str(path))

            assert path.exists()
            data = json.loads(path.read_text())
            assert data["agent"]["model"].startswith("ollama/")


class TestVisionConfig:
    """Test VisionConfig."""

    def test_defaults(self):
        cfg = VisionConfig()
        assert "qwen" in cfg.model.lower() or "ollama" in cfg.model.lower()
        assert cfg.enable_grounding is True
        assert cfg.use_chat_api is True


class TestVectorConfig:
    """Test VectorConfig."""

    def test_defaults(self):
        cfg = VectorConfig()
        assert cfg.chunk_size == 500
        assert cfg.chunk_overlap == 50
        assert "nomic" in cfg.embedding_model.lower()
